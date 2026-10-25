"""
================================================================================
 SmartCross — the three "not done yet" levers, tried properly
================================================================================

     python improve_experiment.py      # ~30-40 min, writes outputs/improve_experiment.json

 Levers (README "What is not done yet")
   1. holdings history as features   (X_hist from phase2_4a_build.py: 48 columns)
   2. per-class decision thresholds  (one multiplier per product)
   3. a hyperparameter search on the booster

 Protocol, fixed BEFORE any test month is scored
   fit   = months < 13          every candidate is trained here
   valid = month 13 (2016-02)   every choice is made here
   test  = months >= 14         scored ONCE, final choice vs the served model

   Selection criterion: validation top-3 (the operative metric); macro F1
   breaks ties within 0.1 points. Thresholds are kept only if they raise
   validation macro F1 while costing at most 0.2 points of validation top-3.
   Metrics hold products masked and count ties against the engine, as
   everywhere else in defense/.
================================================================================
"""

import json
import os
import sys
import time

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import f1_score

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.normpath(os.path.join(HERE, ".."))
sys.path.insert(0, os.path.join(ROOT, "web", "backend"))
import peers                                                      # noqa: E402

NPZ = os.path.join(ROOT, "partB_santander", "data", "sant_events.npz")
OUT = os.path.join(HERE, "outputs", "improve_experiment.json")
TEST_MONTHS, MIN_EVENTS, SEED = 3, 50, 42
SERVED = dict(max_iter=300, max_depth=10, learning_rate=0.1, l2_regularization=1.0)
GRID = [
    ("served settings", SERVED),
    ("slower, longer", dict(max_iter=800, max_depth=10, learning_rate=0.05, l2_regularization=1.0)),
    ("wider trees", dict(max_iter=300, max_depth=None, max_leaf_nodes=63, learning_rate=0.1, l2_regularization=1.0)),
    ("wider + slower", dict(max_iter=800, max_depth=None, max_leaf_nodes=63, learning_rate=0.05, l2_regularization=1.0)),
    ("bigger leaves", dict(max_iter=300, max_depth=10, learning_rate=0.1, l2_regularization=1.0, min_samples_leaf=100)),
    ("less regularised", dict(max_iter=300, max_depth=10, learning_rate=0.1, l2_regularization=0.0)),
]
BOOT = 500


def hgb(params):
    return HistGradientBoostingClassifier(early_stopping=True, validation_fraction=0.1,
                                          random_state=SEED, **params)


def mask_held(p, held):
    q = np.where(held > 0, 0.0, p)
    s = q.sum(axis=1, keepdims=True)
    return np.where(s > 0, q / np.clip(s, 1e-12, None), p)


def rows(p, y):
    t = p[np.arange(len(y)), y][:, None]
    r = (p > t).sum(axis=1) + (p == t).sum(axis=1) - 1
    return {"top1": (r < 1).astype(float), "top3": (r < 3).astype(float),
            "map7": np.where(r < 7, 1 / (r + 1), 0.0)}


def evaluate(p, y):
    r = rows(p, y)
    return {"top1": float(r["top1"].mean()), "top3": float(r["top3"].mean()),
            "map7": float(r["map7"].mean()),
            "macro_f1": float(f1_score(y, p.argmax(1), average="macro", zero_division=0))}


def line(name, r):
    print(f"  {name:44} top-1 {r['top1']:.4f}  top-3 {r['top3']:.4f}  MAP@7 {r['map7']:.4f}  "
          f"macro F1 {r['macro_f1']:.4f}", flush=True)


def better(a, b):
    """a beats b under the declared criterion"""
    if a["top3"] > b["top3"] + 0.001:
        return True
    if abs(a["top3"] - b["top3"]) <= 0.001:
        return a["macro_f1"] > b["macro_f1"]
    return False


def tune_thresholds(p, y, K, rounds=3):
    """Coordinate ascent on one multiplier per product, maximising macro F1."""
    w = np.ones(K)
    grid = np.array([0.5, 0.7, 0.85, 1.0, 1.2, 1.5, 2.0, 3.0, 5.0])
    base = f1_score(y, (p * w).argmax(1), average="macro", zero_division=0)
    for _ in range(rounds):
        for j in range(K):
            best_v, best_s = w[j], base
            for g in grid:
                w2 = w.copy()
                w2[j] = g
                s = f1_score(y, (p * w2).argmax(1), average="macro", zero_division=0)
                if s > best_s + 1e-6:
                    best_v, best_s = g, s
            w[j], base = best_v, best_s
    return w


def apply_w(p, w):
    q = p * w[None, :]
    return q / np.clip(q.sum(axis=1, keepdims=True), 1e-12, None)


def paired(a, b, y):
    ra, rb = rows(a, y), rows(b, y)
    g = np.random.default_rng(SEED + 1)
    out = {}
    for k in ("top1", "top3"):
        d = ra[k] - rb[k]
        bs = [d[g.integers(0, len(d), len(d))].mean() for _ in range(BOOT)]
        out[k] = {"diff": float(d.mean()), "ci": [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))]}
    pa, pb = a.argmax(1), b.argmax(1)
    fs = []
    for _ in range(200):
        i = g.integers(0, len(y), len(y))
        fs.append(f1_score(y[i], pa[i], average="macro", zero_division=0) -
                  f1_score(y[i], pb[i], average="macro", zero_division=0))
    out["macro_f1"] = {"diff": float(f1_score(y, pa, average="macro", zero_division=0) -
                                     f1_score(y, pb, average="macro", zero_division=0)),
                       "ci": [float(np.percentile(fs, 2.5)), float(np.percentile(fs, 97.5))]}
    return out


def main():
    t0 = time.time()
    d = np.load(NPZ, allow_pickle=True)
    if "X_hist" not in d.files:
        sys.exit("[X] sant_events.npz has no X_hist -- run phase2_4a_build.py first")
    Xp = d["X_prev"].astype(np.float32)
    Xd = d["X_demo"].astype(np.float32)
    Xh = d["X_hist"].astype(np.float32)
    y, mi = d["y"].astype(int), d["mi"].astype(int)
    names = [str(s) for s in d["names"]]
    keep = np.where(np.bincount(y, minlength=len(names)) >= MIN_EVENTS)[0]
    m = np.isin(y, keep)
    remap = np.full(len(names), -1)
    remap[keep] = np.arange(len(keep))
    y, Xp, Xd, Xh, mi = remap[y[m]], Xp[m], Xd[m], Xh[m], mi[m]
    K = len(keep)
    held = Xp[:, keep]

    cut = mi.max() - TEST_MONTHS + 1
    val_m = cut - 1
    fit, val, tr, te = mi < val_m, mi == val_m, mi < cut, mi >= cut
    print(f"fit {fit.sum():,} · valid {val.sum():,} · train {tr.sum():,} · test {te.sum():,}")

    F = peers.expanding_features(Xp, y, mi, K, np.unique(mi[tr]))
    F[te] = peers.features(peers.fit(Xp[tr], y[tr], K), Xp[te])
    X_served = np.hstack([Xp, Xd, F])                 # 79, as served
    X_hist = np.hstack([Xp, Xd, F, Xh])               # 127, with history
    res = {"protocol": {"fit_months_lt": int(val_m), "valid_month": int(val_m), "test_months_ge": int(cut),
                        "criterion": "validation top-3, macro F1 breaks ties within 0.1 pts"}}
    yv, hv = y[val], held[val]

    # ---------------------------------------------------- lever 1: history
    print("\n[1] Holdings history -- validation month")
    feat_sets = {"served (79)": X_served, "+ history (127)": X_hist}
    val_res = {}
    for nm, X in feat_sets.items():
        t1 = time.time()
        p = mask_held(hgb(SERVED).fit(X[fit], y[fit]).predict_proba(X[val]), hv)
        val_res[nm] = evaluate(p, yv)
        line(f"{nm}  ({time.time()-t1:.0f}s)", val_res[nm])
    use_hist = better(val_res["+ history (127)"], val_res["served (79)"])
    print(f"  -> history {'KEPT' if use_hist else 'REJECTED'} on validation")
    res["history_validation"] = val_res
    res["history_kept"] = use_hist
    X = X_hist if use_hist else X_served

    # ---------------------------------------------- lever 3: hyperparameters
    print("\n[3] Hyperparameter search -- validation month")
    grid_res, best_name, best_r, best_p = [], None, None, None
    for nm, params in GRID:
        t1 = time.time()
        model = hgb(params).fit(X[fit], y[fit])
        p = mask_held(model.predict_proba(X[val]), hv)
        r = evaluate(p, yv)
        r["n_iter"] = int(model.n_iter_)
        grid_res.append({"name": nm, "params": params, **r})
        line(f"{nm} (rounds {model.n_iter_}, {time.time()-t1:.0f}s)", r)
        if best_r is None or better(r, best_r):
            best_name, best_r, best_p = nm, r, p
    best_params = dict(GRID)[best_name]
    print(f"  -> chosen: {best_name}")
    res["grid"] = grid_res
    res["chosen_params"] = {"name": best_name, **best_params}

    # ---------------------------------------------- lever 2: thresholds
    print("\n[2] Per-class thresholds -- tuned and judged on validation")
    w = tune_thresholds(best_p, yv, K)
    r_w = evaluate(apply_w(best_p, w), yv)
    line("chosen model", best_r)
    line("chosen model + thresholds", r_w)
    use_w = (r_w["macro_f1"] > best_r["macro_f1"] + 0.002) and (r_w["top3"] >= best_r["top3"] - 0.002)
    print(f"  -> thresholds {'KEPT' if use_w else 'REJECTED'}")
    res["thresholds"] = {"weights": {str(names[keep[j]]): float(w[j]) for j in range(K)},
                         "validation_without": best_r, "validation_with": r_w, "kept": use_w}

    # ---------------------------------------------- final: test, scored once
    print("\n[4] TEST months, scored once -- final choice vs the served model")
    ht, yt = held[te], y[te]
    served = mask_held(hgb(SERVED).fit(X_served[tr], y[tr]).predict_proba(X_served[te]), ht)
    final_model = hgb(best_params).fit(X[tr], y[tr])
    final = mask_held(final_model.predict_proba(X[te]), ht)
    if use_w:
        final = apply_w(final, w)
    r_served, r_final = evaluate(served, yt), evaluate(final, yt)
    line("served model (fused, 79 inputs)", r_served)
    line("improved model", r_final)
    cmp = paired(final, served, yt)
    for k in ("top1", "top3", "macro_f1"):
        c = cmp[k]
        real = c["ci"][0] > 0 or c["ci"][1] < 0
        print(f"  improved - served, {k:8}: {c['diff']:+.4f}  95% CI [{c['ci'][0]:+.4f}, {c['ci'][1]:+.4f}]"
              f"  {'real' if real else 'within noise'}")
    res["test"] = {"served": r_served, "improved": r_final, "paired": cmp,
                   "improved_n_iter": int(final_model.n_iter_)}
    res["runtime_seconds"] = round(time.time() - t0)
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    json.dump(res, open(OUT, "w"), indent=2)
    print(f"\nDONE in {time.time()-t0:.0f}s -> {OUT}")


if __name__ == "__main__":
    main()
