"""
================================================================================
 SmartCross — making the collaborative engine pull its weight
================================================================================

 The served item-item CF sums LIFT rows. Lift divides by each product's base
 rate, so it ranks rare products first by construction: a 40x lift on a product
 nobody takes outranks a 2x lift on one half the bank takes. That is why CF
 scored below the no-ML popularity list. This file compares it with two
 collaborative estimators that rank by probability instead, and tunes every
 free parameter on a VALIDATION month the test period never sees.

     python cf_experiment.py

 Engines compared (all collaborative: they use holdings only, no demographics)
   lift-sum       current: sum of lift rows of held products
   cooc           mean over held products of P(add j | holds i), smoothed to
                  the base rate with strength m
   pattern        P(add j | the customer's exact portfolio), shrunk towards
                  cooc with strength m -- "customers with your portfolio added"

 Protocol
   train  = months < cut-1      fit CF tables + a HistGB content model
   valid  = month  cut-1        choose CF variant, m and alpha
   test   = months >= cut       report once, with the choices frozen
================================================================================
"""

import json
import os
import time

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import f1_score

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.normpath(os.path.join(HERE, ".."))
NPZ = os.path.join(ROOT, "partB_santander", "data", "sant_events.npz")
OUT = os.path.join(HERE, "outputs")

TEST_MONTHS, MIN_EVENTS, SEED = 3, 50, 42
HGB_PARAMS = dict(max_iter=300, max_depth=10, learning_rate=0.1,
                  l2_regularization=1.0, early_stopping=True,
                  validation_fraction=0.1, random_state=SEED)
M_GRID = [0, 1, 5, 20, 50, 200]
ALPHA_GRID = np.round(np.linspace(0, 1, 21), 2)


# ------------------------------------------------------------- CF estimators
def fit_cf(H, y, K):
    """Every table a CF variant needs, from training rows only."""
    base = np.bincount(y, minlength=K).astype(float)
    base /= base.sum()
    onehot = np.eye(K)[y]
    co = H.T @ onehot                                   # [24 held x K added]
    n_held = H.sum(axis=0)[:, None]                     # events where i was held
    keys = pack(H)
    uk, inv = np.unique(keys, return_inverse=True)
    pat_counts = np.zeros((len(uk), K))
    np.add.at(pat_counts, (inv, y), 1.0)
    return {"base": base, "co": co, "n_held": n_held,
            "pat_keys": uk, "pat_counts": pat_counts}


def pack(H):
    return (H.astype(np.int64) * (1 << np.arange(H.shape[1], dtype=np.int64))).sum(axis=1)


def cf_lift_sum(T, H):
    lift = (T["co"] / np.clip(T["n_held"], 1, None)) / np.clip(T["base"][None, :], 1e-9, None)
    s = H @ lift
    rs = s.sum(axis=1, keepdims=True)
    return np.where(rs > 0, s / np.clip(rs, 1e-12, None), T["base"][None, :])


def cf_cooc(T, H, m):
    cond = (T["co"] + m * T["base"][None, :]) / (T["n_held"] + m)   # smoothed P(j | i)
    cond = np.where(T["n_held"] > 0, cond, T["base"][None, :])
    n = H.sum(axis=1, keepdims=True)
    s = (H @ cond) / np.clip(n, 1, None)
    return np.where(n > 0, s, T["base"][None, :])


def cf_pattern(T, H, m, m_back):
    back = cf_cooc(T, H, m_back)
    keys = pack(H)
    pos = np.searchsorted(T["pat_keys"], keys)
    pos = np.clip(pos, 0, len(T["pat_keys"]) - 1)
    hit = T["pat_keys"][pos] == keys
    counts = np.where(hit[:, None], T["pat_counts"][pos], 0.0)
    n = counts.sum(axis=1, keepdims=True)
    return (counts + m * back) / (n + m) if m > 0 else np.where(n > 0, counts / np.clip(n, 1, None), back)


# ------------------------------------------------------------------ metrics
def mask_held(p, H):
    q = np.where(H > 0, 0.0, p)
    s = q.sum(axis=1, keepdims=True)
    return np.where(s > 0, q / np.clip(s, 1e-12, None), p)


def evaluate(p, y, K):
    true = p[np.arange(len(y)), y][:, None]
    r = (p > true).sum(axis=1) + (p == true).sum(axis=1) - 1     # ties count against the engine
    return {"top1": float((r < 1).mean()), "top3": float((r < 3).mean()),
            "map7": float(np.where(r < 7, 1 / (r + 1), 0).mean()),
            "macro_f1": float(f1_score(y, p.argmax(1), average="macro",
                                       zero_division=0))}


def line(name, r):
    print(f"  {name:38} top-1 {r['top1']:.3f}  top-3 {r['top3']:.3f}  "
          f"MAP@7 {r['map7']:.3f}  macro F1 {r['macro_f1']:.4f}")


def main():
    t0 = time.time()
    d = np.load(NPZ, allow_pickle=True)
    Hall = d["X_prev"].astype(np.float64)
    X = np.hstack([d["X_prev"].astype(np.float32), d["X_demo"].astype(np.float32)])
    y, mi = d["y"].astype(int), d["mi"].astype(int)
    names = [str(s) for s in d["names"]]
    keep = np.where(np.bincount(y, minlength=len(names)) >= MIN_EVENTS)[0]
    mask = np.isin(y, keep)
    remap = np.full(len(names), -1)
    remap[keep] = np.arange(len(keep))
    y, X, Hall, mi = remap[y[mask]], X[mask], Hall[mask], mi[mask]
    K = len(keep)
    Hk = Hall[:, keep]                         # held flags in product-column space, for masking

    cut = mi.max() - TEST_MONTHS + 1
    val_m = cut - 1
    fit_rows, val_rows, tr_rows, te_rows = mi < val_m, mi == val_m, mi < cut, mi >= cut
    print(f"fit months <{val_m}: {fit_rows.sum():,} · valid month {val_m}: {val_rows.sum():,} · "
          f"test months >= {cut}: {te_rows.sum():,}")
    res = {"protocol": {"fit_months_lt": int(val_m), "valid_month": int(val_m),
                        "test_months_ge": int(cut)}}

    # -------------------------------------------------- 1. CF on validation
    print("\n[1] Collaborative variants on the VALIDATION month (held masked)")
    T = fit_cf(Hall[fit_rows], y[fit_rows], K)
    Hv, yv, Hkv = Hall[val_rows], y[val_rows], Hk[val_rows]
    grid = []
    r = evaluate(mask_held(cf_lift_sum(T, Hv), Hkv), yv, K)
    line("lift-sum (current)", r)
    grid.append({"variant": "lift-sum", **r})
    r = evaluate(mask_held(np.tile(T["base"], (len(yv), 1)), Hkv), yv, K)
    line("popularity (reference)", r)
    for m in M_GRID:
        r = evaluate(mask_held(cf_cooc(T, Hv, m), Hkv), yv, K)
        line(f"cooc m={m}", r)
        grid.append({"variant": "cooc", "m": m, **r})
    for m in M_GRID:
        r = evaluate(mask_held(cf_pattern(T, Hv, m, 5), Hkv), yv, K)
        line(f"pattern m={m} (back-off cooc m=5)", r)
        grid.append({"variant": "pattern", "m": m, "m_back": 5, **r})
    best = max(grid, key=lambda g: (g["map7"], g["top3"]))
    print(f"  -> chosen on validation MAP@7: {best}")
    res["cf_validation_grid"] = grid
    res["cf_chosen"] = best

    def cf_best(T_, H_):
        if best["variant"] == "pattern":
            return cf_pattern(T_, H_, best["m"], best["m_back"])
        if best["variant"] == "cooc":
            return cf_cooc(T_, H_, best["m"])
        return cf_lift_sum(T_, H_)

    # ----------------------------------------- 2. alpha on validation
    print("\n[2] Hybrid weight alpha on the VALIDATION month")
    t1 = time.time()
    hgb_v = HistGradientBoostingClassifier(**HGB_PARAMS).fit(X[fit_rows], y[fit_rows])
    pc_v = hgb_v.predict_proba(X[val_rows])
    print(f"  (content model for validation trained in {time.time()-t1:.0f}s)")
    pcf_v = cf_best(T, Hv)
    pcf_old_v = cf_lift_sum(T, Hv)
    sweep = []
    for a in ALPHA_GRID:
        rn = evaluate(mask_held(a * pc_v + (1 - a) * pcf_v, Hkv), yv, K)
        ro = evaluate(mask_held(a * pc_v + (1 - a) * pcf_old_v, Hkv), yv, K)
        sweep.append({"alpha": float(a), "new_cf": rn, "old_cf": ro})
        print(f"  alpha {a:.2f}  new-CF hybrid top-3 {rn['top3']:.3f} MAP@7 {rn['map7']:.3f} F1 {rn['macro_f1']:.4f}"
              f"   | old-CF hybrid top-3 {ro['top3']:.3f} MAP@7 {ro['map7']:.3f}")
    a_best = max(sweep, key=lambda s: (s["new_cf"]["map7"], s["new_cf"]["top3"]))["alpha"]
    print(f"  -> alpha chosen on validation: {a_best}")
    res["alpha_validation"] = sweep
    res["alpha_chosen"] = a_best

    # ------------------------------------------- 3. frozen, scored once on test
    print("\n[3] TEST months, every choice frozen from validation")
    T2 = fit_cf(Hall[tr_rows], y[tr_rows], K)
    t1 = time.time()
    hgb = HistGradientBoostingClassifier(**HGB_PARAMS).fit(X[tr_rows], y[tr_rows])
    pc = hgb.predict_proba(X[te_rows])
    print(f"  (content model trained in {time.time()-t1:.0f}s)")
    Ht, yt, Hkt = Hall[te_rows], y[te_rows], Hk[te_rows]
    pop = np.tile(T2["base"], (len(yt), 1))
    old_cf, new_cf = cf_lift_sum(T2, Ht), cf_best(T2, Ht)
    rows = {
        "Popularity (no ML)": pop,
        "Item-item CF, lift-sum (old)": old_cf,
        f"Collaborative, {best['variant']} (new)": new_cf,
        "Content (HistGB)": pc,
        "Hybrid old: 0.5 content + 0.5 old CF": 0.5 * pc + 0.5 * old_cf,
        f"Hybrid new: {a_best} content + {1-a_best:.2f} new CF": a_best * pc + (1 - a_best) * new_cf,
    }
    # ------------------------------- 4. feature-augmentation hybrid (Burke)
    #  CF's two estimates become inputs to the content model, so the booster
    #  learns per customer when peers are informative. For every training month
    #  the CF features come from strictly EARLIER months, never its own -- a
    #  row's own outcome must not leak into its features.
    print("\n[4] Feature-augmentation hybrid: CF scores as inputs to the content model")
    cooc_m = max((g for g in grid if g["variant"] == "cooc"), key=lambda g: g["map7"])["m"]
    pat_m = max((g for g in grid if g["variant"] == "pattern"), key=lambda g: g["map7"])["m"]
    print(f"  CF features: cooc m={cooc_m}, pattern m={pat_m} (both chosen on validation)")

    def cf_feats(T_, H_):
        return np.hstack([cf_cooc(T_, H_, cooc_m), cf_pattern(T_, H_, pat_m, 5)]).astype(np.float32)

    F = np.zeros((len(y), 2 * K), dtype=np.float32)
    for m in np.unique(mi[tr_rows]):
        rows_m, past = mi == m, mi < m
        if past.sum() == 0:
            F[rows_m] = 1.0 / K
        else:
            F[rows_m] = cf_feats(fit_cf(Hall[past], y[past], K), Hall[rows_m])
    F[te_rows] = cf_feats(T2, Ht)                 # test: training months only, as served
    XA = np.hstack([X, F])

    t1 = time.time()
    aug_v = HistGradientBoostingClassifier(**HGB_PARAMS).fit(XA[fit_rows], y[fit_rows])
    r_aug_v = evaluate(mask_held(aug_v.predict_proba(XA[val_rows]), Hkv), yv, K)
    r_con_v = evaluate(mask_held(pc_v, Hkv), yv, K)
    print(f"  (validation model trained in {time.time()-t1:.0f}s)")
    line("validation · content only", r_con_v)
    line("validation · feature-augmented hybrid", r_aug_v)
    res["augmented_validation"] = {"content": r_con_v, "augmented": r_aug_v}

    t1 = time.time()
    aug = HistGradientBoostingClassifier(**HGB_PARAMS).fit(XA[tr_rows], y[tr_rows])
    pa = aug.predict_proba(XA[te_rows])
    print(f"  (test model trained in {time.time()-t1:.0f}s)")
    rows["Feature-augmented hybrid (content + CF features)"] = pa

    res["test"] = {}
    for name, p in rows.items():
        r = evaluate(mask_held(p, Hkt), yt, K)
        line(name, r)
        res["test"][name] = r

    # paired bootstrap on test rows: is the augmented hybrid really better?
    def hits(p, k):
        q = mask_held(p, Hkt)
        return ((q > q[np.arange(len(yt)), yt][:, None]).sum(axis=1) < k).astype(float)
    g = np.random.default_rng(SEED + 1)
    comp = {}
    for k in (1, 3):
        diff = hits(pa, k) - hits(pc, k)
        boots = np.array([diff[g.integers(0, len(diff), len(diff))].mean() for _ in range(500)])
        comp[f"top{k}"] = {"diff": float(diff.mean()),
                           "ci": [float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))]}
        print(f"  augmented - content, top-{k}: {diff.mean():+.4f}  "
              f"95% CI [{comp[f'top{k}']['ci'][0]:+.4f}, {comp[f'top{k}']['ci'][1]:+.4f}]")
    res["augmented_vs_content_test"] = comp
    res["runtime_seconds"] = round(time.time() - t0)
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, "cf_experiment.json"), "w") as f:
        json.dump(res, f, indent=2)
    print(f"\nDONE in {time.time()-t0:.0f}s -> defense/outputs/cf_experiment.json")


if __name__ == "__main__":
    main()
