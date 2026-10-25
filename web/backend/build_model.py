"""
================================================================================
 SmartCross API — build the served model
================================================================================

 Trains the model the API serves and writes everything the endpoint needs into
 one artefact. Run this once; the API then starts in about a second.

     python build_model.py

 What it produces (model/smartcross.pkl):
     clf          the FUSED hybrid: gradient boosting over 35 profile features,
                  44 peer features from the collaborative engine, and 48
                  holdings-history features (127 inputs)
     versions     python / numpy / scikit-learn that wrote this file
     peers        the collaborative engine's tables (see peers.py)
     products     the product names the classifier's columns map to
     base_rate    each product's share of observed acquisitions
     lift         item-item lift matrix, P(add j | holds i) / base(j) -- used
                  for EXPLANATIONS and the Evidence view, not for ranking
     metrics      the held-out scores, so the API can report them honestly
     engines      EVERY engine scored on that same split, measured here

 Every table is computed from TRAINING MONTHS ONLY, the same as the
 classifier. Computing them over all months would leak the test period into
 something the API serves as if it were learned beforehand.

 ON THE HYBRID
 The first hybrid was a fixed weighted blend, 0.5 content + 0.5 lift-CF. It
 measured slightly WORSE than content alone, because lift-CF ranked by surprise
 rather than likelihood (31% top-3, below popularity). The collaborative engine
 now ranks by probability (peers.py) and is fused by feature augmentation: its
 scores become inputs the booster learns to trust per customer. The choice
 between fusions was made on a validation month (defense/cf_experiment.py);
 the test months below are scored once, with every choice frozen.

 ON THE ENGINE TABLE
 Every row the console shows is measured by this file, on one split, in one
 run, scored the way the API serves (products already held are never
 recommended, for every engine alike). Nothing is quoted from an earlier
 experiment. If XGBoost is not installed the row is absent rather than filled
 in from memory -- a number the running code did not produce has no business
 being on screen.
================================================================================
"""

import os
import sys
import json
import time

import numpy as np

try:
    import joblib
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.metrics import f1_score, accuracy_score
except ImportError:
    print("\n[X] pip install scikit-learn joblib numpy\n")
    sys.exit(1)

import peers

HERE = os.path.dirname(os.path.abspath(__file__))
NPZ = os.path.join(HERE, "..", "..", "partB_santander", "data", "sant_events.npz")
OUT_DIR = os.path.join(HERE, "model")
OUT = os.path.join(OUT_DIR, "smartcross.pkl")

TEST_MONTHS = 3
MIN_EVENTS = 50
SEED = 42
ALPHA = 1.0          # served weight on the fused model; 1.0 = the fused model as trained.
                     # The peers are already inside it, so blending CF again would count them twice.
                     # Must match app.py's default.
HGB_PARAMS = dict(max_iter=300, max_depth=10, learning_rate=0.1,
                  l2_regularization=1.0, early_stopping=True,
                  validation_fraction=0.1, random_state=SEED)

DEMO_NAMES = ["n_products_held", "months_observed", "months_since_last_acq",
              "acquisitions_so_far", "age", "income", "seniority_months",
              "is_active", "is_new", "segment", "is_male"]


def mask_held(proba, held):
    """The API never recommends what the customer already holds. Score that way."""
    p = np.where(held > 0, 0.0, proba)
    s = p.sum(axis=1, keepdims=True)
    return np.where(s > 0, p / np.clip(s, 1e-12, None), proba)


def topk(proba, y, k):
    k = min(k, proba.shape[1])
    tk = np.argpartition(proba, -k, axis=1)[:, -k:]
    return float(np.mean([y[i] in tk[i] for i in range(len(y))]))


def map_at_k(proba, y, k=7):
    k = min(k, proba.shape[1])
    order = np.argsort(-proba, axis=1)[:, :k]
    s = 0.0
    for i, t in enumerate(y):
        h = np.where(order[i] == t)[0]
        if len(h):
            s += 1.0 / (h[0] + 1)
    return float(s / len(y))


def score(name, proba, y, note=""):
    """Score any engine that can produce a [n_samples x n_products] matrix.

    Every engine goes through this one function on the same held-out rows, so
    the numbers are comparable by construction rather than by assertion.
    """
    pred = proba.argmax(axis=1)
    r = {
        "name": name,
        "macro_f1": float(f1_score(y, pred, average="macro", zero_division=0)),
        "top1": float(accuracy_score(y, pred)),
        "top3": topk(proba, y, 3),
        "top5": topk(proba, y, 5),
        "map7": map_at_k(proba, y, 7),
        "note": note,
    }
    print(f"  {name:34} macro F1 {r['macro_f1']:.4f}  top-1 {r['top1']:.3f}  "
          f"top-3 {r['top3']:.3f}  MAP@7 {r['map7']:.3f}")
    return r


def main():
    t0 = time.time()
    print("=" * 78)
    print("SMARTCROSS API — BUILDING THE SERVED MODEL")
    print("=" * 78)

    npz = os.path.normpath(NPZ)
    if not os.path.exists(npz):
        print(f"\n[X] {npz} not found.")
        print("    Run partB_santander/scripts/phase2_4a_build.py first.\n")
        sys.exit(1)

    print(f"\nEvents file : {npz}")
    d = np.load(npz, allow_pickle=True)
    X_prev = d["X_prev"].astype(np.float32)
    X = np.hstack([X_prev, d["X_demo"]])
    if "X_hist" not in d.files:
        print("\n[X] sant_events.npz has no holdings history (X_hist).")
        print("    Rebuild it: partB_santander/scripts/phase2_4a_build.py\n")
        sys.exit(1)
    X_hist = d["X_hist"].astype(np.float32)
    hist_names = [str(s) for s in d["hist_names"]]
    y = d["y"].astype(int)
    mi = d["mi"].astype(int)
    all_names = [str(s) for s in d["names"]]
    print(f"Events      : {len(y):,}")

    # keep only products with enough events to learn
    counts = np.bincount(y, minlength=len(all_names))
    keep = np.where(counts >= MIN_EVENTS)[0]
    mask = np.isin(y, keep)
    remap = {o: n for n, o in enumerate(keep)}
    y = np.array([remap[v] for v in y[mask]])
    X, X_prev, X_hist, mi = X[mask], X_prev[mask], X_hist[mask], mi[mask]
    products = [all_names[j] for j in keep]
    K = len(products)
    held = X_prev[:, keep]                       # holdings in product-column space
    print(f"Products    : {K}  (dropped {len(all_names) - K} with <{MIN_EVENTS} events)")

    # time-based split: the model only ever sees the past
    cut = mi.max() - TEST_MONTHS + 1
    tr, te = mi < cut, mi >= cut
    print(f"\nTrain       : months 0-{cut-1}   {int(tr.sum()):,} events")
    print(f"Test        : months {cut}-{mi.max()}   {int(te.sum()):,} events")

    # ---- base rates, from training months only
    base = np.bincount(y[tr], minlength=K).astype(float)
    base /= base.sum()

    # ---- item-item lift, from training months only (explanations, Evidence view)
    onehot = np.eye(K, dtype=np.float32)[y[tr]]
    co = X_prev[tr].T @ onehot                     # [held_i x added_j]
    held_counts = np.clip(X_prev[tr].sum(axis=0)[:, None], 1, None)
    lift = (co / held_counts) / np.clip(base[None, :], 1e-9, None)
    lift_names = all_names

    # ---- the collaborative engine, from training months only
    T = peers.fit(X_prev[tr], y[tr], K)

    # ---- peer features for the fused model. Training rows: each month from
    #      strictly earlier months. Test rows: from the training months, exactly
    #      what the API will have when it serves.
    print("\n  building peer features (expanding window) ...", end=" ", flush=True)
    F = peers.expanding_features(X_prev, y, mi, K, np.unique(mi[tr]))
    F[te] = peers.features(T, X_prev[te])
    XF = np.hstack([X, F])                       # 79: the previous served model's inputs
    XH = np.hstack([X, F, X_hist])               # 127: + holdings history, served now
    print("done")

    # ======================================================================
    #  THE BAKE-OFF
    #  Every engine, one split, one scoring function, held products masked
    #  for all of them. Measured here so the console never quotes a number
    #  from somewhere else.
    # ======================================================================
    y_te = y[te]
    n_te = int(te.sum())
    h_te = held[te]
    engines = []
    print("\n" + "-" * 78)
    print("MEASURING EVERY ENGINE ON THE SAME HELD-OUT MONTHS")
    print("-" * 78)

    # --- popularity: rank by base rate. The honesty check.
    pop = mask_held(np.tile(base, (n_te, 1)), h_te)
    engines.append(score("Popularity (no ML)", pop, y_te,
                         "ranks every customer the same way, by base rate"))

    # --- the previous collaborative engine, kept as a measured reference
    old_cf = X_prev[te] @ lift
    rs = old_cf.sum(axis=1, keepdims=True)
    old_cf = mask_held(np.where(rs > 0, old_cf / np.clip(rs, 1e-9, None), base[None, :]), h_te)
    engines.append(score("Item-item CF, lift (previous)", old_cf, y_te,
                         "ranks by surprise, not likelihood -- replaced"))

    # --- the collaborative engine now served
    cf = mask_held(peers.pattern(T, X_prev[te]), h_te)
    engines.append(score("Collaborative (portfolio pattern)", cf, y_te,
                         "what customers with this exact portfolio added next"))

    # --- content alone: the profile model with no peer features
    print("\n  training content-only HistGB (35 features) ...", end=" ", flush=True)
    t1 = time.time()
    content_only = HistGradientBoostingClassifier(**HGB_PARAMS).fit(X[tr], y[tr])
    print(f"done ({time.time()-t1:.0f}s)")
    hgb = mask_held(content_only.predict_proba(X[te]), h_te)
    engines.append(score("Content only (HistGB)", hgb, y_te, "the profile model, no peers"))

    # --- XGBoost content model, only if it is actually installed
    try:
        from xgboost import XGBClassifier
        print("\n  training XGBoost (35 features) ...", end=" ", flush=True)
        t1 = time.time()
        xgb = XGBClassifier(
            n_estimators=300, max_depth=8, learning_rate=0.1,
            subsample=0.9, colsample_bytree=0.9, tree_method="hist",
            objective="multi:softprob", num_class=K,
            random_state=SEED, n_jobs=-1, verbosity=0)
        xgb.fit(X[tr], y[tr])
        print(f"done ({time.time()-t1:.0f}s)")
        engines.append(score("Content only (XGBoost)", mask_held(xgb.predict_proba(X[te]), h_te),
                             y_te, "second content engine, measured on this same split"))
    except ImportError:
        print("\n  XGBoost not installed - row omitted rather than quoted.")
        print("  (pip install xgboost, then rerun, to measure it here.)")

    # --- the previous hybrid, kept so the improvement is measured, not asserted
    engines.append(score("Weighted blend 0.5/0.5 (previous)", mask_held(0.5 * hgb + 0.5 * old_cf, h_te),
                         y_te, "the first hybrid: fixed weight, lift-CF"))

    # --- the fused hybrid without history, kept so the history gain is measured
    print("\n  training the fused hybrid, no history (79 inputs) ...", end=" ", flush=True)
    t1 = time.time()
    no_hist = HistGradientBoostingClassifier(**HGB_PARAMS).fit(XF[tr], y[tr])
    print(f"done ({time.time()-t1:.0f}s)")
    engines.append(score("Fused hybrid, no history (previous)", mask_held(no_hist.predict_proba(XF[te]), h_te),
                         y_te, "content model with collaborative scores, last month's holdings only"))

    # --- the fused hybrid the API serves: + holdings history. History and the
    #     settings below were chosen on a validation month (defense/improve_experiment.py)
    print("\n  training the fused hybrid + history (127 inputs) ...", end=" ", flush=True)
    t1 = time.time()
    clf = HistGradientBoostingClassifier(**HGB_PARAMS).fit(XH[tr], y[tr])
    print(f"done ({time.time()-t1:.0f}s, {clf.n_iter_} rounds)")
    fused = mask_held(clf.predict_proba(XH[te]), h_te)
    SERVED = "Fused hybrid + history (served)"
    engines.append(score(SERVED, fused, y_te,
                         "collaborative scores and 3 months of holdings history as inputs"))

    served = next(e for e in engines if e["name"] == SERVED)
    metrics = {
        "macro_f1": served["macro_f1"], "top1": served["top1"],
        "top3": served["top3"], "top5": served["top5"], "map7": served["map7"],
        "n_train": int(tr.sum()), "n_test": n_te, "n_products": K,
        "chance_macro_f1": 1.0 / K,
        "alpha": ALPHA,
        "served_engine": SERVED,
        "fusion": "feature augmentation",
        "xgboost_available": any("XGBoost" in e["name"] for e in engines),
        "engines": engines,
    }

    print("-" * 78)
    best3 = max(engines, key=lambda e: e["top3"])
    print(f"Best top-3 on this split: {best3['name']} ({best3['top3']:.3f})")
    if best3["name"] != SERVED:
        print("Note: the served engine is not the top-3 winner. That is stated")
        print("      in the console rather than hidden.")

    os.makedirs(OUT_DIR, exist_ok=True)
    joblib.dump({
        "clf": clf,
        "peers": T,
        "products": products,
        "all_names": all_names,
        "feature_names": ([f"holds_{n}" for n in all_names] + DEMO_NAMES +
                          [f"peer_cooc_{p}" for p in products] +
                          [f"peer_pattern_{p}" for p in products] + hist_names),
        "hist_names": hist_names,
        # the libraries that wrote this file: a pickle only opens under these
        "versions": {"python": sys.version.split()[0], "numpy": np.__version__,
                     "scikit_learn": __import__("sklearn").__version__},
        "base_rate": base,
        "lift": lift,
        "lift_names": lift_names,
        "metrics": metrics,
        "built_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }, OUT)
    print(f"\nSaved {OUT}  ({os.path.getsize(OUT)/1e6:.1f} MB)")
    print(f"  {len(engines)} engines measured, all on months {cut}-{mi.max()}")
    print("=" * 78)
    print(f"DONE ({time.time()-t0:.0f}s)   ->   now run  python app.py")
    print("=" * 78)


if __name__ == "__main__":
    main()
