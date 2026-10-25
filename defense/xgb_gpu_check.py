"""
SmartCross — can XGBoost on the GPU beat the chosen HistGB? Validation month only.

Same features as the improved model (35 profile + 44 peer + 48 history), same
fit / validation split, same criterion (validation top-3, macro F1 breaks ties
within 0.1 pts). Early stopping uses a slice of the FIT months, so the
validation month stays untouched until it is scored.

    python xgb_gpu_check.py     # needs an NVIDIA GPU; falls back to CPU otherwise
"""
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import improve_experiment as ie                                    # noqa: E402

from xgboost import XGBClassifier                                   # noqa: E402

CONFIGS = [
    ("depth 6, lr 0.1", dict(max_depth=6, learning_rate=0.1)),
    ("depth 8, lr 0.1", dict(max_depth=8, learning_rate=0.1)),
    ("depth 8, lr 0.05", dict(max_depth=8, learning_rate=0.05)),
    ("depth 10, lr 0.05", dict(max_depth=10, learning_rate=0.05)),
]


def main():
    d = np.load(ie.NPZ, allow_pickle=True)
    Xp, Xd, Xh = (d[k].astype(np.float32) for k in ("X_prev", "X_demo", "X_hist"))
    y, mi = d["y"].astype(int), d["mi"].astype(int)
    keep = np.where(np.bincount(y, minlength=24) >= ie.MIN_EVENTS)[0]
    m = np.isin(y, keep)
    remap = np.full(24, -1)
    remap[keep] = np.arange(len(keep))
    y, Xp, Xd, Xh, mi = remap[y[m]], Xp[m], Xd[m], Xh[m], mi[m]
    K = len(keep)
    cut = mi.max() - ie.TEST_MONTHS + 1
    val_m = cut - 1
    fit, val = mi < val_m, mi == val_m
    F = ie.peers.expanding_features(Xp, y, mi, K, np.unique(mi[mi < cut]))
    X = np.hstack([Xp, Xd, F, Xh])
    held = Xp[:, keep]

    rng = np.random.default_rng(ie.SEED)
    fit_idx = np.flatnonzero(fit)
    es = rng.random(len(fit_idx)) < 0.1                     # early-stopping slice of the FIT months
    a, b = fit_idx[~es], fit_idx[es]
    ref = json.load(open(ie.OUT))["grid"][0]                # the chosen HistGB on validation
    print(f"HistGB (chosen)            top-3 {ref['top3']:.4f}  macro F1 {ref['macro_f1']:.4f}")
    res, best = [], None
    for name, p in CONFIGS:
        t = time.time()
        try:
            clf = XGBClassifier(n_estimators=1500, tree_method="hist", device="cuda",
                                objective="multi:softprob", num_class=K, subsample=0.9,
                                colsample_bytree=0.9, early_stopping_rounds=40,
                                random_state=ie.SEED, verbosity=0, **p)
            clf.fit(X[a], y[a], eval_set=[(X[b], y[b])], verbose=False)
        except Exception as exc:                            # no CUDA: say so, do not fake it
            print(f"  {name}: GPU unavailable ({exc}); skipped")
            continue
        pv = ie.mask_held(clf.predict_proba(X[val]), held[val])
        r = ie.evaluate(pv, y[val])
        r.update(name=name, rounds=int(clf.best_iteration) + 1, seconds=round(time.time() - t))
        res.append(r)
        print(f"XGBoost GPU {name:16} top-3 {r['top3']:.4f}  macro F1 {r['macro_f1']:.4f}  "
              f"({r['rounds']} rounds, {r['seconds']}s)", flush=True)
        if best is None or ie.better(r, best):
            best = r
    wins = best is not None and ie.better(best, ref)
    print(f"-> XGBoost on the GPU {'BEATS' if wins else 'does not beat'} the chosen HistGB on validation")
    json.dump({"histgb_reference": ref, "xgboost_gpu": res, "xgboost_wins": wins},
              open(os.path.join(ie.HERE, "outputs", "xgb_gpu_check.json"), "w"), indent=2)


if __name__ == "__main__":
    main()
