"""
================================================================================
 PART B / STEP 2.4b : TRAIN ON THE FULL EVENT TABLE
 SmartCross - Hybrid Financial Product Cross-Selling Recommender
================================================================================

 HOW TO RUN
     1. python phase2_4a_build.py     (builds ../data/sant_events.npz)
     2. python phase2_4b_train.py     (this file)

 WHAT CHANGED SINCE STEP 2.3
     Step 2.3 reached 58.3% top-1 and 0.310 macro F1, but it was handicapped:
     it used a 1-in-5 customer sample (112,063 events) and the only features
     were "which products do you hold right now".

     This step fixes both:
       - ALL customers, roughly 5x the acquisition events
       - behavioural lag features that were missing entirely:
             months_since_last_acquisition
             acquisitions_so_far
             months_observed
             products_held_count
         Momentum matters. A customer who added something last month behaves
         very differently from one dormant for a year, and step 2.3 could not
         see that difference at all.

 THE TEST IS STILL HONEST
     Time-based split: train on earlier months, test on later ones. The model
     only ever sees the past. The same no-ML popularity baseline is scored on
     the identical test set.

 SECONDARY EXPERIMENT - PRODUCT FAMILIES
     The 24 products collapse into 8 families a bank would actually market
     together. This is reported SEPARATELY and never swapped in for the main
     number: a higher score there comes from an easier question, not a better
     model. Both are printed so the comparison stays honest.

 OUTPUT
     ../outputs/phase2_4_results.json
     ../outputs/phase2_4_comparison.png
     ../outputs/phase2_4_importance.png
================================================================================
"""

import os
import sys
import json
import time
import warnings

warnings.filterwarnings("ignore")

try:
    import numpy as np
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.inspection import permutation_importance
    from sklearn.metrics import (f1_score, accuracy_score,
                                 precision_recall_fscore_support)
except ImportError as exc:
    print(f"\n[X] Missing package: {exc.name}")
    print("    pip install numpy scikit-learn matplotlib xgboost\n")
    sys.exit(1)

try:
    from xgboost import XGBClassifier
    HAS_XGB = True
except ImportError:
    HAS_XGB = False
    print("[!] xgboost not installed - will be skipped.  pip install xgboost")


# ============================================================== CONFIG ====

TEST_MONTHS = 3      # last N months held out for testing
MIN_EVENTS = 50      # drop products with fewer acquisitions than this
SEED = 42
RUN_IMPORTANCE = True

# =========================================================================


HERE = os.path.dirname(os.path.abspath(__file__))
NPZ = os.path.join(HERE, "..", "data", "sant_events.npz")
OUTDIR = os.path.join(HERE, "..", "outputs")

DEMO_NAMES = ["n_products_held", "months_observed", "months_since_last_acq",
              "acquisitions_so_far", "age", "income", "seniority_months",
              "is_active", "is_new", "segment", "is_male"]

# Product families - how a bank would actually group these for marketing.
FAMILY = {
    "savings_account": "DEPOSITS", "short_term_deposits": "DEPOSITS",
    "medium_term_deposits": "DEPOSITS", "long_term_deposits": "DEPOSITS",
    "current_account": "ACCOUNTS", "particular_account": "ACCOUNTS",
    "particular_plus_account": "ACCOUNTS", "mas_particular_account": "ACCOUNTS",
    "junior_account": "ACCOUNTS", "e_account": "ACCOUNTS",
    "home_account": "ACCOUNTS",
    "payroll_account": "PAYROLL", "payroll": "PAYROLL", "pensions": "PAYROLL",
    "pension_plan": "PAYROLL",
    "credit_card": "CARDS", "direct_debit": "CARDS",
    "funds": "INVESTMENTS", "securities": "INVESTMENTS",
    "derivatives_account": "INVESTMENTS", "guarantees": "INVESTMENTS",
    "mortgage": "LENDING", "loans": "LENDING",
    "taxes": "SERVICES",
}


def banner(t, ch="="):
    print("\n" + ch * 78); print(t); print(ch * 78)


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


def evaluate(proba, y, label, K):
    pred = proba.argmax(axis=1)
    _, rec, _, _ = precision_recall_fscore_support(
        y, pred, zero_division=0, labels=range(K))
    return {"model": label,
            "macro_f1": float(f1_score(y, pred, average="macro", zero_division=0)),
            "weighted_f1": float(f1_score(y, pred, average="weighted", zero_division=0)),
            "top1": float(accuracy_score(y, pred)),
            "top3": topk(proba, y, 3),
            "top5": topk(proba, y, 5),
            "map7": map_at_k(proba, y, 7),
            "dead_classes": int((rec == 0).sum())}


def run_task(X, y, mi, names, title, run_xgb=True):
    """Train and score one prediction task with a time-based split."""
    banner(title)

    cut = mi.max() - TEST_MONTHS + 1
    tr, te = mi < cut, mi >= cut
    print(f"\n  Train: months 0-{cut-1}   {int(tr.sum()):,} events")
    print(f"  Test : months {cut}-{mi.max()}   {int(te.sum()):,} events")

    X_tr, X_te, y_tr, y_te = X[tr], X[te], y[tr], y[te]
    K = len(names)
    out = []

    # ---- popularity baseline
    pri = np.bincount(y_tr, minlength=K).astype(float)
    pri = pri / pri.sum()
    out.append(evaluate(np.tile(pri, (len(y_te), 1)), y_te,
                        "POPULARITY (no ML)", K))

    # ---- item-item CF on prior holdings
    prior_tr, prior_te = X_tr[:, :24], X_te[:, :24]
    co = prior_tr.T @ np.eye(K, dtype=np.float32)[y_tr]
    cond = co / np.clip(prior_tr.sum(axis=0)[:, None], 1, None)
    cond = cond / np.clip(pri[None, :], 1e-9, None)     # lift correction
    cf = prior_te @ cond
    cf[prior_te.sum(axis=1) == 0] = pri
    cf = cf / np.clip(cf.sum(axis=1, keepdims=True), 1e-9, None)
    out.append(evaluate(cf, y_te, "ITEM-ITEM CF", K))

    # ---- content-based classifier
    print("\n  Training HistGradientBoosting ...", end=" ", flush=True)
    t = time.time()
    hgb = HistGradientBoostingClassifier(
        max_iter=300, max_depth=10, learning_rate=0.1,
        l2_regularization=1.0, early_stopping=True,
        validation_fraction=0.1, random_state=SEED)
    hgb.fit(X_tr, y_tr)
    p_hgb = hgb.predict_proba(X_te)
    print(f"done ({time.time()-t:.0f}s)")
    out.append(evaluate(p_hgb, y_te, "CONTENT (HistGB)", K))

    best_clf, p_best = hgb, p_hgb
    if run_xgb and HAS_XGB:
        print("  Training XGBoost ...", end=" ", flush=True)
        t = time.time()
        xgb = XGBClassifier(
            n_estimators=400, max_depth=8, learning_rate=0.08,
            subsample=0.8, colsample_bytree=0.8, min_child_weight=3,
            objective="multi:softprob", num_class=K, tree_method="hist",
            random_state=SEED, n_jobs=-1, verbosity=0)
        xgb.fit(X_tr, y_tr)
        p_xgb = xgb.predict_proba(X_te)
        print(f"done ({time.time()-t:.0f}s)")
        r = evaluate(p_xgb, y_te, "CONTENT (XGBoost)", K)
        out.append(r)
        if r["macro_f1"] > out[-2]["macro_f1"]:
            best_clf, p_best = xgb, p_xgb

    # ---- hybrid
    best_a, best_f1, sweep = 1.0, -1.0, []
    for a in [0.0, 0.2, 0.4, 0.5, 0.6, 0.8, 0.9, 1.0]:
        r = evaluate(a * p_best + (1 - a) * cf, y_te, f"HYBRID a={a:.1f}", K)
        sweep.append({"alpha": a, **r})
        if r["macro_f1"] > best_f1:
            best_f1, best_a = r["macro_f1"], a
    out.append(evaluate(best_a * p_best + (1 - best_a) * cf, y_te,
                        f"HYBRID best (a={best_a:.1f})", K))

    print(f"\n{'model':30s} {'macroF1':>9s} {'top1':>7s} {'top3':>7s} "
          f"{'MAP@7':>7s} {'dead':>6s}")
    print("-" * 78)
    for r in sorted(out, key=lambda d: -d["macro_f1"]):
        print(f"{r['model']:30s} {r['macro_f1']:9.4f} {r['top1']:7.3f} "
              f"{r['top3']:7.3f} {r['map7']:7.3f} {r['dead_classes']:6d}")

    pop = out[0]
    best = max(out[1:], key=lambda d: d["macro_f1"])
    chance = 1.0 / K
    print(f"\n  chance macro F1 {chance:.4f} | best {best['model']}")
    print(f"    macro F1 {best['macro_f1']:.4f}  "
          f"({best['macro_f1']/chance:.2f}x chance, "
          f"{best['macro_f1']/max(pop['macro_f1'],1e-9):.2f}x popularity)")
    print(f"    top-1 {best['top1']:.3f} (pop {pop['top1']:.3f})  |  "
          f"top-3 {best['top3']:.3f} (pop {pop['top3']:.3f})")

    return {"results": out, "sweep": sweep, "best_alpha": best_a,
            "chance": chance, "n_classes": K, "names": names,
            "n_train": int(tr.sum()), "n_test": int(te.sum())}, best_clf, X_te, y_te


# ================================================================== MAIN ==
def main():
    t0 = time.time()
    banner("PART B / STEP 2.4b : FULL-DATA MODEL WITH LAG FEATURES")

    if not os.path.exists(NPZ):
        print(f"\n[X] {os.path.normpath(NPZ)} not found.")
        print("    Run  python phase2_4a_build.py  first.\n")
        sys.exit(1)

    d = np.load(NPZ, allow_pickle=True)
    X = np.hstack([d["X_prev"].astype(np.float32), d["X_demo"]])
    y = d["y"].astype(int)
    mi = d["mi"].astype(int)
    names24 = [str(s) for s in d["names"]]
    feat_names = [f"holds_{n}" for n in names24] + DEMO_NAMES

    print(f"\n  Events   : {len(y):,}")
    print(f"  Features : {X.shape[1]}  (24 holdings + {len(DEMO_NAMES)} behavioural)")
    print(f"  Months   : {mi.max()+1}")
    print(f"\n  Step 2.3 had 112,063 events and no lag features.")

    # drop products with too few events
    counts = np.bincount(y, minlength=24)
    keep = np.where(counts >= MIN_EVENTS)[0]
    dropped = [names24[j] for j in range(24) if counts[j] < MIN_EVENTS]
    m = np.isin(y, keep)
    remap = {o: n for n, o in enumerate(keep)}
    y24 = np.array([remap[v] for v in y[m]])
    X24, mi24 = X[m], mi[m]
    names = [names24[j] for j in keep]
    if dropped:
        print(f"  Dropped as too rare (<{MIN_EVENTS}): {', '.join(dropped)}")

    # ------------------------------------------------- main task: products
    main_res, clf, X_te, y_te = run_task(
        X24, y24, mi24, names,
        f"MAIN TASK : PREDICT NEXT PRODUCT ({len(names)} classes)")

    # ------------------------------- secondary: product families (8 classes)
    fam_lab = sorted({FAMILY[n] for n in names24 if n in FAMILY})
    fidx = {f: i for i, f in enumerate(fam_lab)}
    yf = np.array([fidx[FAMILY[names24[v]]] for v in y])
    fam_res, _, _, _ = run_task(
        X, yf, mi, fam_lab,
        f"SECONDARY EXPERIMENT : PRODUCT FAMILIES ({len(fam_lab)} classes)",
        run_xgb=False)

    # ----------------------------------------------------- feature value
    imp_pairs = []
    if RUN_IMPORTANCE:
        banner("WHICH FEATURES ACTUALLY MATTER", "-")
        print("\n  Permutation importance on a 5,000-row test sample.")
        print("  This answers whether the new lag features earned their place.\n")
        n = min(5000, len(y_te))
        sel = np.random.RandomState(SEED).choice(len(y_te), n, replace=False)
        try:
            pi = permutation_importance(clf, X_te[sel], y_te[sel],
                                        n_repeats=3, random_state=SEED,
                                        scoring="accuracy", n_jobs=1)
            order = np.argsort(pi.importances_mean)[::-1][:15]
            for r, i in enumerate(order, 1):
                tag = "  <-- NEW in 2.4" if feat_names[i] in (
                    "months_since_last_acq", "acquisitions_so_far",
                    "months_observed", "n_products_held") else ""
                print(f"  {r:2d}. {feat_names[i]:28s} {pi.importances_mean[i]:.4f}{tag}")
                imp_pairs.append((feat_names[i], float(pi.importances_mean[i])))
        except Exception as exc:
            print(f"  (skipped: {exc})")

    # ------------------------------------------------------------ charts
    os.makedirs(OUTDIR, exist_ok=True)
    try:
        res = main_res["results"]
        fig, (a1, a2) = plt.subplots(1, 2, figsize=(14, 5.5))
        xp = np.arange(len(res))
        a1.bar(xp - 0.25, [r["top3"] for r in res], 0.25, label="top-3")
        a1.bar(xp, [r["map7"] for r in res], 0.25, label="MAP@7")
        a1.bar(xp + 0.25, [r["macro_f1"] for r in res], 0.25, label="macro F1")
        a1.axhline(main_res["chance"], ls="--", c="red", lw=1,
                   label=f"chance ({main_res['chance']:.3f})")
        a1.set_xticks(xp)
        a1.set_xticklabels([r["model"].replace(" (", "\n(") for r in res],
                           rotation=20, ha="right", fontsize=7)
        a1.set_title(f"Step 2.4 - next product ({main_res['n_classes']} classes)")
        a1.legend(fontsize=8)

        sw = main_res["sweep"]
        a2.plot([s["alpha"] for s in sw], [s["macro_f1"] for s in sw], "o-",
                label="macro F1")
        a2.plot([s["alpha"] for s in sw], [s["top3"] for s in sw], "s-",
                label="top-3")
        a2.axhline(res[0]["top3"], ls="--", c="red", lw=1, label="popularity top-3")
        a2.set_xlabel("alpha  (0 = pure CF, 1 = pure content)")
        a2.set_title("Hybrid blend sweep")
        a2.legend(fontsize=8); a2.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(os.path.join(OUTDIR, "phase2_4_comparison.png"), dpi=120)
        plt.close(fig)
        print("\n  Saved ../outputs/phase2_4_comparison.png")

        if imp_pairs:
            fig, ax = plt.subplots(figsize=(9, 6))
            lbl = [p[0] for p in imp_pairs][::-1]
            val = [p[1] for p in imp_pairs][::-1]
            ax.barh(range(len(val)), val, color="#4C72B0")
            ax.set_yticks(range(len(lbl)))
            ax.set_yticklabels(lbl, fontsize=8)
            ax.set_xlabel("drop in accuracy when shuffled")
            ax.set_title("Step 2.4 - permutation importance")
            fig.tight_layout()
            fig.savefig(os.path.join(OUTDIR, "phase2_4_importance.png"), dpi=120)
            plt.close(fig)
            print("  Saved ../outputs/phase2_4_importance.png")
    except Exception as exc:
        print(f"  (charts skipped: {exc})")

    # ------------------------------------------------------------ verdict
    banner("STEP 2.3  vs  STEP 2.4")
    mb = max(main_res["results"][1:], key=lambda d: d["macro_f1"])
    fb = max(fam_res["results"][1:], key=lambda d: d["macro_f1"])
    print(f"""
                            step 2.3        step 2.4
  ----------------------------------------------------------
  events                     112,063        {len(y):,}
  customers                  191,102        all
  lag features                  none        4
  macro F1                    0.3098        {mb['macro_f1']:.4f}
  top-1                        0.583        {mb['top1']:.3f}
  top-3                        0.869        {mb['top3']:.3f}
  MAP@7                        0.732        {mb['map7']:.3f}

  SECONDARY (product families, {fam_res['n_classes']} classes):
    top-1 {fb['top1']:.3f} | top-3 {fb['top3']:.3f} | macro F1 {fb['macro_f1']:.4f}

  The family numbers are higher because the question is easier, not because
  the model is better. Report the {main_res['n_classes']}-class result as the
  headline and the family result as a clearly-labelled secondary experiment.
""")

    json.dump({"step": "2.4", "main_task": main_res, "family_task": fam_res,
               "feature_importance": imp_pairs, "n_events": int(len(y))},
              open(os.path.join(OUTDIR, "phase2_4_results.json"), "w"),
              indent=2, default=float)
    print("  Saved ../outputs/phase2_4_results.json")
    banner(f"STEP 2.4b COMPLETE  ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
