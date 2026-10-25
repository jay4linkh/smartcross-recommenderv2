"""
================================================================================
 PHASE 4 : PEER-BASED COLLABORATIVE FILTERING + HYBRID ENGINE
 Smart Financial Cross-Selling Recommender System
================================================================================

 HOW TO RUN IN VSCODE
     1. Run phase3.py FIRST. This script reuses the exact split it saved,
        otherwise the comparison would not be like-for-like.
     2. Keep this file in the same folder as phase3.py
     3. Right-click -> "Run Python File in Terminal"   (or: python phase4.py)

 WHAT THIS IS
     This is "Example A: The Peer Group Match" from the project spec.

     Phases 1-3 all asked the same question: "what does this customer's
     profile PREDICT?" That is a content-based model, and it hit a ceiling
     because 16 credit features do not really encode a life decision.

     Phase 4 asks a completely different question:
         "What did the 200 most SIMILAR customers actually buy?"

     Rahul and Aman are both 25, both engineers, both earning the same.
     Rahul took a travel card. Aman has not. So we recommend it to Aman -
     not because his DTI implies it, but because people like him chose it.

     Three engines get compared:
         CONTENT  : the Phase 3 model            (feature -> product rules)
         PEER CF  : nearest-neighbour voting     (people like you bought X)
         HYBRID   : a weighted blend of both
     ...and all three are measured against the popularity baseline again.

 OUTPUTS (next to this script)
     phase4_results.json          all metrics
     phase4_comparison.png        content vs peer vs hybrid chart
     phase4_recommender.pkl       the deployable engine for Phase 5/6
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
    import pandas as pd
    import joblib
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.neighbors import NearestNeighbors
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import (f1_score, accuracy_score,
                                 precision_recall_fscore_support)
except ImportError as exc:
    print(f"\n[X] Missing package: {exc.name}")
    print("    pip install pandas numpy scikit-learn matplotlib joblib\n")
    sys.exit(1)


# ============================================================== CONFIG ====

N_NEIGHBORS = 200      # how many "peers" vote on each customer.
                       # smaller = more personalised but noisier
                       # larger  = smoother but drifts toward popularity

PEER_POOL = 150_000    # customers indexed for neighbour lookup.
                       #    8 GB RAM ->  80_000
                       #   16 GB RAM -> 150_000   <- default
                       #   32 GB RAM -> 300_000

RARITY_POWER = 0.5     # 0.0 = raw peer votes (majority products dominate)
                       # 0.5 = balanced                       <- default
                       # 1.0 = strongly boost rare products

CHUNK = 20_000         # test rows scored at a time. Lower it if you hit MemoryError.
SEED = 42

# =========================================================================


HERE = os.path.dirname(os.path.abspath(__file__))
SPLIT_DIR = os.path.join(HERE, "phase3_split")


def banner(text, ch="="):
    print("\n" + ch * 78)
    print(text)
    print(ch * 78)


def load_phase3():
    """Load the model and the exact split Phase 3 saved."""
    model_path = os.path.join(HERE, "phase3_best_model.pkl")
    if not os.path.exists(model_path):
        print("\n[X] phase3_best_model.pkl not found.")
        print("    Run  python phase3.py  first.\n")
        sys.exit(1)
    if not os.path.isdir(SPLIT_DIR):
        print("\n[X] phase3_split/ not found.")
        print("    Run  python phase3.py  first.\n")
        sys.exit(1)

    bundle = joblib.load(model_path)

    def read(name):
        pq = os.path.join(SPLIT_DIR, f"{name}.parquet")
        cs = os.path.join(SPLIT_DIR, f"{name}.csv")
        if os.path.exists(pq):
            return pd.read_parquet(pq)
        if os.path.exists(cs):
            return pd.read_csv(cs)
        print(f"\n[X] Missing {name} in phase3_split/. Re-run phase3.py\n")
        sys.exit(1)

    X_tr = read("X_train")
    X_te = read("X_test")
    y_tr = np.load(os.path.join(SPLIT_DIR, "y_train.npy"))
    y_te = np.load(os.path.join(SPLIT_DIR, "y_test.npy"))
    return bundle, X_tr, X_te, y_tr, y_te


# ---------------------------------------------------------------- metrics
def topk_hit_rate(proba, y_true, k):
    k = min(k, proba.shape[1])
    topk = np.argpartition(proba, -k, axis=1)[:, -k:]
    return float(np.mean([y_true[i] in topk[i] for i in range(len(y_true))]))


def evaluate(proba, y_true, label, n_classes):
    pred = proba.argmax(axis=1)
    _, rec, _, _ = precision_recall_fscore_support(
        y_true, pred, zero_division=0, labels=range(n_classes))
    return {
        "model": label,
        "macro_f1": float(f1_score(y_true, pred, average="macro", zero_division=0)),
        "top1": float(accuracy_score(y_true, pred)),
        "top3": topk_hit_rate(proba, y_true, 3),
        "top5": topk_hit_rate(proba, y_true, 5),
        "dead_classes": int((rec == 0).sum()),
    }


def normalise(mat):
    return mat / np.clip(mat.sum(axis=1, keepdims=True), 1e-9, None)


# ================================================================== MAIN ==
def main():
    t_start = time.time()
    banner("PHASE 4 : PEER-BASED COLLABORATIVE FILTERING")

    bundle, X_tr, X_te, y_tr, y_te = load_phase3()
    le = bundle["label_encoder"]
    names = list(le.classes_)
    K = len(names)

    print(f"\nPhase 3 model : {bundle['model_name']}")
    print(f"Train {len(y_tr):,}   |   Test {len(y_te):,}   |   Products {K}")

    # ------------------------------------------------- build the peer index
    banner("BUILDING THE PEER INDEX", "-")

    rng = np.random.RandomState(SEED)
    pool_n = min(PEER_POOL, len(y_tr))
    pool = rng.choice(len(y_tr), pool_n, replace=False)
    X_pool = X_tr.iloc[pool].values
    y_pool = y_tr[pool]

    # Distance needs comparable scales, otherwise annual_inc (~50,000)
    # completely drowns out dti (~20). Standardising is not optional here.
    scaler = StandardScaler().fit(X_pool)
    X_pool_s = scaler.transform(X_pool)

    print(f"\n  Indexing {pool_n:,} customers ...", end=" ", flush=True)
    nn = NearestNeighbors(n_neighbors=N_NEIGHBORS, algorithm="auto",
                          n_jobs=-1).fit(X_pool_s)
    print("done")

    # Inverse-frequency weights. A peer holding renewable_energy is far more
    # informative than yet another debt_consolidation, so rare products get
    # a louder vote. Without this, peer CF collapses into the popularity model.
    freq = np.bincount(y_pool, minlength=K).astype(float)
    freq[freq == 0] = 1.0
    rarity_w = (freq.sum() / freq) ** RARITY_POWER
    rarity_w /= rarity_w.mean()

    print(f"  Peer group size : {N_NEIGHBORS}")
    print(f"  Rarity power    : {RARITY_POWER}")

    # -------------------------------------------------- score the test set
    banner("SCORING TEST CUSTOMERS AGAINST THEIR PEER GROUPS", "-")

    X_te_s = scaler.transform(X_te.values)
    cf = np.zeros((len(y_te), K))

    print()
    for start in range(0, len(y_te), CHUNK):
        end = min(start + CHUNK, len(y_te))
        dist, idx = nn.kneighbors(X_te_s[start:end])
        sim = 1.0 / (1.0 + dist)            # closer peer = louder vote
        for row in range(end - start):
            np.add.at(cf[start + row], y_pool[idx[row]], sim[row])
        pct = 100 * end / len(y_te)
        print(f"  {end:,}/{len(y_te):,}  ({pct:5.1f}%)", end="\r", flush=True)
    print("\n  Peer scoring complete.")

    cf_raw = normalise(cf)
    cf_weighted = normalise(cf * rarity_w)

    # ------------------------------------------------- compare the engines
    prior = np.bincount(y_tr, minlength=K) / len(y_tr)
    pop_proba = np.tile(prior, (len(y_te), 1))

    if bundle.get("scaler") is not None:
        content = bundle["model"].predict_proba(bundle["scaler"].transform(X_te))
    else:
        content = bundle["model"].predict_proba(X_te)

    rows = [
        evaluate(pop_proba, y_te, "POPULARITY (no ML)", K),
        evaluate(content, y_te, f"CONTENT ({bundle['model_name']})", K),
        evaluate(cf_raw, y_te, "PEER CF (raw votes)", K),
        evaluate(cf_weighted, y_te, "PEER CF (rarity-weighted)", K),
    ]

    # ---------------------------------------------------- hybrid blending
    banner("HYBRID BLEND SWEEP    alpha * content  +  (1 - alpha) * peer")
    print("\n  alpha = 1.0 is pure content (Phase 3), alpha = 0.0 is pure peer.")
    print(f"\n  {'alpha':>6s} {'macroF1':>9s} {'top1':>7s} {'top3':>7s} {'top5':>7s}")
    print("  " + "-" * 42)

    best_alpha, best_f1, sweep = 1.0, -1.0, []
    for a in [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
        blend = a * content + (1 - a) * cf_weighted
        r = evaluate(blend, y_te, f"HYBRID a={a:.1f}", K)
        sweep.append({"alpha": a, **r})
        print(f"  {a:6.1f} {r['macro_f1']:9.4f} {r['top1']:7.3f} "
              f"{r['top3']:7.3f} {r['top5']:7.3f}")
        if r["macro_f1"] > best_f1:
            best_f1, best_alpha = r["macro_f1"], a

    hybrid = best_alpha * content + (1 - best_alpha) * cf_weighted
    rows.append(evaluate(hybrid, y_te, f"HYBRID best (a={best_alpha:.1f})", K))

    # ------------------------------------------------------------ results
    banner("PHASE 4 RESULTS")
    print(f"\n{'engine':32s} {'macroF1':>9s} {'top1':>7s} {'top3':>7s} "
          f"{'top5':>7s} {'dead':>6s}")
    print("-" * 78)
    for r in sorted(rows, key=lambda d: -d["macro_f1"]):
        print(f"{r['model']:32s} {r['macro_f1']:9.4f} {r['top1']:7.3f} "
              f"{r['top3']:7.3f} {r['top5']:7.3f} {r['dead_classes']:6d}")

    pop = rows[0]
    best = max(rows[1:], key=lambda d: d["macro_f1"])
    chance = 1.0 / K

    print(f"\n  Winner         : {best['model']}")
    print(f"  Macro F1       : {best['macro_f1']:.4f}   "
          f"(chance {chance:.4f}, lift {best['macro_f1']/chance:.2f}x)")
    print(f"  Top-3 hit rate : {best['top3']:.3f}   vs   "
          f"popularity {pop['top3']:.3f}")

    if best["top3"] > pop["top3"] + 0.01:
        print("\n  VERDICT: the peer signal adds real value over recommending")
        print("           popular products. Personalisation is working.")
    else:
        print("\n  VERDICT: still not clearly beating the popularity baseline")
        print("           on top-3. The honest read is that loan PURPOSE is")
        print("           weakly determined by credit features. Two options:")
        print("             1. use the 5-group collapse from Phase 3")
        print("             2. get real product-ownership data for true CF")
        print("           Either way, write this up - a negative result you")
        print("           diagnosed is stronger than a number you inflated.")

    # ------------------------------------------------------------- chart
    try:
        alphas = [s["alpha"] for s in sweep]
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))

        ax1.plot(alphas, [s["macro_f1"] for s in sweep], "o-", label="macro F1")
        ax1.plot(alphas, [s["top3"] for s in sweep], "s-", label="top-3")
        ax1.axhline(pop["top3"], ls="--", c="red", lw=1, label="popularity top-3")
        ax1.axvline(best_alpha, ls=":", c="green", lw=1,
                    label=f"best alpha = {best_alpha}")
        ax1.set_xlabel("alpha   (0 = pure peer,  1 = pure content)")
        ax1.set_ylabel("score")
        ax1.set_title("Hybrid blend sweep")
        ax1.legend(fontsize=8)
        ax1.grid(alpha=0.3)

        lbl = [r["model"].replace(" (", "\n(") for r in rows]
        xp = np.arange(len(rows))
        ax2.bar(xp - 0.2, [r["top3"] for r in rows], 0.4, label="top-3")
        ax2.bar(xp + 0.2, [r["macro_f1"] for r in rows], 0.4, label="macro F1")
        ax2.axhline(chance, ls="--", c="red", lw=1, label=f"chance ({chance:.3f})")
        ax2.set_xticks(xp)
        ax2.set_xticklabels(lbl, rotation=25, ha="right", fontsize=7)
        ax2.set_title("Engine comparison")
        ax2.legend(fontsize=8)

        fig.tight_layout()
        fig.savefig(os.path.join(HERE, "phase4_comparison.png"), dpi=120)
        plt.close(fig)
        print("\n  Saved phase4_comparison.png")
    except Exception as exc:
        print(f"  (chart skipped: {exc})")

    # ------------------------------------ what the final output looks like
    banner("SAMPLE OUTPUT : TOP-3 RECOMMENDATION CARDS")
    print("\n  This is the shape Phase 5 will serve from the API.\n")

    for i in rng.choice(len(y_te), 4, replace=False):
        row = X_te.iloc[i]
        print("  " + "-" * 62)
        print(f"  Customer  income ${row['annual_inc']:,.0f} | "
              f"FICO {row['fico_range_low']:.0f} | "
              f"DTI {row['dti']:.1f} | "
              f"loan ${row['loan_amnt']:,.0f}")
        for rank, j in enumerate(np.argsort(hybrid[i])[::-1][:3], 1):
            hit = "   <-- actual choice" if j == y_te[i] else ""
            print(f"     {rank}. {names[j]:22s} {hybrid[i][j]*100:5.1f}%{hit}")

    # -------------------------------------------------------------- save
    banner("SAVING ARTIFACTS", "-")

    joblib.dump({
        "content_model": bundle["model"],
        "content_scaler": bundle.get("scaler"),
        "peer_scaler": scaler,
        "peer_index": nn,
        "peer_labels": y_pool,
        "rarity_weights": rarity_w,
        "label_encoder": le,
        "features": list(X_te.columns),
        "best_alpha": best_alpha,
        "n_neighbors": N_NEIGHBORS,
        "popularity_prior": prior,
    }, os.path.join(HERE, "phase4_recommender.pkl"))
    print("  phase4_recommender.pkl   (the deployable engine for Phase 5)")

    with open(os.path.join(HERE, "phase4_results.json"), "w") as f:
        json.dump({
            "phase": 4,
            "n_classes": K,
            "class_names": names,
            "random_chance_macro_f1": chance,
            "n_neighbors": N_NEIGHBORS,
            "peer_pool": pool_n,
            "rarity_power": RARITY_POWER,
            "best_alpha": best_alpha,
            "results": rows,
            "alpha_sweep": sweep,
        }, f, indent=2)
    print("  phase4_results.json")

    banner(f"PHASE 4 COMPLETE  ({time.time() - t_start:.0f}s)")


if __name__ == "__main__":
    main()
