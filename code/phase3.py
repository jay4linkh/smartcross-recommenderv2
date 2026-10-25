"""
================================================================================
 PHASE 3 : MULTI-MODEL COMPARISON + PROPER RECOMMENDER METRICS
 Smart Financial Cross-Selling Recommender System
================================================================================

 HOW TO RUN IN VSCODE
     1. Put this file in the SAME folder as lending_club_model_ready.csv
        (or set DATA_PATH in the CONFIG block below)
     2. In the terminal:
            pip install pandas numpy scikit-learn imbalanced-learn xgboost matplotlib joblib
     3. Right-click the file -> "Run Python File in Terminal"
        (or:  python phase3.py)

 WHY THIS PHASE EXISTS
     Phase 2 had two blind spots:
       (a) we only ever tried Random Forest
       (b) we only measured TOP-1 (argmax). But the product shows the customer
           THREE recommendations - so top-1 accuracy is the wrong exam.

     Phase 3 fixes both, and adds the honesty check Phase 2 was missing:
     a POPULARITY BASELINE. That baseline just shows everyone the most common
     products, with zero machine learning. If our model can't beat it, the ML
     is decorative and we need to know that now, not in Phase 6.

 OUTPUTS (all written next to this script)
     phase3_best_model.pkl          best model + encoder + feature names
     phase3_results.json            every metric, ready for your report
     phase3_model_comparison.png    bar chart for your slides
     phase3_split/                  exact train/test split for Phase 4 to reuse
================================================================================
"""

import os
import sys
import json
import time
import warnings

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------- imports
try:
    import numpy as np
    import pandas as pd
    import joblib
    import matplotlib
    matplotlib.use("Agg")           # no GUI window, just saves PNG files
    import matplotlib.pyplot as plt
    from sklearn.model_selection import train_test_split
    from sklearn.ensemble import (RandomForestClassifier,
                                  HistGradientBoostingClassifier)
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import LabelEncoder, StandardScaler
    from sklearn.metrics import (f1_score, accuracy_score,
                                 precision_recall_fscore_support)
    from imblearn.under_sampling import RandomUnderSampler
    from imblearn.over_sampling import SMOTE
except ImportError as exc:
    print(f"\n[X] Missing package: {exc.name}")
    print("    Install everything with:\n")
    print("    pip install pandas numpy scikit-learn imbalanced-learn "
          "xgboost matplotlib joblib\n")
    sys.exit(1)

try:
    from xgboost import XGBClassifier
    HAS_XGB = True
except ImportError:
    HAS_XGB = False
    print("[!] xgboost not found - it will be skipped.  pip install xgboost")


# ============================================================== CONFIG ====
# The only block you should need to touch.

DATA_PATH = None
# None = auto-find the CSV near this script. To hardcode instead:
# DATA_PATH = r"C:\Users\jay4l\Desktop\RecommenderSysProject\data\raw\user_transaction_data\lending_club_model_ready.csv"

SUBSAMPLE = 500_000
# How many rows to use. Tune to your RAM:
#     8 GB  ->  300_000
#    16 GB  ->  500_000    <- default, good balance of speed and signal
#    32 GB  ->  None       (uses all 1.95M rows)

TARGET_PER_CLASS = 20_000   # balanced training rows per product class
TEST_SIZE = 0.2
SEED = 42

# =========================================================================


HERE = os.path.dirname(os.path.abspath(__file__))
SPLIT_DIR = os.path.join(HERE, "phase3_split")


def banner(text, ch="="):
    print("\n" + ch * 78)
    print(text)
    print(ch * 78)


def find_data():
    """Locate the CSV without hardcoding a Colab or Windows path.

    Phase 1 broke because of a hardcoded /content/ path, so this searches
    a few sensible relative locations and then walks the project tree.
    """
    if DATA_PATH and os.path.exists(DATA_PATH):
        return DATA_PATH

    candidates = [
        "lending_club_model_ready.csv",
        "lending_club_final.csv",
        os.path.join("data", "lending_club_model_ready.csv"),
        os.path.join("data", "raw", "user_transaction_data",
                     "lending_club_model_ready.csv"),
        os.path.join("..", "lending_club_model_ready.csv"),
    ]
    for c in candidates:
        p = c if os.path.isabs(c) else os.path.join(HERE, c)
        if os.path.exists(p):
            return p

    for root, _, files in os.walk(HERE):
        for f in files:
            if f.startswith("lending_club") and f.endswith(".csv"):
                return os.path.join(root, f)
    return None


def parquet_available():
    try:
        import pyarrow  # noqa: F401
        return True
    except ImportError:
        return False


def stratified_subsample(df, n, label_col="target", min_per_class=50, seed=SEED):
    """Shrink the dataset without losing the rare classes.

    A plain df.sample() is dangerous here. renewable_energy is only 0.06% of
    the data, so a small random sample can leave it with 1 row - and then the
    stratified train/test split crashes outright. This keeps every class'
    natural proportion but guarantees a floor of min_per_class rows.
    """
    if n is None or n >= len(df):
        return df

    counts = df[label_col].value_counts()
    frac = n / len(df)
    parts = []
    for cls, total in counts.items():
        take = max(min_per_class, int(round(total * frac)))
        take = min(take, total)                    # can't take more than exists
        parts.append(
            df[df[label_col] == cls].sample(take, random_state=seed)
        )
    out = pd.concat(parts).sample(frac=1.0, random_state=seed).reset_index(drop=True)

    tiny = [c for c, v in out[label_col].value_counts().items() if v < 10]
    if tiny:
        print(f"  [!] Very small classes even after sampling: {tiny}")
        print("      Raise SUBSAMPLE if results for these look unstable.")
    return out


# ---------------------------------------------------------------- metrics
def topk_hit_rate(proba, y_true, k):
    """Is the customer's ACTUAL product inside the model's top-k picks?

    This is the metric that matches the product. The dashboard shows three
    recommendations, so "did we surface the right one at all" matters much
    more than "was our single best guess exactly right".
    """
    k = min(k, proba.shape[1])
    topk = np.argpartition(proba, -k, axis=1)[:, -k:]
    return float(np.mean([y_true[i] in topk[i] for i in range(len(y_true))]))


def evaluate(proba, y_true, label, n_classes, fit_sec=None):
    pred = proba.argmax(axis=1)
    _, rec, _, _ = precision_recall_fscore_support(
        y_true, pred, zero_division=0, labels=range(n_classes))
    return {
        "model": label,
        "macro_f1": float(f1_score(y_true, pred, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, pred, average="weighted", zero_division=0)),
        "top1": float(accuracy_score(y_true, pred)),
        "top3": topk_hit_rate(proba, y_true, 3),
        "top5": topk_hit_rate(proba, y_true, 5),
        "dead_classes": int((rec == 0).sum()),
        "fit_seconds": fit_sec,
    }


def balance(X, y, target, seed=SEED):
    """Hybrid resampling: undersample the giants, SMOTE the tiny ones.

    Pure SMOTE is not an option here. The raw imbalance is about 917:1, so
    oversampling every class up to the majority would create ~11.5M rows.
    """
    counts = pd.Series(y).value_counts()

    shrink = {c: target for c, n in counts.items() if n > target}
    if shrink:
        X, y = RandomUnderSampler(sampling_strategy=shrink,
                                  random_state=seed).fit_resample(X, y)

    counts = pd.Series(y).value_counts()
    grow = {c: target for c, n in counts.items() if n < target}
    if grow:
        k = max(1, min(5, int(counts.min()) - 1))
        X, y = SMOTE(sampling_strategy=grow, k_neighbors=k,
                     random_state=seed).fit_resample(X, y)
    return X, y


# ================================================================== MAIN ==
def main():
    t_start = time.time()
    banner("PHASE 3 : MULTI-MODEL COMPARISON + TOP-K RANKING METRICS")

    # ------------------------------------------------------------ load
    path = find_data()
    if path is None:
        print("\n[X] Could not find the dataset.")
        print("    Put lending_club_model_ready.csv next to this script, or")
        print("    set DATA_PATH at the top of the file.\n")
        sys.exit(1)

    print(f"\nDataset : {path}")
    df = pd.read_csv(path)
    print(f"Loaded  : {len(df):,} rows x {df.shape[1]} columns")

    if SUBSAMPLE and len(df) > SUBSAMPLE:
        df = stratified_subsample(df, SUBSAMPLE)
        print(f"Sampled : {len(df):,} rows   (SUBSAMPLE = {SUBSAMPLE:,}, "
              f"class-aware so rare products survive)")

    # A class needs at least 2 rows to be splittable into train and test.
    too_rare = [c for c, n in df["target"].value_counts().items() if n < 2]
    if too_rare:
        print(f"  [!] Dropping unsplittable classes (<2 rows): {too_rare}")
        df = df[~df["target"].isin(too_rare)].reset_index(drop=True)

    X = df.drop("target", axis=1)
    le = LabelEncoder()
    y = le.fit_transform(df["target"])
    class_names = list(le.classes_)
    K = len(class_names)
    print(f"Features: {X.shape[1]}   |   Product classes: {K}")

    # ----------------------------------------------------------- split
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=SEED, stratify=y)
    print(f"\nTrain {len(y_tr):,}   |   Test {len(y_te):,}   (stratified)")
    print("Test set keeps its real imbalanced mix, so the metrics stay honest.")

    # --------------------------------------------------------- balance
    banner(f"HYBRID RESAMPLING  ->  {TARGET_PER_CLASS:,} rows per class", "-")
    X_bal, y_bal = balance(X_tr, y_tr, TARGET_PER_CLASS)
    print(f"\n  {len(y_tr):,} raw rows  ->  {len(y_bal):,} balanced training rows")

    results = []

    # --------------------------------------------- baseline: popularity
    banner("BASELINE : POPULARITY  (zero machine learning)", "-")
    print("\n  'Recommend the most common products to literally everyone.'")
    print("  Every model below has to beat this to justify existing.")

    prior = np.bincount(y_tr, minlength=K) / len(y_tr)
    pop_proba = np.tile(prior, (len(y_te), 1))
    r = evaluate(pop_proba, y_te, "POPULARITY (no ML)", K)
    results.append(r)
    print(f"\n  top-1 {r['top1']:.3f}   top-3 {r['top3']:.3f}   "
          f"top-5 {r['top5']:.3f}   macro F1 {r['macro_f1']:.4f}")

    # ----------------------------------------------------------- models
    models = {
        "RandomForest (Phase 2)": RandomForestClassifier(
            n_estimators=150, max_depth=20, min_samples_split=10,
            min_samples_leaf=4, class_weight="balanced",
            random_state=SEED, n_jobs=-1),
        "HistGradientBoosting": HistGradientBoostingClassifier(
            max_iter=200, max_depth=10, learning_rate=0.1, random_state=SEED),
        "LogisticRegression": LogisticRegression(
            max_iter=300, n_jobs=-1, random_state=SEED),
    }
    if HAS_XGB:
        models["XGBoost"] = XGBClassifier(
            n_estimators=300, max_depth=8, learning_rate=0.1,
            subsample=0.8, colsample_bytree=0.8,
            objective="multi:softprob", num_class=K,
            tree_method="hist", random_state=SEED, n_jobs=-1, verbosity=0)

    banner("TRAINING MODELS", "-")
    fitted = {}
    for label, mdl in models.items():
        print(f"\n  -> {label} ...", end=" ", flush=True)
        t0 = time.time()

        if label == "LogisticRegression":
            scaler = StandardScaler().fit(X_bal)
            mdl.fit(scaler.transform(X_bal), y_bal)
            proba = mdl.predict_proba(scaler.transform(X_te))
            fitted[label] = (mdl, scaler)
        else:
            mdl.fit(X_bal, y_bal)
            proba = mdl.predict_proba(X_te)
            fitted[label] = (mdl, None)

        secs = round(time.time() - t0, 1)
        r = evaluate(proba, y_te, label, K, secs)
        results.append(r)
        print(f"done in {secs}s")
        print(f"     macro F1 {r['macro_f1']:.4f} | top-1 {r['top1']:.3f} "
              f"| top-3 {r['top3']:.3f} | dead classes {r['dead_classes']}")

    # ---------------------------------------------------------- report
    banner("PHASE 3 RESULTS")
    print(f"\n{'model':28s} {'macroF1':>9s} {'top1':>7s} {'top3':>7s} "
          f"{'top5':>7s} {'dead':>6s}")
    print("-" * 78)
    for r in sorted(results, key=lambda d: -d["macro_f1"]):
        print(f"{r['model']:28s} {r['macro_f1']:9.4f} {r['top1']:7.3f} "
              f"{r['top3']:7.3f} {r['top5']:7.3f} {r['dead_classes']:6d}")

    ml_only = [r for r in results if "POPULARITY" not in r["model"]]
    best = max(ml_only, key=lambda d: d["macro_f1"])
    pop = results[0]

    print(f"\n  Best model     : {best['model']}")
    print(f"  Macro F1       : {best['macro_f1']:.4f}    "
          f"(random chance {1/K:.4f}, lift {best['macro_f1']/(1/K):.2f}x)")
    print(f"  Top-3 hit rate : {best['top3']:.3f}   vs   "
          f"popularity {pop['top3']:.3f}")

    if best["top3"] > pop["top3"] + 0.01:
        print("\n  VERDICT: the model BEATS the no-ML baseline on top-3.")
        print("           Personalisation is doing real work.")
    else:
        print("\n  VERDICT: the model does NOT clearly beat 'just recommend")
        print("           the popular products'. That points to the 16 features")
        print("           not encoding loan PURPOSE. Report it as a finding,")
        print("           not a bug - and lean on Phase 4's peer signal.")

    # ----------------------------------------------------------- chart
    try:
        labels = [r["model"] for r in results]
        xpos = np.arange(len(labels))
        w = 0.27
        fig, ax = plt.subplots(figsize=(11, 6))
        ax.bar(xpos - w, [r["top1"] for r in results], w, label="top-1")
        ax.bar(xpos, [r["top3"] for r in results], w, label="top-3")
        ax.bar(xpos + w, [r["macro_f1"] for r in results], w, label="macro F1")
        ax.axhline(1 / K, ls="--", c="red", lw=1, label=f"chance ({1/K:.3f})")
        ax.set_xticks(xpos)
        ax.set_xticklabels(labels, rotation=20, ha="right", fontsize=9)
        ax.set_ylabel("score")
        ax.set_title("Phase 3 - model comparison (top-3 is what the product needs)")
        ax.legend()
        fig.tight_layout()
        fig.savefig(os.path.join(HERE, "phase3_model_comparison.png"), dpi=120)
        plt.close(fig)
        print("\n  Saved phase3_model_comparison.png")
    except Exception as exc:
        print(f"  (chart skipped: {exc})")

    # --------------------------------------- experiment: 13 -> 5 classes
    banner("EXPERIMENT : COLLAPSE 13 PRODUCTS -> 5 GROUPS")
    print("\n  'wedding' vs 'vacation' vs 'moving' are not really separable from")
    print("  a credit profile. Grouping them into coherent product families")
    print("  should score better AND map more cleanly onto real bank products.")

    GROUPS = {
        "debt_consolidation": "DEBT_RELIEF",
        "credit_card":        "DEBT_RELIEF",
        "home_improvement":   "HOME",
        "house":              "HOME",
        "renewable_energy":   "HOME",
        "car":                "DISCRETIONARY",
        "major_purchase":     "DISCRETIONARY",
        "vacation":           "DISCRETIONARY",
        "wedding":            "DISCRETIONARY",
        "moving":             "DISCRETIONARY",
        "small_business":     "BUSINESS",
        "medical":            "LIFE_EVENT",
        "other":              "LIFE_EVENT",
    }

    grp = df["target"].map(GROUPS)
    gle = LabelEncoder()
    yg = gle.fit_transform(grp)
    Kg = len(gle.classes_)

    print(f"\n  {Kg} groups:")
    for nm, c in pd.Series(grp).value_counts().items():
        print(f"    {nm:16s} {c:>9,}   ({100*c/len(grp):5.1f}%)")

    Xg_tr, Xg_te, yg_tr, yg_te = train_test_split(
        X, yg, test_size=TEST_SIZE, random_state=SEED, stratify=yg)
    Xg_bal, yg_bal = balance(Xg_tr, yg_tr, TARGET_PER_CLASS)

    if HAS_XGB:
        gmodel = XGBClassifier(
            n_estimators=300, max_depth=8, learning_rate=0.1,
            subsample=0.8, colsample_bytree=0.8,
            objective="multi:softprob", num_class=Kg,
            tree_method="hist", random_state=SEED, n_jobs=-1, verbosity=0)
    else:
        gmodel = RandomForestClassifier(
            n_estimators=150, max_depth=20, min_samples_leaf=4,
            class_weight="balanced", random_state=SEED, n_jobs=-1)

    print("\n  Training grouped model ...", end=" ", flush=True)
    gmodel.fit(Xg_bal, yg_bal)
    print("done")

    pg = gmodel.predict_proba(Xg_te)
    r5 = evaluate(pg, yg_te, "5-group model", Kg)

    lift5 = r5["macro_f1"] / (1 / Kg)
    lift13 = best["macro_f1"] / (1 / K)

    print(f"\n  {'':22s} {'macroF1':>9s} {'top1':>7s} {'chance':>8s} {'lift':>7s}")
    print("  " + "-" * 58)
    print(f"  {'13 classes':22s} {best['macro_f1']:9.4f} {best['top1']:7.3f} "
          f"{1/K:8.4f} {lift13:6.2f}x")
    print(f"  {'5 groups':22s} {r5['macro_f1']:9.4f} {r5['top1']:7.3f} "
          f"{1/Kg:8.4f} {lift5:6.2f}x")
    winner = "5 groups" if lift5 > lift13 else "13 classes"
    print(f"\n  -> {winner} wins on lift-over-chance. Carry that into Phase 4.")

    # ------------------------------------------------------------ save
    banner("SAVING ARTIFACTS", "-")

    best_model, best_scaler = fitted[best["model"]]
    joblib.dump({
        "model": best_model,
        "scaler": best_scaler,
        "label_encoder": le,
        "features": list(X.columns),
        "model_name": best["model"],
        "group_map": GROUPS,
    }, os.path.join(HERE, "phase3_best_model.pkl"))
    print("  phase3_best_model.pkl")

    # Phase 4 must reuse the IDENTICAL split, or the comparison is meaningless
    os.makedirs(SPLIT_DIR, exist_ok=True)
    if parquet_available():
        X_tr.to_parquet(os.path.join(SPLIT_DIR, "X_train.parquet"))
        X_te.to_parquet(os.path.join(SPLIT_DIR, "X_test.parquet"))
    else:
        X_tr.to_csv(os.path.join(SPLIT_DIR, "X_train.csv"), index=False)
        X_te.to_csv(os.path.join(SPLIT_DIR, "X_test.csv"), index=False)
    np.save(os.path.join(SPLIT_DIR, "y_train.npy"), y_tr)
    np.save(os.path.join(SPLIT_DIR, "y_test.npy"), y_te)
    print("  phase3_split/   (so Phase 4 compares like-for-like)")

    payload = {
        "phase": 3,
        "rows_used": int(len(df)),
        "n_classes": K,
        "class_names": class_names,
        "random_chance_macro_f1": 1 / K,
        "results": results,
        "best_model": best["model"],
        "collapse_experiment": {
            "n_groups": Kg,
            "group_names": list(gle.classes_),
            "macro_f1": r5["macro_f1"],
            "top1": r5["top1"],
            "lift_over_chance_5group": lift5,
            "lift_over_chance_13class": lift13,
        },
    }
    with open(os.path.join(HERE, "phase3_results.json"), "w") as f:
        json.dump(payload, f, indent=2)
    print("  phase3_results.json")

    banner(f"PHASE 3 COMPLETE  ({time.time() - t_start:.0f}s)"
           f"   ->   now run  python phase4.py")


if __name__ == "__main__":
    main()
