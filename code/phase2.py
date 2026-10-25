"""
PHASE 2 - Class Imbalance Fix
Recommender System Project

Phase 1 problem: 57% accuracy but minority classes had 0% recall.
Phase 2 fix: hybrid resampling (undersample majority + SMOTE minority),
             class_weight='balanced', macro F1 as the primary metric.

Note on strategy: the raw imbalance is ~917:1 (debt_consolidation 1.1M vs
renewable_energy 1.2k). Pure SMOTE would inflate training data to ~11.5M rows.
Instead we cap every class at TARGET_PER_CLASS: majority classes get randomly
undersampled down to it, minority classes get SMOTE'd up to it.

IMPORTANT: resampling is applied to the TRAINING SET ONLY. The test set keeps
its real-world imbalanced distribution so the metrics stay honest.
"""

import json
import numpy as np
import pandas as pd
import joblib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    f1_score, precision_score, recall_score, accuracy_score,
    confusion_matrix, precision_recall_fscore_support
)
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline as ImbPipeline

# ---------------------------------------------------------------- config
DATA_PATH = "lending_club_model_ready.csv"   # change if your path differs
TARGET_PER_CLASS = 25_000                     # balanced size per class
RANDOM_STATE = 42
RUN_CV = False                                 # set False to skip 5-fold CV (slow)

rng = np.random.RandomState(RANDOM_STATE)

print("=" * 72)
print("PHASE 2 : CLASS IMBALANCE FIX")
print("=" * 72)

# ---------------------------------------------------------------- 1. load
df = pd.read_csv(DATA_PATH)
print(f"\nLoaded {DATA_PATH}  ->  {df.shape[0]:,} rows x {df.shape[1]} cols")

X = df.drop("target", axis=1)
y_raw = df["target"]

le = LabelEncoder()
y = le.fit_transform(y_raw)
class_names = list(le.classes_)

print(f"Features : {X.shape[1]}")
print(f"Classes  : {len(class_names)}")

print("\n--- CLASS DISTRIBUTION (RAW) ---")
counts = pd.Series(y_raw).value_counts()
for name, c in counts.items():
    print(f"  {name:22s} {c:>9,}  ({100*c/len(y_raw):5.2f}%)")
print(f"\n  Imbalance ratio: {counts.max()/counts.min():.0f} : 1")

# ---------------------------------------------------------------- 2. split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
)
print(f"\nTrain: {X_train.shape[0]:,}   Test: {X_test.shape[0]:,}  (stratified)")
print("Test set keeps the original imbalanced distribution.")

# ---------------------------------------------------------------- 3. resample
print("\n" + "=" * 72)
print(f"HYBRID RESAMPLING  (target = {TARGET_PER_CLASS:,} per class)")
print("=" * 72)

train_counts = pd.Series(y_train).value_counts()

# classes ABOVE target -> undersample down to target
under_strategy = {c: TARGET_PER_CLASS for c, n in train_counts.items()
                  if n > TARGET_PER_CLASS}
# classes BELOW target -> SMOTE up to target
over_strategy = {c: TARGET_PER_CLASS for c, n in train_counts.items()
                 if n < TARGET_PER_CLASS}

print(f"\n  Undersampling {len(under_strategy)} majority classes")
print(f"  SMOTE-ing     {len(over_strategy)} minority classes")

# smallest minority class decides how many neighbours SMOTE can use
smallest = train_counts.min()
k_neighbors = max(1, min(5, int(smallest) - 1))
print(f"  Smallest class in train: {smallest:,}  ->  k_neighbors={k_neighbors}")

rus = RandomUnderSampler(sampling_strategy=under_strategy,
                         random_state=RANDOM_STATE)
X_tmp, y_tmp = rus.fit_resample(X_train, y_train)

sm = SMOTE(sampling_strategy=over_strategy,
           k_neighbors=k_neighbors, random_state=RANDOM_STATE)
X_bal, y_bal = sm.fit_resample(X_tmp, y_tmp)

print(f"\n  {X_train.shape[0]:,} -> {X_tmp.shape[0]:,} (after undersample)"
      f" -> {X_bal.shape[0]:,} (after SMOTE)")
print("\n--- CLASS DISTRIBUTION (BALANCED TRAIN) ---")
for c, n in sorted(pd.Series(y_bal).value_counts().items()):
    print(f"  {class_names[c]:22s} {n:>9,}")

# ---------------------------------------------------------------- 4. train
print("\n" + "=" * 72)
print("TRAIN RANDOM FOREST  (class_weight='balanced')")
print("=" * 72)

rf = RandomForestClassifier(
    n_estimators=150,
    max_depth=20,
    min_samples_split=10,
    min_samples_leaf=4,
    class_weight="balanced",
    random_state=RANDOM_STATE,
    n_jobs=-1,
)
rf.fit(X_bal, y_bal)
print(f"\n  Trained on {X_bal.shape[0]:,} balanced samples.")

# ---------------------------------------------------------------- 5. evaluate
print("\n" + "=" * 72)
print("TEST SET EVALUATION")
print("=" * 72)

y_pred = rf.predict(X_test)

macro_f1 = f1_score(y_test, y_pred, average="macro", zero_division=0)
macro_p = precision_score(y_test, y_pred, average="macro", zero_division=0)
macro_r = recall_score(y_test, y_pred, average="macro", zero_division=0)
weighted_f1 = f1_score(y_test, y_pred, average="weighted", zero_division=0)
acc = accuracy_score(y_test, y_pred)

print(f"\n  MACRO F1 (primary) : {macro_f1:.4f}")
print(f"  Macro precision    : {macro_p:.4f}")
print(f"  Macro recall       : {macro_r:.4f}")
print(f"  Weighted F1        : {weighted_f1:.4f}")
print(f"  Accuracy           : {acc:.4f}")

# per class
p_c, r_c, f_c, sup = precision_recall_fscore_support(
    y_test, y_pred, zero_division=0, labels=range(len(class_names))
)

print("\n--- PER-CLASS METRICS ---")
print(f"  {'class':22s} {'prec':>7s} {'recall':>7s} {'f1':>7s} {'support':>9s}")
print("  " + "-" * 55)
order = np.argsort(-sup)
for i in order:
    print(f"  {class_names[i]:22s} {p_c[i]:7.3f} {r_c[i]:7.3f} "
          f"{f_c[i]:7.3f} {sup[i]:>9,}")

dead = [class_names[i] for i in range(len(class_names)) if r_c[i] == 0]
if dead:
    print(f"\n  WARNING: {len(dead)} class(es) still at 0% recall: {dead}")
else:
    print(f"\n  All {len(class_names)} classes now have non-zero recall "
          f"(Phase 1 had many at 0%).")

# ---------------------------------------------------------------- 6. confusion
cm = confusion_matrix(y_test, y_pred, labels=range(len(class_names)))
cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)

fig, ax = plt.subplots(figsize=(11, 9))
im = ax.imshow(cm_norm, cmap="Blues", vmin=0, vmax=1)
ax.set_xticks(range(len(class_names)))
ax.set_yticks(range(len(class_names)))
ax.set_xticklabels(class_names, rotation=45, ha="right", fontsize=8)
ax.set_yticklabels(class_names, fontsize=8)
ax.set_xlabel("Predicted")
ax.set_ylabel("True")
ax.set_title(f"Phase 2 — Row-normalised confusion matrix (Macro F1 = {macro_f1:.3f})")
for i in range(len(class_names)):
    for j in range(len(class_names)):
        if cm_norm[i, j] > 0.01:
            ax.text(j, i, f"{cm_norm[i, j]:.2f}", ha="center", va="center",
                    fontsize=6,
                    color="white" if cm_norm[i, j] > 0.5 else "black")
fig.colorbar(im, ax=ax, shrink=0.8)
fig.tight_layout()
fig.savefig("phase2_confusion_matrix.png", dpi=120)
plt.close(fig)
print("\n  Saved phase2_confusion_matrix.png")

# ---------------------------------------------------------------- 7. features
imp = rf.feature_importances_
idx = np.argsort(imp)[::-1]
print("\n--- FEATURE IMPORTANCE (all 16) ---")
for rank, i in enumerate(idx, 1):
    bar = "#" * int(imp[i] * 100)
    print(f"  {rank:2d}. {X.columns[i]:28s} {imp[i]:.4f}  {bar}")

fig, ax = plt.subplots(figsize=(9, 6))
ax.barh(range(len(idx)), imp[idx][::-1], color="#4C72B0")
ax.set_yticks(range(len(idx)))
ax.set_yticklabels([X.columns[i] for i in idx][::-1], fontsize=9)
ax.set_xlabel("Importance")
ax.set_title("Phase 2 — Feature importance")
fig.tight_layout()
fig.savefig("phase2_feature_importance.png", dpi=120)
plt.close(fig)
print("\n  Saved phase2_feature_importance.png")

# ---------------------------------------------------------------- 8. CV
cv_mean = cv_std = None
if RUN_CV:
    print("\n" + "=" * 72)
    print("STRATIFIED 5-FOLD CROSS-VALIDATION")
    print("=" * 72)

    # subsample for CV speed; resampling happens inside each fold (no leakage)
    cv_n = min(300_000, X_train.shape[0])
    sub = rng.choice(X_train.shape[0], cv_n, replace=False)
    X_cv, y_cv = X_train.iloc[sub], y_train[sub]

    cv_counts = pd.Series(y_cv).value_counts()
    cv_target = 4_000
    cv_pipe = ImbPipeline([
        ("under", RandomUnderSampler(
            sampling_strategy={c: cv_target for c, n in cv_counts.items()
                               if n > cv_target},
            random_state=RANDOM_STATE)),
        ("smote", SMOTE(
            sampling_strategy={c: cv_target for c, n in cv_counts.items()
                               if n < cv_target},
            k_neighbors=max(1, min(5, int(cv_counts.min() * 0.8) - 1)),
            random_state=RANDOM_STATE)),
        ("rf", RandomForestClassifier(
            n_estimators=100, max_depth=20, min_samples_split=10,
            min_samples_leaf=4, class_weight="balanced",
            random_state=RANDOM_STATE, n_jobs=-1)),
    ])

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    scores = cross_validate(cv_pipe, X_cv, y_cv, cv=skf,
                            scoring=["f1_macro", "precision_macro",
                                     "recall_macro", "accuracy"],
                            n_jobs=1)
    cv_mean = float(scores["test_f1_macro"].mean())
    cv_std = float(scores["test_f1_macro"].std())
    print(f"\n  (on {cv_n:,}-row subsample of train)")
    print(f"  Macro F1   : {cv_mean:.4f}  +/- {cv_std:.4f}")
    print(f"  Macro prec : {scores['test_precision_macro'].mean():.4f}")
    print(f"  Macro rec  : {scores['test_recall_macro'].mean():.4f}")
    print(f"  Accuracy   : {scores['test_accuracy'].mean():.4f}")
    print("\n  Low std across folds = stable, not a lucky split.")

# ---------------------------------------------------------------- 9. compare
print("\n" + "=" * 72)
print("PHASE 1  vs  PHASE 2")
print("=" * 72)
print(f"""
                        Phase 1              Phase 2
  ------------------------------------------------------------
  Accuracy              ~0.57                {acc:.4f}
  Macro F1              not tracked          {macro_f1:.4f}
  Macro recall          very low             {macro_r:.4f}
  Classes at 0% recall  many                 {len(dead)}
  Training data         raw (917:1)          balanced ({TARGET_PER_CLASS:,}/class)

  Accuracy is EXPECTED to drop. Phase 1's 57% came from predicting
  debt_consolidation almost every time. Phase 2 trades that hollow
  accuracy for real coverage across all 13 products - which is what a
  cross-selling recommender actually needs.
""")

# ---------------------------------------------------------------- 10. save
joblib.dump({"model": rf, "label_encoder": le,
             "feature_names": list(X.columns)}, "phase2_rf_model.pkl")

results = {
    "phase": 2,
    "macro_f1": float(macro_f1),
    "macro_precision": float(macro_p),
    "macro_recall": float(macro_r),
    "weighted_f1": float(weighted_f1),
    "accuracy": float(acc),
    "cv_macro_f1_mean": cv_mean,
    "cv_macro_f1_std": cv_std,
    "classes_zero_recall": dead,
    "target_per_class": TARGET_PER_CLASS,
    "balanced_train_rows": int(X_bal.shape[0]),
    "per_class": {
        class_names[i]: {"precision": float(p_c[i]),
                         "recall": float(r_c[i]),
                         "f1": float(f_c[i]),
                         "support": int(sup[i])}
        for i in range(len(class_names))
    },
    "feature_importance": {X.columns[i]: float(imp[i]) for i in idx},
}
with open("phase2_results.json", "w") as f:
    json.dump(results, f, indent=2)

print("  Saved phase2_rf_model.pkl")
print("  Saved phase2_results.json")
print("\n" + "=" * 72)
print("PHASE 2 COMPLETE")
print("=" * 72)
