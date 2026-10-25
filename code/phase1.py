"""
================================================================================
PHASE 1: CORE RECOMMENDER MODEL
================================================================================
Goal: Build a baseline recommender system that predicts which financial 
product to recommend based on customer demographics and behavioral signals.

Steps:
  1. Load and explore data
  2. Split into train/test
  3. Scale features
  4. Train Random Forest classifier
  5. Evaluate model performance
  6. Analyze feature importance
  7. Make sample predictions

================================================================================
"""

# ============================================================================
# STEP 1: IMPORTS
# ============================================================================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Sklearn imports
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, 
    precision_score, 
    recall_score, 
    f1_score,
    confusion_matrix, 
    classification_report,
    roc_auc_score
)

# For saving model
import joblib
import os

# outputs land next to this script (was a hardcoded Colab /content/ path)
HERE = os.path.dirname(os.path.abspath(__file__))

print("✓ All imports successful\n")

# ============================================================================
# STEP 2: LOAD DATA
# ============================================================================
print("="*80)
print("STEP 2: LOAD DATA")
print("="*80)

# NEW (Your local Windows path):
df = pd.read_csv(os.path.join(HERE, 'lending_club_model_ready.csv'))

print(f"Dataset shape: {df.shape}")
print(f"Columns: {list(df.columns)}\n")

# Display first few rows
print("First 5 rows:")
print(df.head())
print()

# ============================================================================
# STEP 3: EXPLORE DATA
# ============================================================================
print("="*80)
print("STEP 3: DATA EXPLORATION")
print("="*80)

print(f"\nData Info:")
print(df.info())

print(f"\nBasic Statistics:")
print(df.describe())

print(f"\nMissing Values:")
missing = df.isnull().sum()
if missing.sum() > 0:
    print(missing[missing > 0])
else:
    print("No missing values ✓")

print(f"\nTarget Distribution (What we're predicting):")
target_counts = df['target'].value_counts()
print(target_counts)
print(f"\nClass balance:")
for target, count in target_counts.items():
    pct = (count / len(df)) * 100
    print(f"  {target:30s}: {count:8,} ({pct:5.2f}%)")

# ============================================================================
# STEP 4: PREPARE FEATURES AND TARGET
# ============================================================================
print("\n" + "="*80)
print("STEP 4: PREPARE FEATURES AND TARGET")
print("="*80)

# Separate features from target~
# Features: All columns except target (the product name)
# Target: The product to predict
feature_cols = [col for col in df.columns if col not in ['target', 'target_encoded']]
X = df[feature_cols]
from sklearn.preprocessing import LabelEncoder
# Encode target variable
le = LabelEncoder()
y = le.fit_transform(df['target'])

print(f"\nFeatures (X) shape: {X.shape}")
print(f"  {len(feature_cols)} features for each customer")
print(f"\nFeatures list:")
for i, col in enumerate(feature_cols, 1):
    print(f"  {i:2d}. {col}")

print(f"\nTarget (y) shape: {y.shape}")
print(f"  13 loan products to predict")
print(f"  Target values: {sorted(np.unique(y))}")
# ============================================================================
# STEP 5: TRAIN-TEST SPLIT
# ============================================================================
print("\n" + "="*80)
print("STEP 5: TRAIN-TEST SPLIT (80% Train, 20% Test)")
print("="*80)

# Split data: 80% for training, 20% for testing
# stratify=y ensures each class is represented equally in train/test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, 
    test_size=0.2,           # 20% for testing
    random_state=42,         # For reproducibility
    stratify=y               # Maintain class distribution
)

print(f"\nTraining set:")
print(f"  X_train shape: {X_train.shape}")
print(f"  y_train shape: {y_train.shape}")
print(f"  Samples: {len(X_train):,} customers")

print(f"\nTest set:")
print(f"  X_test shape: {X_test.shape}")
print(f"  y_test shape: {y_test.shape}")
print(f"  Samples: {len(X_test):,} customers")

print(f"\nClass distribution in training set:")
y_train = pd.Series(y_train, index=X_train.index)
y_test = pd.Series(y_test, index=X_test.index)
for encoded_val in sorted(np.unique(y_train)):
    count = (y_train == encoded_val).sum()
    pct = (count / len(y_train)) * 100
    print(f"  Class {encoded_val}: {count:7,} ({pct:5.2f}%)")

# ============================================================================
# STEP 6: FEATURE SCALING
# ============================================================================
print("\n" + "="*80)
print("STEP 6: FEATURE SCALING (Standardization)")
print("="*80)

# Standardize features: mean=0, std=1
# This helps the model train faster and more evenly
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)      # Fit on train data
X_test_scaled = scaler.transform(X_test)            # Transform test data

print(f"\nScaling statistics:")
print(f"  Before scaling - X_train mean: {X_train.values.mean():.4f}, std: {X_train.values.std():.4f}")
print(f"  After scaling  - X_train mean: {X_train_scaled.mean():.4f}, std: {X_train_scaled.std():.4f}")

print(f"\n✓ Features scaled successfully")
print(f"  Training set: {X_train_scaled.shape}")
print(f"  Test set: {X_test_scaled.shape}")

# ============================================================================
# STEP 7: TRAIN RANDOM FOREST CLASSIFIER
# ============================================================================
print("\n" + "="*80)
print("STEP 7: TRAIN RANDOM FOREST CLASSIFIER")
print("="*80)

print(f"\nModel Configuration:")
print(f"  Algorithm: Random Forest Classifier")
print(f"  Number of trees: 100")
print(f"  Max depth: 15 (prevents overfitting)")
print(f"  Random state: 42 (reproducibility)")
print(f"  Jobs: -1 (use all CPU cores)")

print(f"\nTraining in progress...")
start_time = datetime.now()

# Create and train the model
rf_model = RandomForestClassifier(
    n_estimators=100,        # Number of trees
    random_state=42,         # Reproducibility
    n_jobs=-1,              # Use all CPU cores
    max_depth=15,           # Limit depth to prevent overfitting
    min_samples_split=10,   # Minimum samples to split a node
    min_samples_leaf=4      # Minimum samples in leaf node
)

# Fit the model on training data
rf_model.fit(X_train_scaled, y_train)

end_time = datetime.now()
training_time = (end_time - start_time).total_seconds()

print(f"✓ Model trained successfully!")
print(f"  Training time: {training_time:.2f} seconds")

# ============================================================================
# STEP 8: MAKE PREDICTIONS
# ============================================================================
print("\n" + "="*80)
print("STEP 8: MAKE PREDICTIONS ON TEST SET")
print("="*80)

print(f"\nGenerating predictions...")
y_pred = rf_model.predict(X_test_scaled)              # Class prediction
y_pred_proba = rf_model.predict_proba(X_test_scaled)  # Probability for each class

print(f"✓ Predictions generated")
print(f"  Prediction shape: {y_pred.shape}")
print(f"  Probability shape: {y_pred_proba.shape}")
print(f"  (Each row has 13 probabilities for 13 products)")

# ============================================================================
# STEP 9: MODEL EVALUATION
# ============================================================================
print("\n" + "="*80)
print("STEP 9: MODEL EVALUATION METRICS")
print("="*80)

# Overall accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"\nOverall Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
print(f"  → Model predicts the correct product {accuracy*100:.2f}% of the time")

# Precision, Recall, F1 (weighted averages for imbalanced data)
precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)

print(f"\nWeighted Averages (accounting for class imbalance):")
print(f"  Precision: {precision:.4f}")
print(f"  Recall:    {recall:.4f}")
print(f"  F1-Score:  {f1:.4f}")

# Cross-validation
print(f"\nCross-Validation (5-fold):")
cv_scores = cross_val_score(rf_model, X_train_scaled, y_train, cv=5, scoring='accuracy')
print(f"  Fold scores: {[f'{score:.4f}' for score in cv_scores]}")
print(f"  Mean CV accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")

# ============================================================================
# STEP 10: DETAILED CLASSIFICATION REPORT
# ============================================================================
print("\n" + "="*80)
print("STEP 10: CLASSIFICATION REPORT (Per Product)")
print("="*80)

# Create mapping from encoded to product name
encoding_map = {}
for idx, product in enumerate(df['target'].unique()):
    encoding_map[idx] = product

print(f"\nDetailed metrics for each loan product:")
print(classification_report(y_test, y_pred, digits=4))

# Print product-specific accuracy
print(f"\nPer-Product Accuracy:")
y_test = pd.Series(y_test, index=X_test.index)
for encoded_val in sorted(y_test.unique()):
    mask = y_test == encoded_val
    if mask.sum() > 0:
        product_acc = accuracy_score(y_test[mask], y_pred[mask])
        product_name = encoding_map[encoded_val]
        count = mask.sum()
        print(f"  {product_name:30s}: {product_acc:.4f} ({count:5,} samples)")

# ============================================================================
# STEP 11: CONFUSION MATRIX
# ============================================================================
print("\n" + "="*80)
print("STEP 11: CONFUSION MATRIX")
print("="*80)

cm = confusion_matrix(y_test, y_pred)
print(f"\nConfusion Matrix shape: {cm.shape} (13x13 products)")
print(f"Diagonal = correct predictions")
print(f"Off-diagonal = misclassifications")

# ============================================================================
# STEP 12: FEATURE IMPORTANCE ANALYSIS
# ============================================================================
print("\n" + "="*80)
print("STEP 12: FEATURE IMPORTANCE ANALYSIS")
print("="*80)

# Extract feature importances from the trained model
feature_importance = pd.DataFrame({
    'feature': feature_cols,
    'importance': rf_model.feature_importances_
}).sort_values('importance', ascending=False)

print(f"\nTop 15 Most Important Features:")
print(f"(These drive the product recommendations)\n")
for idx, row in feature_importance.head(15).iterrows():
    feature_name = row['feature']
    importance_val = row['importance']
    # Create visual bar
    bar_length = int(importance_val * 100)
    bar = '█' * bar_length
    print(f"  {feature_name:30s} {importance_val:.6f}  {bar}")

# Save feature importance
feature_importance.to_csv(os.path.join(HERE, 'feature_importance.csv'), index=False)
print(f"\n✓ Feature importance saved to feature_importance.csv")

# ============================================================================
# STEP 13: VISUALIZATION
# ============================================================================
print("\n" + "="*80)
print("STEP 13: CREATING VISUALIZATIONS")
print("="*80)

# Create figure with multiple subplots
fig, axes = plt.subplots(2, 2, figsize=(15, 12))

# 1. Feature Importance (Top 15)
ax1 = axes[0, 0]
top_features = feature_importance.head(15)
ax1.barh(range(len(top_features)), top_features['importance'].values, color='steelblue')
ax1.set_yticks(range(len(top_features)))
ax1.set_yticklabels(top_features['feature'].values)
ax1.set_xlabel('Importance Score')
ax1.set_title('Top 15 Features Driving Product Recommendations', fontsize=12, fontweight='bold')
ax1.invert_yaxis()

# 2. Model Accuracy
ax2 = axes[0, 1]
metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
values = [accuracy, precision, recall, f1]
colors = ['#2ecc71', '#3498db', '#e74c3c', '#f39c12']
bars = ax2.bar(metrics, values, color=colors, alpha=0.7, edgecolor='black')
ax2.set_ylim([0, 1])
ax2.set_ylabel('Score')
ax2.set_title('Model Performance Metrics', fontsize=12, fontweight='bold')
# Add value labels on bars
for bar in bars:
    height = bar.get_height()
    ax2.text(bar.get_x() + bar.get_width()/2., height,
            f'{height:.3f}',
            ha='center', va='bottom', fontweight='bold')

# 3. Cross-validation scores
ax3 = axes[1, 0]
fold_nums = range(1, len(cv_scores) + 1)
ax3.plot(fold_nums, cv_scores, marker='o', linestyle='-', linewidth=2, markersize=8, color='#9b59b6')
ax3.axhline(y=cv_scores.mean(), color='r', linestyle='--', label=f'Mean: {cv_scores.mean():.4f}')
ax3.fill_between(fold_nums, 
                  cv_scores.mean() - cv_scores.std(),
                  cv_scores.mean() + cv_scores.std(),
                  alpha=0.2, color='#9b59b6')
ax3.set_xlabel('Fold Number')
ax3.set_ylabel('Accuracy')
ax3.set_title('5-Fold Cross-Validation Scores', fontsize=12, fontweight='bold')
ax3.set_ylim([0, 1])
ax3.legend()
ax3.grid(True, alpha=0.3)

# 4. Class distribution in test set
ax4 = axes[1, 1]
class_dist = pd.Series(y_test).value_counts().sort_index()
product_names = [encoding_map[i] for i in class_dist.index]
ax4.bar(range(len(class_dist)), class_dist.values, color='coral', alpha=0.7, edgecolor='black')
ax4.set_xticks(range(len(class_dist)))
ax4.set_xticklabels([name[:12] for name in product_names], rotation=45, ha='right')
ax4.set_ylabel('Count')
ax4.set_title('Test Set Class Distribution', fontsize=12, fontweight='bold')

plt.tight_layout()
plt.savefig(os.path.join(HERE, 'phase1_model_evaluation.png'), dpi=300, bbox_inches='tight')
print(f"✓ Visualization saved: phase1_model_evaluation.png")
plt.show()

# ============================================================================
# STEP 14: SAMPLE PREDICTIONS
# ============================================================================
print("\n" + "="*80)
print("STEP 14: SAMPLE PREDICTIONS (Show how model works)")
print("="*80)

# Select 5 random test samples
sample_indices = np.random.choice(len(X_test), size=5, replace=False)

print(f"\n5 Sample Predictions:\n")
for i, idx in enumerate(sample_indices, 1):
    customer_features = X_test.iloc[idx]
    actual_product_code = y_test.iloc[idx]
    predicted_product_code = y_pred[idx]
    confidence = y_pred_proba[idx][predicted_product_code]
    
    actual_product = encoding_map[actual_product_code]
    predicted_product = encoding_map[predicted_product_code]
    
    match = "✓ CORRECT" if actual_product_code == predicted_product_code else "✗ WRONG"
    
    print(f"Sample {i}:")
    print(f"  Actual product:      {actual_product}")
    print(f"  Predicted product:   {predicted_product}")
    print(f"  Confidence:          {confidence*100:.2f}%")
    print(f"  Result:              {match}")
    print(f"  Customer profile:")
    print(f"    - Annual Income:   ${customer_features['annual_inc']:,.0f}")
    print(f"    - Employment yrs:  {customer_features['emp_length_numeric']:.1f}")
    print(f"    - FICO Score:      {customer_features['fico_range_low']:.0f}-{customer_features['fico_range_high']:.0f}")
    print(f"    - DTI:             {customer_features['dti']:.2f}%")
    print(f"    - Loan Amount:     ${customer_features['loan_amnt']:,.0f}")
    print()

# ============================================================================
# STEP 15: SAVE MODEL AND SCALER
# ============================================================================
print("="*80)
print("STEP 15: SAVE MODEL AND SCALER FOR NEXT PHASES")
print("="*80)

# Save the trained model
joblib.dump(rf_model, r'C:\Users\jay4l\Desktop\RecommenderSysProject\phase1_model.pkl')
print(f"✓ Model saved: phase1_model.pkl")

# Save the scaler
joblib.dump(scaler, os.path.join(HERE, 'phase1_scaler.pkl'))
print(f"✓ Scaler saved: phase1_scaler.pkl")

# Save feature columns
np.save(os.path.join(HERE, 'feature_columns.npy'), np.array(feature_cols))
print(f"✓ Feature columns saved: feature_columns .npy")

# Save encoding map
import json
joblib.dump(encoding_map, os.path.join(HERE, 'encoding_map.pkl'))
print(f"✓ Encoding map saved: encoding_map.pkl")

# ============================================================================
# SUMMARY
# ============================================================================
print("\n" + "="*80)
print("PHASE 1 COMPLETE - SUMMARY")
print("="*80)

print(f"""
MODEL PERFORMANCE:
  • Accuracy:         {accuracy*100:.2f}%
  • Precision:        {precision:.4f}
  • Recall:           {recall:.4f}
  • F1-Score:         {f1:.4f}
  • CV Mean:          {cv_scores.mean():.4f}

DATASET:
  • Total samples:    {len(df):,}
  • Training:         {len(X_train):,}
  • Testing:          {len(X_test):,}
  • Features:         {len(feature_cols)}
  • Products:         13

TOP DRIVER FEATURES:
  1. {feature_importance.iloc[0]['feature']}
  2. {feature_importance.iloc[1]['feature']}
  3. {feature_importance.iloc[2]['feature']}

FILES SAVED:
  ✓ phase1_model.pkl (trained model)
  ✓ phase1_scaler.pkl (feature scaler)
  ✓ feature_columns.npy (feature names)
  ✓ encoding_map.pkl (product mappings)
  ✓ feature_importance.csv (importance scores)
  ✓ phase1_model_evaluation.png (visualizations)

NEXT PHASES:
  Phase 2: Integrate Macro Economic Data (interest rates, inflation)
  Phase 3: Integrate Sector Performance (stock/ETF trends)
  Phase 4: Add Knowledge-Based Constraints (eligibility rules)
  Phase 5: Create Final Recommendation Function
""")

print("="*80)
print("Ready for Phase 2! 🚀")
print("="*80)