#!/usr/bin/env python3
"""
Trains and evaluates LightGBM for citation count prediction (threshold: citation_count > 5).
Uses early stopping on validation AUC. Prints test set accuracy, balanced accuracy, ROC-AUC,
classification report, confusion matrix, and top-30 features by gain and split importance.
Input:  data/90k_arxiv_citation_prediction_splits/
"""

import re
import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
)

from lightgbm import LGBMClassifier, early_stopping, log_evaluation


TRAIN_PATH = "data/90k_arxiv_citation_prediction_splits/90k_arxiv_citation_prediction_train.csv"
VAL_PATH = "data/90k_arxiv_citation_prediction_splits/90k_arxiv_citation_prediction_val.csv"
TEST_PATH = "data/90k_arxiv_citation_prediction_splits/90k_arxiv_citation_prediction_test.csv"

THRESHOLD = 5
citation_col = "citation_count"

drop_cols = [
    "arxiv_id",
    "corpus_id",
    "citation_count",
    "citation_bin",
]


def clean_lgbm_feature_names(columns):
    cleaned = []
    seen = {}

    for col in columns:
        new_col = re.sub(r"[^A-Za-z0-9_]+", "_", col)
        new_col = re.sub(r"_+", "_", new_col).strip("_")

        if new_col in seen:
            seen[new_col] += 1
            new_col = f"{new_col}_{seen[new_col]}"
        else:
            seen[new_col] = 0

        cleaned.append(new_col)

    return cleaned


# ============================================================
# Load data
# ============================================================

train_df = pd.read_csv(TRAIN_PATH, low_memory=False, dtype={"arxiv_id": str})
val_df = pd.read_csv(VAL_PATH, low_memory=False, dtype={"arxiv_id": str})
test_df = pd.read_csv(TEST_PATH, low_memory=False, dtype={"arxiv_id": str})

print(f"Train shape: {train_df.shape}")
print(f"Val shape:   {val_df.shape}")
print(f"Test shape:  {test_df.shape}")


# ============================================================
# Target: citation_count > 5
# ============================================================

y_train = (train_df[citation_col] > THRESHOLD).astype(int)
y_val = (val_df[citation_col] > THRESHOLD).astype(int)
y_test = (test_df[citation_col] > THRESHOLD).astype(int)

print(f"\nCitation Prediction: citation_count > {THRESHOLD}")
print(f"Train positive rate: {y_train.mean():.4f}")
print(f"Val positive rate:   {y_val.mean():.4f}")
print(f"Test positive rate:  {y_test.mean():.4f}")


# ============================================================
# Prepare features
# ============================================================

X_train = train_df.drop(columns=drop_cols, errors="ignore")
X_val = val_df.drop(columns=drop_cols, errors="ignore")
X_test = test_df.drop(columns=drop_cols, errors="ignore")

# Ensure same columns across splits
X_val = X_val[X_train.columns]
X_test = X_test[X_train.columns]

# Ensure all feature columns are numeric
for col in X_train.columns:
    X_train[col] = pd.to_numeric(X_train[col], errors="coerce")
    X_val[col] = pd.to_numeric(X_val[col], errors="coerce")
    X_test[col] = pd.to_numeric(X_test[col], errors="coerce")

X_train = X_train.replace([np.inf, -np.inf], np.nan)
X_val = X_val.replace([np.inf, -np.inf], np.nan)
X_test = X_test.replace([np.inf, -np.inf], np.nan)

medians = X_train.median(numeric_only=True)

X_train = X_train.fillna(medians)
X_val = X_val.fillna(medians)
X_test = X_test.fillna(medians)


# ============================================================
# Feature scaling
# ============================================================

binary_cols = [
    col for col in X_train.columns
    if X_train[col].dropna().nunique() <= 2
]

# Scaling binary/dummy columns would change their meaning; tree models are scale-invariant anyway but logistic reg is not
numeric_cols_to_scale = [
    col for col in X_train.columns
    if col not in binary_cols
]

print(f"\nBinary / dummy columns not scaled: {len(binary_cols)}")
print(f"Continuous numeric columns scaled: {len(numeric_cols_to_scale)}")

scaler = StandardScaler()

X_train_scaled = X_train.copy()
X_val_scaled = X_val.copy()
X_test_scaled = X_test.copy()

X_train_scaled[numeric_cols_to_scale] = scaler.fit_transform(
    X_train[numeric_cols_to_scale]
)

X_val_scaled[numeric_cols_to_scale] = scaler.transform(
    X_val[numeric_cols_to_scale]
)

X_test_scaled[numeric_cols_to_scale] = scaler.transform(
    X_test[numeric_cols_to_scale]
)


# ============================================================
# Clean feature names for LightGBM
# ============================================================

cleaned_feature_names = clean_lgbm_feature_names(X_train_scaled.columns)

X_train_scaled.columns = cleaned_feature_names
X_val_scaled.columns = cleaned_feature_names
X_test_scaled.columns = cleaned_feature_names


# ============================================================
# Train selected LightGBM model
# ============================================================

model = LGBMClassifier(
    objective="binary",
    random_state=42,
    class_weight="balanced",
    n_jobs=-1,
    verbosity=-1,
    n_estimators=500,
    learning_rate=0.05,
    num_leaves=31,
    max_depth=-1,  # unlimited depth; tree complexity is controlled by num_leaves instead
)

model.fit(
    X_train_scaled,
    y_train,
    eval_set=[(X_val_scaled, y_val)],
    eval_metric="auc",
    callbacks=[
        early_stopping(stopping_rounds=50),  # 50 rounds of no AUC improvement on val triggers early stop
        log_evaluation(period=50),
    ],
)


# ============================================================
# Final evaluation on test set
# ============================================================

y_pred = model.predict(X_test_scaled)
y_proba = model.predict_proba(X_test_scaled)[:, 1]

acc = accuracy_score(y_test, y_pred)
bal_acc = balanced_accuracy_score(y_test, y_pred)
auc = roc_auc_score(y_test, y_proba)

print("\n" + "=" * 80)
print("Final Test Set Evaluation")
print("=" * 80)

print(f"Accuracy:          {acc:.4f}")
print(f"Balanced accuracy: {bal_acc:.4f}")
print(f"ROC-AUC:           {auc:.4f}")

print("\nClassification report:")
print(
    classification_report(
        y_test,
        y_pred,
        target_names=[
            f"citation_count <= {THRESHOLD}",
            f"citation_count > {THRESHOLD}",
        ],
        digits=4,
    )
)

print("\nConfusion matrix:")
print(confusion_matrix(y_test, y_pred))


# ============================================================
# Feature importance: Gain-based and Split-based
# ============================================================

feature_importance = pd.DataFrame({
    "feature": X_train_scaled.columns,
    "gain_importance": model.booster_.feature_importance(
        importance_type="gain"
    ),
    "split_importance": model.booster_.feature_importance(
        importance_type="split"
    ),
})

feature_importance["gain_percent"] = (
    feature_importance["gain_importance"]
    / feature_importance["gain_importance"].sum()
) * 100

feature_importance["split_percent"] = (
    feature_importance["split_importance"]
    / feature_importance["split_importance"].sum()
) * 100

gain_importance = feature_importance.sort_values(
    "gain_importance",
    ascending=False
)

split_importance = feature_importance.sort_values(
    "split_importance",
    ascending=False
)

print("\nTop 30 features: Gain Importance")
print(gain_importance.head(30).to_string(index=False))

print("\nTop 30 features: Split Importance")
print(split_importance.head(30).to_string(index=False))


# ============================================================
# Summary
# ============================================================

results_df = pd.DataFrame([{
    "threshold": THRESHOLD,
    "train_positive_rate": y_train.mean(),
    "val_positive_rate": y_val.mean(),
    "test_positive_rate": y_test.mean(),
    "accuracy": acc,
    "balanced_accuracy": bal_acc,
    "roc_auc": auc,
    "best_iteration": model.best_iteration_,
}])

print("\n" + "=" * 80)
print("Summary")
print("=" * 80)
print(results_df.to_string(index=False))