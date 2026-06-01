#!/usr/bin/env python3

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.stats import chi2

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
)


TRAIN_PATH = "data/90k_arxiv_doi_prediction_splits/90k_arxiv_doi_prediction_train.csv"
VAL_PATH = "data/90k_arxiv_doi_prediction_splits/90k_arxiv_doi_prediction_val.csv"
TEST_PATH = "data/90k_arxiv_doi_prediction_splits/90k_arxiv_doi_prediction_test.csv"

target_col = "has_doi"

drop_cols = [
    "arxiv_id",
    "corpus_id",
    "has_doi",
    "citation_count",
]

MAX_ROWS_STATSMODELS = None
# MAX_ROWS_STATSMODELS = 20000


train_df = pd.read_csv(TRAIN_PATH, low_memory=False, dtype={"arxiv_id": str})
val_df = pd.read_csv(VAL_PATH, low_memory=False, dtype={"arxiv_id": str})
test_df = pd.read_csv(TEST_PATH, low_memory=False, dtype={"arxiv_id": str})

print(f"Train shape: {train_df.shape}")
print(f"Val shape:   {val_df.shape}")
print(f"Test shape:  {test_df.shape}")


y_train = train_df[target_col].astype(int)
y_val = val_df[target_col].astype(int)
y_test = test_df[target_col].astype(int)

print("\nLogistic Regression: DOI presence prediction")
print(f"Train positive rate: {y_train.mean():.4f}")
print(f"Val positive rate:   {y_val.mean():.4f}")
print(f"Test positive rate:  {y_test.mean():.4f}")


X_train = train_df.drop(columns=drop_cols, errors="ignore")
X_val = val_df.drop(columns=drop_cols, errors="ignore")
X_test = test_df.drop(columns=drop_cols, errors="ignore")

X_val = X_val[X_train.columns]
X_test = X_test[X_train.columns]


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


# Drop columns with no variance based on train only
constant_cols = [
    col for col in X_train.columns
    if X_train[col].nunique(dropna=True) <= 1
]

if constant_cols:
    print(f"\nDropping constant columns: {len(constant_cols)}")
    X_train = X_train.drop(columns=constant_cols)
    X_val = X_val.drop(columns=constant_cols)
    X_test = X_test.drop(columns=constant_cols)


binary_cols = [
    col for col in X_train.columns
    if X_train[col].dropna().nunique() <= 2
]

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


# Drop duplicate and highly correlated columns based on train only
X_train_scaled = X_train_scaled.loc[:, ~X_train_scaled.T.duplicated()]
X_val_scaled = X_val_scaled[X_train_scaled.columns]
X_test_scaled = X_test_scaled[X_train_scaled.columns]

corr_matrix = X_train_scaled.corr().abs()

upper = corr_matrix.where(
    np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
)

high_corr_pairs = []

for col in upper.columns:
    for row in upper.index:
        corr_val = upper.loc[row, col]

        if pd.notna(corr_val) and corr_val > 0.95:
            high_corr_pairs.append((row, col, corr_val))

print("\nHighly correlated column pairs (>0.95):")

if len(high_corr_pairs) == 0:
    print("None found.")
else:
    for row, col, corr_val in sorted(high_corr_pairs, key=lambda x: -x[2]):
        print(f"{row:40s} <-> {col:40s} : {corr_val:.6f}")

high_corr_cols = list(set([pair[1] for pair in high_corr_pairs]))

print(f"\nDropping highly correlated columns: {len(high_corr_cols)}")
print(high_corr_cols)

X_train_scaled = X_train_scaled.drop(columns=high_corr_cols)
X_val_scaled = X_val_scaled.drop(columns=high_corr_cols)
X_test_scaled = X_test_scaled.drop(columns=high_corr_cols)


if MAX_ROWS_STATSMODELS is not None and len(X_train_scaled) > MAX_ROWS_STATSMODELS:
    sample_idx = (
        pd.concat([X_train_scaled, y_train.rename("target")], axis=1)
        .groupby("target", group_keys=False)
        .apply(
            lambda x: x.sample(
                min(len(x), MAX_ROWS_STATSMODELS // 2),
                random_state=42
            )
        )
        .index
    )

    X_train_model = X_train_scaled.loc[sample_idx]
    y_train_model = y_train.loc[sample_idx]
else:
    X_train_model = X_train_scaled
    y_train_model = y_train


X_train_sm = sm.add_constant(X_train_model, has_constant="add")
X_val_sm = sm.add_constant(X_val_scaled, has_constant="add")
X_test_sm = sm.add_constant(X_test_scaled, has_constant="add")

null_model = sm.GLM(
    y_train_model,
    np.ones((len(y_train_model), 1)),
    family=sm.families.Binomial()
).fit(maxiter=200)

model = sm.GLM(
    y_train_model,
    X_train_sm,
    family=sm.families.Binomial()
).fit(maxiter=200)


y_val_proba = model.predict(X_val_sm)
y_val_pred = (y_val_proba >= 0.5).astype(int)

print("\n" + "=" * 80)
print("Validation Set Evaluation")
print("=" * 80)

print(f"Accuracy:          {accuracy_score(y_val, y_val_pred):.4f}")
print(f"Balanced accuracy: {balanced_accuracy_score(y_val, y_val_pred):.4f}")
print(f"ROC-AUC:           {roc_auc_score(y_val, y_val_proba):.4f}")


y_proba = model.predict(X_test_sm)
y_pred = (y_proba >= 0.5).astype(int)

acc = accuracy_score(y_test, y_pred)
bal_acc = balanced_accuracy_score(y_test, y_pred)
auc = roc_auc_score(y_test, y_proba)

print("\n" + "=" * 80)
print("Test Set Evaluation")
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
            "DOI absent",
            "DOI present",
        ],
        digits=4,
    )
)

print("\nConfusion matrix:")
print(confusion_matrix(y_test, y_pred))


coef_table = pd.DataFrame({
    "feature": model.params.index,
    "coef": model.params.values,
    "odds_ratio": np.exp(model.params.values),
    "p_value": model.pvalues.values,
})

coef_table["direction"] = np.where(
    coef_table["coef"] > 0,
    "positive",
    "negative"
)

coef_table["significant_0_05"] = coef_table["p_value"] < 0.05

coef_table = coef_table[coef_table["feature"] != "const"]

coef_table = coef_table.sort_values("p_value")

print("\nTop significant coefficients:")
print(coef_table.head(30).to_string(index=False))

print("\nMost positive coefficients:")
print(
    coef_table
    .sort_values("coef", ascending=False)
    .head(20)
    .to_string(index=False)
)

print("\nMost negative coefficients:")
print(
    coef_table
    .sort_values("coef", ascending=True)
    .head(20)
    .to_string(index=False)
)


lr_stat = 2 * (model.llf - null_model.llf)
lr_df = model.df_model
lr_pvalue = chi2.sf(lr_stat, lr_df)

print("\nANOVA-style likelihood ratio test:")
print(f"Null log-likelihood:  {null_model.llf:.4f}")
print(f"Model log-likelihood: {model.llf:.4f}")
print(f"LR statistic:         {lr_stat:.4f}")
print(f"df:                   {lr_df:.0f}")
print(f"p-value:              {lr_pvalue:.6g}")


results_df = pd.DataFrame([{
    "task": "DOI presence prediction",
    "train_positive_rate": y_train.mean(),
    "val_positive_rate": y_val.mean(),
    "test_positive_rate": y_test.mean(),
    "accuracy": acc,
    "balanced_accuracy": bal_acc,
    "roc_auc": auc,
    "lr_statistic": lr_stat,
    "lr_df": lr_df,
    "lr_pvalue": lr_pvalue,
}])

print("\n" + "=" * 80)
print("Summary")
print("=" * 80)
print(results_df.to_string(index=False))