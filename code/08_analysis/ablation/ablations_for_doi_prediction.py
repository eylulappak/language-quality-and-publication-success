#!/usr/bin/env python3

import re
import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, balanced_accuracy_score, roc_auc_score

from xgboost import XGBClassifier


TRAIN_PATH = "/mnt/beegfs/work/appak/final_pipeline/preprocessed_90k_classifiers/new_splits/arxiv_90k_doi_prediction_train.csv"
VAL_PATH = "/mnt/beegfs/work/appak/final_pipeline/preprocessed_90k_classifiers/new_splits/arxiv_90k_doi_prediction_val.csv"
TEST_PATH = "/mnt/beegfs/work/appak/final_pipeline/preprocessed_90k_classifiers/new_splits/arxiv_90k_doi_prediction_test.csv"

OUTPUT_PATH = "ablation_doi_prediction_results_xgboost.csv"

target_col = "has_doi"


def clean_xgb_feature_names(columns):
    cleaned, seen = [], {}
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


def existing(cols, df):
    return [c for c in cols if c in df.columns]


def startswith_existing(prefixes, df):
    return [c for c in df.columns if any(c.startswith(p) for p in prefixes)]


train_df = pd.read_csv(TRAIN_PATH, low_memory=False, dtype={"arxiv_id": str})
val_df = pd.read_csv(VAL_PATH, low_memory=False, dtype={"arxiv_id": str})
test_df = pd.read_csv(TEST_PATH, low_memory=False, dtype={"arxiv_id": str})

y_train = train_df[target_col].astype(int)
y_val = val_df[target_col].astype(int)
y_test = test_df[target_col].astype(int)

print(f"Train shape: {train_df.shape}")
print(f"Val shape:   {val_df.shape}")
print(f"Test shape:  {test_df.shape}")
print("\nDOI presence prediction")
print(f"Train positive rate: {y_train.mean():.4f}")
print(f"Val positive rate:   {y_val.mean():.4f}")
print(f"Test positive rate:  {y_test.mean():.4f}")


# ============================================================
# Feature groups
# ============================================================

temporal_features = ["year", "paper_age"]

text_structure_features = [
    "n_characters", "n_words", "n_sentences",
    "words_per_sentence", "chars_per_word",
    "title_length", "abstract_length",
]

# For DOI prediction, publication/indexing metadata should be ignored
reference_count = ["reference_count"]

author_impact_features = [
    "first_author_hindex", "max_author_hindex", "mean_author_hindex",
    "num_authors", "last_author_hindex",
    "mean_author_paper_count", "mean_author_citation_count",
]

paper_readability_features = [
    "flesch_kincaid_mean", "flesch_mean", "gunning_fog_mean",
    "coleman_liau_mean", "dale_chall_mean", "ari_mean",
    "linsear_write_mean",
]

abstract_readability_features = [
    "abstract_flesch_kincaid_mean", "abstract_flesch_mean",
    "abstract_gunning_fog_mean", "abstract_coleman_liau_mean",
    "abstract_dale_chall_mean", "abstract_ari_mean",
    "abstract_linsear_write_mean", "abstract_readability_missing",
]

perplexity_features = [
    "abstract_perplexity",
    "abstract_log_perplexity",
]

grammar_features = [
    "grammar_edits",
    "grammar_error_rate",
]

nativeness_features = [
    "native_like_score",
    "confidence",
    "verdict",
]

primary_category_features = startswith_existing(["primary_category"], train_df)

linguistic_features = (
    paper_readability_features
    + abstract_readability_features
    + perplexity_features
    + grammar_features
    + nativeness_features
)

feature_groups = {
    "temporal": temporal_features,
    "text_structure": text_structure_features,
    "reference_count": reference_count,
    "research_field": primary_category_features,
    "author_impact": author_impact_features,
    "linguistic_quality": linguistic_features,

    "paper_readability": paper_readability_features,
    "abstract_readability": abstract_readability_features,
    "perplexity": perplexity_features,
    "grammar": grammar_features,
    "nativeness": nativeness_features,
}

feature_groups = {
    group: existing(features, train_df)
    for group, features in feature_groups.items()
}

for group, features in feature_groups.items():
    print(f"{group}: {len(features)} features")


main_groups = [
    "temporal",
    "text_structure",
    "reference_count",
    "research_field",
    "author_impact",
    "linguistic_quality",
]

full_features = [
    c for c in train_df.columns
    if c not in ["arxiv_id", "corpus_id", "has_doi", "citation_count"]
]


# ============================================================
# Ablation configs
# ============================================================

ablation_configs = {
    "M0_full_model": full_features,

    "M1_remove_temporal_features": [
        f for f in full_features if f not in set(feature_groups["temporal"])
    ],
    "M2_remove_text_length_structural_features": [
        f for f in full_features if f not in set(feature_groups["text_structure"])
    ],
    "M4_remove_reference_count": [
        f for f in full_features if f not in set(feature_groups["reference_count"])
    ],
    "M5_remove_research_field_features": [
        f for f in full_features if f not in set(feature_groups["research_field"])
    ],
    "M6_remove_author_impact_features": [
        f for f in full_features if f not in set(feature_groups["author_impact"])
    ],
    "M7_remove_linguistic_quality_features": [
        f for f in full_features if f not in set(feature_groups["linguistic_quality"])
    ],

    "M8_temporal_features_only": feature_groups["temporal"],
    "M9_text_length_structural_features_only": feature_groups["text_structure"],
    "M11_reference_count_only": feature_groups["reference_count"],
    "M12_research_field_features_only": feature_groups["research_field"],
    "M13_author_impact_features_only": feature_groups["author_impact"],
    "M14_linguistic_quality_features_only": feature_groups["linguistic_quality"],

    "M15_remove_paper_readability": [
        f for f in full_features if f not in set(feature_groups["paper_readability"])
    ],
    "M16_remove_abstract_readability": [
        f for f in full_features if f not in set(feature_groups["abstract_readability"])
    ],
    "M17_remove_perplexity": [
        f for f in full_features if f not in set(feature_groups["perplexity"])
    ],
    "M18_remove_grammatical_accuracy": [
        f for f in full_features if f not in set(feature_groups["grammar"])
    ],
    "M19_remove_native_like_language_use": [
        f for f in full_features if f not in set(feature_groups["nativeness"])
    ],

    "M20_paper_readability_only": feature_groups["paper_readability"],
    "M21_abstract_readability_only": feature_groups["abstract_readability"],
    "M22_perplexity_only": feature_groups["perplexity"],
    "M23_grammatical_accuracy_only": feature_groups["grammar"],
    "M24_native_like_language_use_only": feature_groups["nativeness"],
}


# ============================================================
# Run ablation models
# ============================================================

results = []

for model_name, selected_features in ablation_configs.items():
    print("\n" + "=" * 80)
    print(model_name)
    print("=" * 80)
    print(f"Number of features: {len(selected_features)}")

    X_train = train_df[selected_features].copy()
    X_val = val_df[selected_features].copy()
    X_test = test_df[selected_features].copy()

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

    binary_cols = [
        col for col in X_train.columns
        if X_train[col].dropna().nunique() <= 2
    ]

    numeric_cols_to_scale = [
        col for col in X_train.columns
        if col not in binary_cols
    ]

    scaler = StandardScaler()

    X_train_scaled = X_train.copy()
    X_val_scaled = X_val.copy()
    X_test_scaled = X_test.copy()

    if numeric_cols_to_scale:
        X_train_scaled[numeric_cols_to_scale] = scaler.fit_transform(
            X_train[numeric_cols_to_scale]
        )
        X_val_scaled[numeric_cols_to_scale] = scaler.transform(
            X_val[numeric_cols_to_scale]
        )
        X_test_scaled[numeric_cols_to_scale] = scaler.transform(
            X_test[numeric_cols_to_scale]
        )

    cleaned_feature_names = clean_xgb_feature_names(X_train_scaled.columns)

    X_train_scaled.columns = cleaned_feature_names
    X_val_scaled.columns = cleaned_feature_names
    X_test_scaled.columns = cleaned_feature_names

    n_pos = y_train.sum()
    n_neg = len(y_train) - n_pos
    scale_pos_weight = n_neg / n_pos if n_pos > 0 else 1.0

    model = XGBClassifier(
        objective="binary:logistic",
        eval_metric="auc",
        n_estimators=500,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_lambda=1.0,
        reg_alpha=0.0,
        scale_pos_weight=scale_pos_weight,
        random_state=42,
        n_jobs=1,           # deterministic
        tree_method="hist",
        early_stopping_rounds=50,
    )

    model.fit(
        X_train_scaled,
        y_train,
        eval_set=[(X_val_scaled, y_val)],
        verbose=False,
    )

    y_pred = model.predict(X_test_scaled)
    y_proba = model.predict_proba(X_test_scaled)[:, 1]

    acc = accuracy_score(y_test, y_pred)
    bal_acc = balanced_accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_proba)

    print(f"Accuracy:          {acc:.4f}")
    print(f"Balanced accuracy: {bal_acc:.4f}")
    print(f"ROC-AUC:           {auc:.4f}")

    results.append({
        "model": model_name,
        "task": "DOI presence prediction",
        "n_features": len(selected_features),
        "train_positive_rate": y_train.mean(),
        "val_positive_rate": y_val.mean(),
        "test_positive_rate": y_test.mean(),
        "accuracy": acc,
        "balanced_accuracy": bal_acc,
        "roc_auc": auc,
        "best_iteration": model.best_iteration,
    })


results_df = pd.DataFrame(results)

full_auc = results_df.loc[
    results_df["model"] == "M0_full_model",
    "roc_auc"
].iloc[0]

full_bal_acc = results_df.loc[
    results_df["model"] == "M0_full_model",
    "balanced_accuracy"
].iloc[0]

results_df["delta_roc_auc_vs_full"] = results_df["roc_auc"] - full_auc
results_df["delta_bal_acc_vs_full"] = results_df["balanced_accuracy"] - full_bal_acc

results_df["model_number"] = results_df["model"].str.extract(r"M(\d+)").astype(int)
results_df = results_df.sort_values("model_number").drop(columns="model_number")

print("\n" + "=" * 80)
print("DOI Ablation summary")
print("=" * 80)
print(results_df.to_string(index=False))

results_df.to_csv(OUTPUT_PATH, index=False)
print(f"\nSaved results to: {OUTPUT_PATH}")