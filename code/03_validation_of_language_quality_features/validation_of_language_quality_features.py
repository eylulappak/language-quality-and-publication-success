#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import re
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import mannwhitneyu


# ============================================================
# Input / output
# ============================================================

INPUT_CSV = (
    "data\\200-paper-benchmark_validation_of_linguistic_features.csv"
)

OUTPUT_DIR = Path("data\\200-paper-benchmark\\linguistic_feature_boxplots")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

STATS_OUTPUT_CSV = OUTPUT_DIR / "mannwhitney_rank_biserial_results.csv"


# ============================================================
# Features and target
# ============================================================

FEATURE_COLUMNS = [
    "perplexity",
    "flesch_kincaid_mean",
    "flesch_mean",
    "gunning_fog_mean",
    "coleman_liau_mean",
    "dale_chall_mean",
    "ari_mean",
    "linsear_write_mean",
    "errors_per_sentence",
]

TARGET_COLUMN = "ground_truth"


# ============================================================
# Helper functions
# ============================================================

DISPLAY_NAMES = {
    "perplexity": "Perplexity",
    "flesch_kincaid_mean": "Flesch-Kincaid Readability Score",
    "flesch_mean": "Flesch Readability Score",
    "gunning_fog_mean": "Gunning Fog Readability Score",
    "coleman_liau_mean": "Coleman-Liau Readability Score",
    "dale_chall_mean": "Dale-Chall Readability Score",
    "ari_mean": "Automated Readability Index",
    "linsear_write_mean": "Linsear Write Readability Score",
    "errors_per_sentence": "Grammar Error Rate",
}


def pretty_name(col):
    return DISPLAY_NAMES.get(col, col.replace("_", " ").title())


def safe_filename(name):
    name = re.sub(r"[^A-Za-z0-9_]+", "_", name)
    return re.sub(r"_+", "_", name).strip("_")


# ============================================================
# Load data
# ============================================================

df = pd.read_csv(INPUT_CSV)
print("Dataset shape:", df.shape)

df = df[FEATURE_COLUMNS + [TARGET_COLUMN]].copy()


# ============================================================
# Clean / label target groups
# ============================================================

df[TARGET_COLUMN] = df[TARGET_COLUMN].astype(str).str.strip()

label_map = {
    "0": "Native-Like",
    "1": "Non-Native-Like",
    "native-like": "Native-Like",
    "non-native-like": "Non-Native-Like",
    "Native-Like": "Native-Like",
    "Non-Native-Like": "Non-Native-Like",
}

df["group_label"] = df[TARGET_COLUMN].map(label_map).fillna(df[TARGET_COLUMN])

group_order = ["Native-Like", "Non-Native-Like"]


# ============================================================
# Mann-Whitney U tests and rank-biserial correlations
# ============================================================

stats_results = []

for feature in FEATURE_COLUMNS:
    data = df[[feature, "group_label"]].copy()
    data[feature] = pd.to_numeric(data[feature], errors="coerce")
    data = data.dropna(subset=[feature, "group_label"])

    groups = [g for g in group_order if g in data["group_label"].unique()]

    if len(groups) < 2:
        print(f"Skipping statistical test for {feature}: fewer than two groups found.")
        continue

    group_1, group_2 = groups[0], groups[1]

    x = data.loc[data["group_label"] == group_1, feature]
    y = data.loc[data["group_label"] == group_2, feature]

    u_stat, p_value = mannwhitneyu(x, y, alternative="two-sided")

    n1 = len(x)
    n2 = len(y)

    rank_biserial = (2 * u_stat) / (n1 * n2) - 1

    stats_results.append({
        "feature": feature,
        "feature_label": pretty_name(feature),
        "group_1": group_1,
        "group_2": group_2,
        "n_group_1": n1,
        "n_group_2": n2,
        "median_group_1": x.median(),
        "median_group_2": y.median(),
        "u_statistic": u_stat,
        "p_value": p_value,
        "rank_biserial_correlation": rank_biserial,
        "abs_rank_biserial_correlation": abs(rank_biserial),
    })

stats_df = pd.DataFrame(stats_results)
stats_df.to_csv(STATS_OUTPUT_CSV, index=False)

print(f"Saved statistical results: {STATS_OUTPUT_CSV}")


# ============================================================
# Generate one boxplot per feature
# ============================================================

for feature in FEATURE_COLUMNS:
    data = df[[feature, "group_label"]].copy()
    data[feature] = pd.to_numeric(data[feature], errors="coerce")
    data = data.dropna(subset=[feature, "group_label"])

    groups = [g for g in group_order if g in data["group_label"].unique()]

    if len(groups) < 2:
        print(f"Skipping {feature}: fewer than two groups found.")
        continue

    values = [
        data.loc[data["group_label"] == group, feature].values
        for group in groups
    ]

    feature_label = pretty_name(feature)

    plt.figure(figsize=(10, 6))

    plt.boxplot(
        values,
        labels=groups,
        patch_artist=True,
        showfliers=False,
        widths=0.18,
        boxprops=dict(
            facecolor="tab:blue",
            edgecolor="black",
            linewidth=1.2,
        ),
        medianprops=dict(
            color="red",
            linewidth=2.4,
        ),
        whiskerprops=dict(
            color="black",
            linewidth=1.2,
        ),
        capprops=dict(
            color="black",
            linewidth=1.2,
        ),
    )

    plt.title(
        f"{feature_label} in Native-Like and Non-Native-Like Papers",
        fontsize=18,
    )
    plt.ylabel(feature_label, fontsize=14)
    plt.xticks(fontsize=13)
    plt.yticks(fontsize=12)
    plt.grid(axis="y", alpha=0.5)

    plt.tight_layout()

    output_path = OUTPUT_DIR / f"{safe_filename(feature)}_boxplot_by_ground_truth.png"
    plt.savefig(output_path, dpi=300)
    plt.close()

    print(f"Saved: {output_path}")

print("Done.")