#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
EDA and correlation study for the publication success dataset. Computes descriptive
statistics, Mann-Whitney U group difference tests, and Spearman correlations for all
feature groups against citation count and DOI presence. Produces distribution plots,
boxplots by group, and Spearman heatmaps.
Inputs:  data/90k_arxiv_citation_prediction_full.csv, data/90k_arxiv_doi_prediction_full.csv
Output:  results/eda_publication_success_outputs/
"""

import os
import re
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

try:
    from scipy.stats import spearmanr, mannwhitneyu
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False


# ============================================================
# Paths and settings
# ============================================================

CITATION_INPUT_PATH = "data/90k_arxiv_citation_prediction_full.csv"

DOI_INPUT_PATH = "data/90k_arxiv_doi_prediction_full.csv"

OUTPUT_DIR = Path("results/eda_publication_success_outputs")

# 5 was selected as the binary citation threshold based on the model selection experiment in code/06_model_selection/
CITATION_THRESHOLD = 5

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
PLOTS_DIR = OUTPUT_DIR / "plots"
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

CITATION_BIN_PLOTS_DIR = PLOTS_DIR / "citation_bin_boxplots"
CITATION_BIN_PLOTS_DIR.mkdir(parents=True, exist_ok=True)

warnings.filterwarnings("ignore")

# ============================================================
# Feature definitions
# ============================================================

ID_COLS = ["corpus_id", "arxiv_id"]
TARGET_COLS = ["has_doi", "citation_count"]

TEMPORAL_FEATURES = [
    "year", "paper_age"
]

TEXT_STRUCTURE_FEATURES = [
    "n_characters", "n_words", "n_sentences",
    "words_per_sentence", "chars_per_word",
    "title_length", "abstract_length"
]

PUBLICATION_METADATA_FEATURES = [
    "reference_count",
    "is_journal_article", "is_conference", "is_review", "is_book",
    "has_MAG", "is_ACL", "is_PubMedCentral"
]

PAPER_READABILITY_FEATURES = [
    "flesch_kincaid_mean", "flesch_mean", "gunning_fog_mean",
    "coleman_liau_mean", "dale_chall_mean", "ari_mean",
    "linsear_write_mean"
]

ABSTRACT_READABILITY_FEATURES = [
    "abstract_flesch_kincaid_mean", "abstract_flesch_mean",
    "abstract_gunning_fog_mean", "abstract_coleman_liau_mean",
    "abstract_dale_chall_mean", "abstract_ari_mean",
    "abstract_linsear_write_mean", "abstract_readability_missing"
]

ABSTRACT_PERPLEXITY_FEATURES = [
    "abstract_perplexity", "abstract_log_perplexity"
]

GRAMMAR_FEATURES = [
    "grammar_edits", "grammar_error_rate"
]

NATIVENESS_FEATURES = [
    "native_like_score", "confidence", "verdict"
]

AUTHOR_IMPACT_FEATURES = [
    "first_author_hindex", "max_author_hindex", "mean_author_hindex",
    "num_authors", "last_author_hindex",
    "mean_author_paper_count", "mean_author_citation_count"
]



# ============================================================
# Helper functions
# ============================================================

def safe_filename(name: str) -> str:
    name = re.sub(r"[^A-Za-z0-9_]+", "_", name)
    return re.sub(r"_+", "_", name).strip("_")


def existing(cols, df):
    return [c for c in cols if c in df.columns]


def add_group_columns(df):
    if "has_doi" in df.columns:
        df["doi_group"] = np.where(df["has_doi"] == 1, "has_doi = 1", "has_doi = 0")

    if "citation_count" in df.columns:
        df[f"citation_gt_{CITATION_THRESHOLD}"] = (
            df["citation_count"] > CITATION_THRESHOLD
        ).astype(int)

        df["citation_group"] = np.where(
            df[f"citation_gt_{CITATION_THRESHOLD}"] == 1,
            f"citation_count > {CITATION_THRESHOLD}",
            f"citation_count <= {CITATION_THRESHOLD}",
        )

        df["citation_bin"] = pd.cut(
        df["citation_count"],
        bins=[-0.1, 0, 4, 12, 30, np.inf],
        labels=["0", "1-4", "5-12", "13-30", "31+"]
)

    return df


def numeric_summary(df, features, group_col=None):
    rows = []

    if group_col is None:
        for col in features:
            s = pd.to_numeric(df[col], errors="coerce")
            rows.append({
                "feature": col,
                "group": "overall",
                "n": s.notna().sum(),
                "missing": s.isna().sum(),
                "missing_rate": s.isna().mean(),
                "mean": s.mean(),
                "std": s.std(),
                "min": s.min(),
                "p25": s.quantile(0.25),
                "median": s.median(),
                "p75": s.quantile(0.75),
                "max": s.max(),
            })
    else:
        for col in features:
            for group_value, sub in df.groupby(group_col):
                s = pd.to_numeric(sub[col], errors="coerce")
                rows.append({
                    "feature": col,
                    "group": group_value,
                    "n": s.notna().sum(),
                    "missing": s.isna().sum(),
                    "missing_rate": s.isna().mean(),
                    "mean": s.mean(),
                    "std": s.std(),
                    "min": s.min(),
                    "p25": s.quantile(0.25),
                    "median": s.median(),
                    "p75": s.quantile(0.75),
                    "max": s.max(),
                })

    return pd.DataFrame(rows)


def group_difference_tests(df, features, group_binary_col):
    rows = []

    if not SCIPY_AVAILABLE:
        return pd.DataFrame()

    for col in features:
        x0 = pd.to_numeric(df.loc[df[group_binary_col] == 0, col], errors="coerce").dropna()
        x1 = pd.to_numeric(df.loc[df[group_binary_col] == 1, col], errors="coerce").dropna()

        if len(x0) < 5 or len(x1) < 5:
            continue

        try:
            stat, p = mannwhitneyu(x0, x1, alternative="two-sided")
        except Exception:
            stat, p = np.nan, np.nan

        rows.append({
            "feature": col,
            "group_0_n": len(x0),
            "group_1_n": len(x1),
            "group_0_mean": x0.mean(),
            "group_1_mean": x1.mean(),
            "mean_difference_group1_minus_group0": x1.mean() - x0.mean(),
            "group_0_median": x0.median(),
            "group_1_median": x1.median(),
            "median_difference_group1_minus_group0": x1.median() - x0.median(),
            "mannwhitney_u": stat,
            "p_value": p,
        })

    return pd.DataFrame(rows).sort_values("p_value")


def spearman_correlations(df, features, target_col):
    rows = []

    if not SCIPY_AVAILABLE:
        return pd.DataFrame()

    target = pd.to_numeric(df[target_col], errors="coerce")

    for col in features:
        x = pd.to_numeric(df[col], errors="coerce")
        valid = x.notna() & target.notna()

        if valid.sum() < 10:
            continue

        # Spearman rather than Pearson because many features (e.g. citation count, h-index) are heavily skewed
        rho, p = spearmanr(x[valid], target[valid])

        rows.append({
            "feature": col,
            "target": target_col,
            "n": valid.sum(),
            "spearman_rho": rho,
            "p_value": p,
            "abs_spearman_rho": abs(rho),
        })

    return pd.DataFrame(rows).sort_values("abs_spearman_rho", ascending=False)


def plot_hist_by_group(df, feature, group_col, out_dir, bins=50, log_x=False):
    data = df[[feature, group_col]].copy()
    data[feature] = pd.to_numeric(data[feature], errors="coerce")
    data = data.dropna()

    if data.empty:
        return

    if log_x:
        data = data[data[feature] > 0]
        data[feature] = np.log1p(data[feature])

    plt.figure(figsize=(8, 5))

    groups = list(data[group_col].dropna().unique())

    for group in groups:
        values = data.loc[data[group_col] == group, feature]
        plt.hist(
            values,
            bins=bins,
            alpha=0.5,
            density=True,
            label=pretty_name(str(group))
        )

    feature_label = pretty_name(feature)
    group_label = pretty_name(group_col)

    xlabel = f"log1p({feature_label})" if log_x else feature_label

    plt.xlabel(xlabel)
    plt.ylabel("Density")
    plt.title(f"Distribution of {feature_label} by {group_label}")
    plt.legend()
    plt.tight_layout()

    suffix = "_log1p" if log_x else ""
    path = out_dir / f"hist_{safe_filename(feature)}_by_{safe_filename(group_col)}{suffix}.png"
    plt.savefig(path, dpi=300)
    plt.close()


def plot_box_by_group(df, feature, group_col, out_dir, log_y=False):

    data = df[[feature, group_col]].copy()

    data[feature] = pd.to_numeric(data[feature], errors="coerce")

    data = data.dropna()

    if data.empty:
        return

    if log_y:
        data = data[data[feature] > 0]
        data[feature] = np.log1p(data[feature])

    # ========================================================
    # Force ordered citation bins
    # ========================================================

    if group_col == "citation_bin":

        ordered_groups = ["0", "1-4", "5-12", "13-30", "31+"]

        data[group_col] = data[group_col].astype(str)

        groups = [
            g for g in ordered_groups
            if g in data[group_col].unique()
        ]

    else:
        if group_col == "citation_group":
            ordered_groups = [
                f"citation_count <= {CITATION_THRESHOLD}",
                f"citation_count > {CITATION_THRESHOLD}",
            ]

            groups = [
                g for g in ordered_groups
                if g in data[group_col].unique()
            ]
        else:
            groups = list(data[group_col].dropna().unique())

    # ========================================================
    # Collect values
    # ========================================================

    values = [
        data.loc[data[group_col] == g, feature].values
        for g in groups
    ]

    if len(values) == 0:
        return

    # ========================================================
    # Pretty labels
    # ========================================================

    pretty_groups = [pretty_name(str(g)) for g in groups]

    feature_label = pretty_name(feature)
    group_label = pretty_name(group_col)

    # ========================================================
    # Plot
    # ========================================================

    ylabel = f"log1p({feature_label})" if log_y else feature_label

    plt.figure(figsize=(6, 5))

    box = plt.boxplot(
        values,
        labels=pretty_groups,
        patch_artist=True,
        showfliers=False,
        widths=0.4,  # narrower boxes
        boxprops=dict(
            facecolor="tab:blue",   # blue boxes
            edgecolor="black",
            linewidth=1.1
        ),
        medianprops=dict(
            color="red",           # red median line
            linewidth=1.1
        ),
        whiskerprops=dict(
            color="black",
            linestyle="--",
            linewidth=1.0
        ),
        capprops=dict(
            color="black",
            linewidth=1.0
        )
    )

    plt.ylabel(ylabel)

    # x-axis label
    if group_col == "citation_bin":
        plt.xlabel("Citation count bins")
    else:
        plt.xlabel(group_label)

    plt.title(
        feature_label,
        fontweight="bold"
    )

    plt.xticks(rotation=0)

    plt.tight_layout()

    suffix = "_log1p" if log_y else ""

    path = (
        out_dir /
        f"box_{safe_filename(feature)}_by_{safe_filename(group_col)}{suffix}.png"
    )

    plt.savefig(path, dpi=300)

    plt.close()


def plot_correlation_bar(corr_df, title, out_path, top_n=30):
    if corr_df.empty:
        return

    top = corr_df.head(top_n).copy()
    top["feature"] = top["feature"].map(pretty_name)
    top = top.sort_values("spearman_rho")

    plt.figure(figsize=(8, max(5, 0.28 * len(top))))
    plt.barh(top["feature"], top["spearman_rho"])
    plt.xlabel("Spearman correlation")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()


def plot_correlation_heatmap(df, features, out_path, title):
    selected = existing(features, df)
    if len(selected) < 2:
        return

    # Spearman correlation is rank-based and robust to the non-normal distributions of linguistic scores
    corr = df[selected].apply(pd.to_numeric, errors="coerce").corr(method="spearman")

    plt.figure(figsize=(max(8, len(selected) * 0.45), max(6, len(selected) * 0.4)))
    plt.imshow(corr, aspect="auto")
    plt.colorbar(label="Spearman correlation")
    pretty_selected = [pretty_name(x) for x in selected]

    plt.xticks(range(len(selected)), pretty_selected, rotation=90)
    plt.yticks(range(len(selected)), pretty_selected)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()

    corr.to_csv(out_path.with_suffix(".csv"))


def plot_count_distribution(df):
    plt.figure(figsize=(8, 5))
    citation = pd.to_numeric(df["citation_count"], errors="coerce").dropna()
    citation = citation[citation >= 0]
    plt.hist(np.log1p(citation), bins=60)
    plt.xlabel("log1p(Citation count)")
    plt.ylabel("Number of papers")
    plt.title("Distribution of citation count")
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "citation_count_log_distribution.png", dpi=300)
    plt.close()

    if "has_doi" in df.columns:
        counts = df["has_doi"].value_counts().sort_index()
        plt.figure(figsize=(6, 4))
        plt.bar(["DOI absent", "DOI present"], [counts.get(0, 0), counts.get(1, 0)])
        plt.ylabel("Number of papers")
        plt.title("DOI presence distribution")
        plt.tight_layout()
        plt.savefig(PLOTS_DIR / "has_doi_distribution.png", dpi=300)
        plt.close()

    if f"citation_gt_{CITATION_THRESHOLD}" in df.columns:
        counts = df[f"citation_gt_{CITATION_THRESHOLD}"].value_counts().sort_index()
        plt.figure(figsize=(6, 4))
        plt.bar(
            [f"Citation count ≤ {CITATION_THRESHOLD}", f"Citation count > {CITATION_THRESHOLD}"],
            [counts.get(0, 0), counts.get(1, 0)]
        )
        plt.ylabel("Number of papers")
        plt.title(f"Citation threshold distribution: Citation count > {CITATION_THRESHOLD}")
        plt.xticks(rotation=15, ha="right")
        plt.tight_layout()
        plt.savefig(PLOTS_DIR / f"citation_gt_{CITATION_THRESHOLD}_distribution.png", dpi=300)
        plt.close()


# ============================================================
# Human-readable feature labels
# ============================================================

DISPLAY_NAMES = {

    # Targets / groups
    "has_doi": "DOI present",
    "has_doi = 1": "DOI present",
    "has_doi = 0": "DOI absent",

    "citation_count": "Citation count",
    f"citation_count > {CITATION_THRESHOLD}": f"Citation count > {CITATION_THRESHOLD}",
    f"citation_count <= {CITATION_THRESHOLD}": f"Citation count ≤ {CITATION_THRESHOLD}",

    "doi_group": "DOI group",
    "citation_group": "citation group",

    "citation_bin": "citation bin",

    # Temporal
    "year": "Publication year",
    "paper_age": "Paper age",

    # Text structure
    "n_characters": "Number of characters",
    "n_words": "Number of words",
    "n_sentences": "Number of sentences",
    "words_per_sentence": "Words per sentence",
    "chars_per_word": "Characters per word",
    "title_length": "Title length",
    "abstract_length": "Abstract length",

    # Publication metadata
    "reference_count": "Reference count",
    "is_journal_article": "Journal article",
    "is_conference": "Conference paper",
    "is_review": "Review paper",
    "is_book": "Book",
    "has_MAG": "Indexed in MAG",
    "is_ACL": "ACL paper",
    "is_PubMedCentral": "Indexed in PubMed Central",

    # Paper readability
    "flesch_kincaid_mean": "Flesch-Kincaid readability score",
    "flesch_mean": "Flesch readability score",
    "gunning_fog_mean": "Gunning Fog readability score",
    "coleman_liau_mean": "Coleman-Liau readability score",
    "dale_chall_mean": "Dale-Chall readability score",
    "ari_mean": "Automated Readability Index",
    "linsear_write_mean": "Linsear Write readability score",

    # Abstract readability
    "abstract_flesch_kincaid_mean": "Abstract Flesch-Kincaid readability score",
    "abstract_flesch_mean": "Abstract Flesch readability score",
    "abstract_gunning_fog_mean": "Abstract Gunning Fog readability score",
    "abstract_coleman_liau_mean": "Abstract Coleman-Liau readability score",
    "abstract_dale_chall_mean": "Abstract Dale-Chall readability score",
    "abstract_ari_mean": "Abstract Automated Readability Index",
    "abstract_linsear_write_mean": "Abstract Linsear Write readability score",
    "abstract_readability_missing": "Missing abstract readability values",

    # Perplexity
    "abstract_perplexity": "Perplexity",
    "abstract_log_perplexity": "Log perplexity",

    # Grammar
    "grammar_edits": "Grammar correction edits",
    "grammar_error_rate": "Grammar error rate",

    # Nativeness
    "native_like_score": "Native-like score",
    "confidence": "LLM confidence score",
    "verdict": "LLM nativeness verdict",

    # Author impact
    "first_author_hindex": "First author h-index",
    "max_author_hindex": "Maximum author h-index",
    "mean_author_hindex": "Mean author h-index",
    "last_author_hindex": "Last author h-index",
    "num_authors": "Number of authors",
    "mean_author_paper_count": "Mean author paper count",
    "mean_author_citation_count": "Mean author citation count",
}


def pretty_name(name):
    return DISPLAY_NAMES.get(name, name.replace("_", " ").title())

# ============================================================
# Load datasets
# ============================================================

print(f"Loading citation dataset: {CITATION_INPUT_PATH}")
citation_df = pd.read_csv(
    CITATION_INPUT_PATH,
    low_memory=False,
    dtype={"arxiv_id": str, "corpus_id": str},
)

print(f"Loading DOI dataset: {DOI_INPUT_PATH}")
doi_df = pd.read_csv(
    DOI_INPUT_PATH,
    low_memory=False,
    dtype={"arxiv_id": str, "corpus_id": str},
)

print(f"Citation dataset shape: {citation_df.shape}")
print(f"DOI dataset shape:      {doi_df.shape}")


# ============================================================
# Merge DOI labels
# ============================================================

doi_cols = ["arxiv_id", "has_doi"]

doi_df = doi_df[doi_cols].drop_duplicates(subset=["arxiv_id"])

df = citation_df.merge(
    doi_df,
    on="arxiv_id",
    how="left",
)

print(f"Merged dataset shape: {df.shape}")

missing_doi = df["has_doi"].isna().sum()
print(f"Rows with missing DOI labels after merge: {missing_doi}")

df = add_group_columns(df)

# ensure numeric targets
df["has_doi"] = pd.to_numeric(df["has_doi"], errors="coerce")
df["citation_count"] = pd.to_numeric(df["citation_count"], errors="coerce")


# ============================================================
# Feature groups
# ============================================================

primary_category_features = [c for c in df.columns if c.startswith("primary_category_")]
publication_source_features = [c for c in df.columns if c.startswith("publication_source_")]

# For citation prediction / citation-related EDA:
# publication type, indexing, and source features are allowed.
CITATION_PUBLICATION_METADATA_FEATURES = (
    existing(PUBLICATION_METADATA_FEATURES, df)
    + publication_source_features
)

# For DOI prediction / DOI-related EDA:
# publication type, venue, and indexing flags are determined after DOI assignment, so they are excluded to avoid leakage.
DOI_PUBLICATION_METADATA_FEATURES = existing(["reference_count"], df)

COMMON_FEATURE_GROUPS = {
    "temporal": existing(TEMPORAL_FEATURES, df),
    "text_structure": existing(TEXT_STRUCTURE_FEATURES, df),
    "field_category": primary_category_features,
    "author_impact": existing(AUTHOR_IMPACT_FEATURES, df),
    "paper_readability": existing(PAPER_READABILITY_FEATURES, df),
    "abstract_readability": existing(ABSTRACT_READABILITY_FEATURES, df),
    "abstract_perplexity": existing(ABSTRACT_PERPLEXITY_FEATURES, df),
    "grammar": existing(GRAMMAR_FEATURES, df),
    "nativeness": existing(NATIVENESS_FEATURES, df),
}

LINGUISTIC_FEATURES = (
    COMMON_FEATURE_GROUPS["paper_readability"]
    + COMMON_FEATURE_GROUPS["abstract_readability"]
    + COMMON_FEATURE_GROUPS["abstract_perplexity"]
    + COMMON_FEATURE_GROUPS["grammar"]
    + COMMON_FEATURE_GROUPS["nativeness"]
)

DOI_FEATURES = sorted(set(
    COMMON_FEATURE_GROUPS["temporal"]
    + COMMON_FEATURE_GROUPS["text_structure"]
    + DOI_PUBLICATION_METADATA_FEATURES
    + COMMON_FEATURE_GROUPS["field_category"]
    + COMMON_FEATURE_GROUPS["author_impact"]
    + LINGUISTIC_FEATURES
))

CITATION_FEATURES = sorted(set(
    COMMON_FEATURE_GROUPS["temporal"]
    + COMMON_FEATURE_GROUPS["text_structure"]
    + CITATION_PUBLICATION_METADATA_FEATURES
    + COMMON_FEATURE_GROUPS["field_category"]
    + COMMON_FEATURE_GROUPS["author_impact"]
    + LINGUISTIC_FEATURES
))

FEATURE_GROUPS = COMMON_FEATURE_GROUPS.copy()

FEATURE_GROUPS["publication_metadata_citation"] = CITATION_PUBLICATION_METADATA_FEATURES
FEATURE_GROUPS["publication_metadata_doi"] = DOI_PUBLICATION_METADATA_FEATURES

print("\nFeature set sizes:")
print(f"DOI feature set:      {len(DOI_FEATURES)}")
print(f"Citation feature set: {len(CITATION_FEATURES)}")


# ============================================================
# Basic dataset summaries
# ============================================================

dataset_overview = pd.DataFrame({
    "statistic": [
        "n_rows",
        "n_columns",
        "has_doi_positive_rate",
        f"citation_count_gt_{CITATION_THRESHOLD}_rate",
        "citation_count_mean",
        "citation_count_median",
        "citation_count_max",
    ],
    "value": [
        len(df),
        df.shape[1],
        df["has_doi"].mean() if "has_doi" in df.columns else np.nan,
        df[f"citation_gt_{CITATION_THRESHOLD}"].mean() if f"citation_gt_{CITATION_THRESHOLD}" in df.columns else np.nan,
        df["citation_count"].mean() if "citation_count" in df.columns else np.nan,
        df["citation_count"].median() if "citation_count" in df.columns else np.nan,
        df["citation_count"].max() if "citation_count" in df.columns else np.nan,
    ]
})

dataset_overview.to_csv(OUTPUT_DIR / "dataset_overview.csv", index=False)

missing_summary = pd.DataFrame({
    "feature": df.columns,
    "missing_count": df.isna().sum().values,
    "missing_rate": df.isna().mean().values,
}).sort_values("missing_rate", ascending=False)

missing_summary.to_csv(OUTPUT_DIR / "missing_value_summary.csv", index=False)


# ============================================================
# Descriptive statistics
# ============================================================

overall_summary_doi = numeric_summary(df, DOI_FEATURES)
overall_summary_doi.to_csv(OUTPUT_DIR / "numeric_feature_summary_overall_doi_features.csv", index=False)

overall_summary_citation = numeric_summary(df, CITATION_FEATURES)
overall_summary_citation.to_csv(OUTPUT_DIR / "numeric_feature_summary_overall_citation_features.csv", index=False)

doi_summary = numeric_summary(df, DOI_FEATURES, group_col="doi_group")
doi_summary.to_csv(OUTPUT_DIR / "numeric_feature_summary_by_doi.csv", index=False)

citation_group_summary = numeric_summary(df, CITATION_FEATURES, group_col="citation_group")
citation_group_summary.to_csv(
    OUTPUT_DIR / f"numeric_feature_summary_by_citation_gt_{CITATION_THRESHOLD}.csv",
    index=False
)

# ============================================================
# Group difference tests
# ============================================================

if "has_doi" in df.columns and SCIPY_AVAILABLE:
    doi_tests = group_difference_tests(
        df.dropna(subset=["has_doi"]),
        DOI_FEATURES,
        "has_doi"
    )
    doi_tests.to_csv(OUTPUT_DIR / "group_differences_has_doi_mannwhitney.csv", index=False)

if f"citation_gt_{CITATION_THRESHOLD}" in df.columns and SCIPY_AVAILABLE:
    citation_tests = group_difference_tests(
        df,
        CITATION_FEATURES,
        f"citation_gt_{CITATION_THRESHOLD}"
    )
    citation_tests.to_csv(
        OUTPUT_DIR / f"group_differences_citation_gt_{CITATION_THRESHOLD}_mannwhitney.csv",
        index=False
    )

# ============================================================
# Correlation analyses
# ============================================================

if SCIPY_AVAILABLE:
    corr_citation = spearman_correlations(df, CITATION_FEATURES, "citation_count")
    corr_citation.to_csv(OUTPUT_DIR / "spearman_correlations_with_citation_count.csv", index=False)

    corr_has_doi = spearman_correlations(
        df.dropna(subset=["has_doi"]),
        DOI_FEATURES,
        "has_doi"
    )
    corr_has_doi.to_csv(OUTPUT_DIR / "spearman_correlations_with_has_doi.csv", index=False)

    corr_citation_binary = spearman_correlations(
        df,
        CITATION_FEATURES,
        f"citation_gt_{CITATION_THRESHOLD}"
    )
    corr_citation_binary.to_csv(
        OUTPUT_DIR / f"spearman_correlations_with_citation_gt_{CITATION_THRESHOLD}.csv",
        index=False
    )
    plot_correlation_bar(
        corr_citation,
        "Top Spearman correlations with citation count",
        PLOTS_DIR / "top_correlations_with_citation_count.png",
    )

    plot_correlation_bar(
        corr_has_doi,
        "Top Spearman correlations with DOI presence",
        PLOTS_DIR / "top_correlations_with_has_doi.png",
    )

    plot_correlation_bar(
        corr_citation_binary,
        f"Top Spearman correlations with citation_count > {CITATION_THRESHOLD}",
        PLOTS_DIR / f"top_correlations_with_citation_gt_{CITATION_THRESHOLD}.png",
    )

# ============================================================
# Correlation heatmaps for selected feature groups
# ============================================================

plot_correlation_heatmap(
    df,
    LINGUISTIC_FEATURES,
    PLOTS_DIR / "spearman_heatmap_linguistic_features.png",
    "Spearman correlation among linguistic quality features",
)

plot_correlation_heatmap(
    df,
    FEATURE_GROUPS["paper_readability"] + FEATURE_GROUPS["abstract_readability"],
    PLOTS_DIR / "spearman_heatmap_readability_features.png",
    "Spearman correlation among readability features",
)

plot_correlation_heatmap(
    df,
    FEATURE_GROUPS["author_impact"],
    PLOTS_DIR / "spearman_heatmap_author_impact_features.png",
    "Spearman correlation among author impact features",
)


# ============================================================
# Distribution plots
# ============================================================

plot_count_distribution(df)

IMPORTANT_PLOT_FEATURES = existing([
    "year",
    "paper_age",
    "n_words",
    "n_sentences",
    "words_per_sentence",
    "reference_count",
    "flesch_kincaid_mean",
    "flesch_mean",
    "gunning_fog_mean",
    "coleman_liau_mean",
    "dale_chall_mean",
    "ari_mean",
    "linsear_write_mean",
    "abstract_perplexity",
    "abstract_log_perplexity",
    "grammar_error_rate",
    "grammar_edits",
    "native_like_score",
    "verdict",
    "confidence",
    "first_author_hindex",
    "max_author_hindex",
    "mean_author_hindex",
    "num_authors",
    "mean_author_citation_count",
], df)

LOG_SCALE_FEATURES = {
    "n_words",
    "n_sentences",
    "reference_count",
    "grammar_edits",
    "first_author_hindex",
    "max_author_hindex",
    "mean_author_hindex",
    "mean_author_citation_count",
}

for feature in IMPORTANT_PLOT_FEATURES:
    use_log = feature in LOG_SCALE_FEATURES

    if "doi_group" in df.columns:
        plot_hist_by_group(df, feature, "doi_group", PLOTS_DIR, log_x=use_log)
        plot_box_by_group(df, feature, "doi_group", PLOTS_DIR, log_y=use_log)

    if "citation_group" in df.columns:
        plot_hist_by_group(df, feature, "citation_group", PLOTS_DIR, log_x=use_log)
        plot_box_by_group(df, feature, "citation_group", PLOTS_DIR, log_y=use_log)

    if "citation_bin" in df.columns:
        plot_box_by_group(
            df,
            feature,
            "citation_bin",
            CITATION_BIN_PLOTS_DIR,
            log_y=use_log
        )

# ============================================================
# Categorical / one-hot summaries
# ============================================================

def onehot_group_summary(df, onehot_cols, target_group_col, output_name):
    rows = []

    for col in onehot_cols:
        for group_value, sub in df.groupby(target_group_col):
            values = pd.to_numeric(sub[col], errors="coerce")
            rows.append({
                "feature": col,
                "group": group_value,
                "count": int(values.sum(skipna=True)),
                "share_within_group": values.mean(skipna=True),
                "group_n": len(sub),
            })

    out = pd.DataFrame(rows)
    out.to_csv(OUTPUT_DIR / output_name, index=False)
    return out


if primary_category_features:
    onehot_group_summary(
        df,
        primary_category_features,
        "doi_group",
        "primary_category_distribution_by_doi.csv",
    )
    onehot_group_summary(
        df,
        primary_category_features,
        "citation_group",
        f"primary_category_distribution_by_citation_gt_{CITATION_THRESHOLD}.csv",
    )

if publication_source_features:
    onehot_group_summary(
        df,
        publication_source_features,
        "citation_group",
        f"publication_source_distribution_by_citation_gt_{CITATION_THRESHOLD}.csv",
    )

# ============================================================
# Feature-group-level summaries
# ============================================================

feature_group_overview = []

for group, cols in FEATURE_GROUPS.items():
    for col in cols:
        feature_group_overview.append({
            "feature_group": group,
            "feature": col,
            "missing_rate": df[col].isna().mean() if col in df.columns else np.nan,
            "mean": pd.to_numeric(df[col], errors="coerce").mean() if col in df.columns else np.nan,
            "std": pd.to_numeric(df[col], errors="coerce").std() if col in df.columns else np.nan,
        })

feature_group_overview = pd.DataFrame(feature_group_overview)
feature_group_overview.to_csv(OUTPUT_DIR / "feature_group_overview.csv", index=False)


# ============================================================
# Done
# ============================================================

print("\nEDA completed.")
print(f"Outputs saved to: {OUTPUT_DIR.resolve()}")
print(f"Plots saved to:   {PLOTS_DIR.resolve()}")