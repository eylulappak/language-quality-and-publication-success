import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# ============================================================
# Create output directory
# ============================================================

OUTPUT_DIR = Path("results/feature_importance_plots")

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print(f"\nSaving plots to:\n{OUTPUT_DIR}")

# ============================================================
# 1. Paste your top-30 feature lists here
# ============================================================

xgboost_gain = [
    "primary_category_Physics",
    "primary_category_Computer_Science",
    "reference_count",
    "primary_category_Mathematics",
    "num_authors",
    "verdict",
    "native_like_score",
    "primary_category_Chemistry",
    "primary_category_Materials_Science",
    "max_author_hindex",
    "mean_author_hindex",
    "primary_category_Engineering",
    "year",
    "n_words",
    "primary_category_Medicine",
    "n_characters",
    "n_sentences",
    "paper_age",
    "coleman_liau_mean",
    "abstract_coleman_liau_mean",
    "confidence",
    "mean_author_citation_count",
    "abstract_length",
    "abstract_dale_chall_mean",
    "primary_category_Economics",
    "words_per_sentence",
    "flesch_kincaid_mean",
    "first_author_hindex",
    "dale_chall_mean",
    "flesch_mean",
]

xgboost_split = [
    "reference_count",
    "abstract_length",
    "abstract_perplexity",
    "title_length",
    "grammar_error_rate",
    "dale_chall_mean",
    "words_per_sentence",
    "mean_author_paper_count",
    "coleman_liau_mean",
    "chars_per_word",
    "grammar_edits",
    "mean_author_citation_count",
    "n_sentences",
    "n_characters",
    "abstract_dale_chall_mean",
    "year",
    "flesch_mean",
    "linsear_write_mean",
    "last_author_hindex",
    "mean_author_hindex",
    "first_author_hindex",
    "ari_mean",
    "abstract_coleman_liau_mean",
    "n_words",
    "gunning_fog_mean",
    "max_author_hindex",
    "num_authors",
    "abstract_linsear_write_mean",
    "flesch_kincaid_mean",
    "abstract_gunning_fog_mean",
]

lgbm_gain = [
    "reference_count",
    "primary_category_Physics",
    "num_authors",
    "year",
    "native_like_score",
    "mean_author_hindex",
    "abstract_length",
    "coleman_liau_mean",
    "max_author_hindex",
    "n_sentences",
    "primary_category_Computer_Science",
    "n_characters",
    "n_words",
    "title_length",
    "words_per_sentence",
    "abstract_coleman_liau_mean",
    "dale_chall_mean",
    "abstract_dale_chall_mean",
    "mean_author_citation_count",
    "chars_per_word",
    "mean_author_paper_count",
    "grammar_error_rate",
    "grammar_edits",
    "abstract_perplexity",
    "first_author_hindex",
    "last_author_hindex",
    "flesch_mean",
    "flesch_kincaid_mean",
    "linsear_write_mean",
    "abstract_gunning_fog_mean",
]

lgbm_split = [
    "reference_count",
    "abstract_length",
    "title_length",
    "dale_chall_mean",
    "coleman_liau_mean",
    "words_per_sentence",
    "grammar_error_rate",
    "abstract_dale_chall_mean",
    "abstract_perplexity",
    "mean_author_paper_count",
    "chars_per_word",
    "grammar_edits",
    "year",
    "n_sentences",
    "mean_author_citation_count",
    "n_characters",
    "mean_author_hindex",
    "flesch_mean",
    "abstract_coleman_liau_mean",
    "last_author_hindex",
    "n_words",
    "num_authors",
    "linsear_write_mean",
    "max_author_hindex",
    "first_author_hindex",
    "abstract_gunning_fog_mean",
    "gunning_fog_mean",
    "flesch_kincaid_mean",
    "ari_mean",
    "abstract_linsear_write_mean",
]

logreg = [
    "reference_count",
    "n_characters",
    "mean_author_citation_count",
    "native_like_score",
    "year",
    "primary_category_Physics",
    "abstract_dale_chall_mean",
    "max_author_hindex",
    "grammar_edits",
    "mean_author_hindex",
    "grammar_error_rate",
    "title_length",
    "primary_category_Materials_Science",
    "first_author_hindex",
    "mean_author_paper_count",
    "primary_category_Chemistry",
    "words_per_sentence",
    "ari_mean",
    "abstract_readability_missing",
    "abstract_length",
    "flesch_mean",
    "flesch_kincaid_mean",
    "primary_category_Engineering",
    "abstract_gunning_fog_mean",
    "coleman_liau_mean",
    "dale_chall_mean",
    "abstract_coleman_liau_mean",
    "chars_per_word",
    "primary_category_Medicine",
    "abstract_linsear_write_mean",
]

sources = {
    "XGBoost split": xgboost_split,
    "XGBoost gain": xgboost_gain,
    "LGBM split": lgbm_split,
    "LGBM gain": lgbm_gain,
    "Logistic Regression": logreg
}


# ============================================================
# 2. Feature group mapping
# ============================================================

def assign_main_group(feature):
    if feature in ["year", "paper_age"]:
        return "A. Temporal"

    if feature in [
        "n_characters", "n_words", "n_sentences", "words_per_sentence",
        "chars_per_word", "title_length", "abstract_length"
    ]:
        return "B. Text length / structural"

    if feature in [
        "reference_count"
    ]:
        return "D. Reference count"

    if feature.startswith("primary_category_"):
        return "E. Research field"

    if feature in [
        "first_author_hindex", "max_author_hindex", "mean_author_hindex",
        "last_author_hindex", "mean_author_paper_count",
        "mean_author_citation_count", "num_authors"
    ]:
        return "F. Author impact"

    linguistic_keywords = [
        "flesch", "gunning", "coleman", "dale", "ari", "linsear",
        "perplexity", "grammar", "native", "verdict", "readability"
    ]

    if any(k in feature for k in linguistic_keywords):
        return "G. Linguistic quality"

    return "Other"


def assign_linguistic_group(feature):
    if feature in ["grammar_edits", "grammar_error_rate"]:
        return "Grammatical accuracy"

    if feature in ["native_like_score", "confidence", "verdict"]:
        return "Native-like language use"

    if "abstract_perplexity" in feature:
        return "Perplexity"

    if feature.startswith("abstract_") and (
        "flesch" in feature or "gunning" in feature or "coleman" in feature
        or "dale" in feature or "ari" in feature or "linsear" in feature
        or "readability" in feature
    ):
        return "Abstract readability"

    if (
        "flesch" in feature or "gunning" in feature or "coleman" in feature
        or "dale" in feature or "ari" in feature or "linsear" in feature
    ):
        return "Paper readability"

    return None


# ============================================================
# 3. Convert rankings into weighted scores
# ============================================================

rows = []

for source, features in sources.items():
    n = len(features)

    for rank, feature in enumerate(features, start=1):
        # Rank-based score: top-ranked feature gets 30, last gets 1
        rank_score = n - rank + 1

        rows.append({
            "source": source,
            "feature": feature,
            "rank": rank,
            "rank_score": rank_score,
            "main_group": assign_main_group(feature),
            "linguistic_group": assign_linguistic_group(feature)
        })

df = pd.DataFrame(rows)

# Optional: inspect frequency across models
feature_summary = (
    df.groupby("feature")
      .agg(
          frequency=("source", "nunique"),
          total_rank_score=("rank_score", "sum"),
          groups=("main_group", "first")
      )
      .sort_values(["frequency", "total_rank_score"], ascending=False)
)

print("\nMost consistently important individual features:")
print(feature_summary.head(30))

# ============================================================
# MAX-RANK FEATURE GROUP IMPORTANCE FOR DOI
# Formula: group score = max(rank_score within group)
# Combined score = average group score across the 5 models
# ============================================================

main_group_order = [
    "A. Temporal",
    "B. Text length / structural",
    "D. Reference count",
    "E. Research field",
    "F. Author impact",
    "G. Linguistic quality"
]

linguistic_order = [
    "Paper readability",
    "Abstract readability",
    "Perplexity",
    "Grammatical accuracy",
    "Native-like language use"
]


# ============================================================
# 4. Main feature group plot: max-score method
# ============================================================

main_max_scores = (
    df[df["main_group"].isin(main_group_order)]
    .groupby(["main_group", "source"])["rank_score"]
    .max()
    .reset_index()
)

main_pivot = (
    main_max_scores
    .pivot(index="main_group", columns="source", values="rank_score")
    .reindex(main_group_order)
    .fillna(0)
)

fig, ax = plt.subplots(figsize=(12, 6))

main_pivot.plot(kind="bar", ax=ax, width=0.8)

plt.ylabel("Maximum rank score within group")
plt.xlabel("")
plt.title(
    "Rank of the Top Feature per Group Across Models (DOI Presence Prediction)"
)
plt.xticks(rotation=30, ha="right")
plt.legend(title="Importance source")
plt.tight_layout()

main_plot_path = OUTPUT_DIR / "doi_main_feature_groups_importance_MAX_SCORE.png"

plt.savefig(main_plot_path, dpi=300, bbox_inches="tight")
plt.close()

print(f"\nSaved DOI main max-score plot:\n{main_plot_path}")


# ============================================================
# 5. Linguistic subgroup plot: max-score method
# ============================================================

ling_df = df[(df["linguistic_group"].notna())].copy()

ling_max_scores = (
    ling_df
    .groupby(["linguistic_group", "source"])["rank_score"]
    .max()
    .reset_index()
)

ling_pivot = (
    ling_max_scores
    .pivot(index="linguistic_group", columns="source", values="rank_score")
    .reindex(linguistic_order)
    .fillna(0)
)

fig, ax = plt.subplots(figsize=(12, 6))

ling_pivot.plot(kind="bar", ax=ax, width=0.8)

plt.ylabel("Maximum rank score within subgroup")
plt.xlabel("")
plt.title(
    "Rank of the Top Feature per Linguistic Subgroup Across Models (DOI Presence Prediction)"
)
plt.xticks(rotation=30, ha="right")
plt.legend(title="Importance source")
plt.tight_layout()

ling_plot_path = OUTPUT_DIR / "doi_linguistic_subgroup_importance_MAX_SCORE.png"

plt.savefig(ling_plot_path, dpi=300, bbox_inches="tight")
plt.close()

print(f"Saved DOI linguistic max-score plot:\n{ling_plot_path}")


# ============================================================
# 6. Rank + frequency summaries
# ============================================================

feature_summary = (
    df.groupby("feature")
      .agg(
          frequency=("source", "nunique"),
          total_rank_score=("rank_score", "sum"),
          mean_rank=("rank", "mean"),
          best_rank=("rank", "min"),
          feature_group=("main_group", "first")
      )
      .sort_values(
          ["frequency", "total_rank_score", "best_rank"],
          ascending=[False, False, True]
      )
)

print("\n" + "=" * 70)
print("MOST CONSISTENTLY IMPORTANT FEATURES ACROSS MODELS")
print("=" * 70)
print(feature_summary.head(40).to_string())


group_summary = (
    df.groupby("main_group")
      .agg(
          total_mentions=("feature", "count"),
          unique_features=("feature", "nunique"),
          total_rank_score=("rank_score", "sum"),
          mean_rank_score=("rank_score", "mean"),
          max_rank_score=("rank_score", "max")
      )
      .sort_values("max_rank_score", ascending=False)
)

print("\n" + "=" * 70)
print("MAIN FEATURE GROUP SUMMARY")
print("=" * 70)
print(group_summary.to_string())


ling_summary = (
    df[df["linguistic_group"].notna()]
      .groupby("linguistic_group")
      .agg(
          total_mentions=("feature", "count"),
          unique_features=("feature", "nunique"),
          total_rank_score=("rank_score", "sum"),
          mean_rank_score=("rank_score", "mean"),
          max_rank_score=("rank_score", "max")
      )
      .sort_values("max_rank_score", ascending=False)
)

print("\n" + "=" * 70)
print("LINGUISTIC SUBGROUP SUMMARY")
print("=" * 70)
print(ling_summary.to_string())


all5 = feature_summary[feature_summary["frequency"] == 5]

print("\n" + "=" * 70)
print("FEATURES APPEARING IN ALL 5 MODELS")
print("=" * 70)
print(all5.to_string())


print("\n" + "=" * 70)
print("TOP FEATURES WITHIN EACH FEATURE GROUP")
print("=" * 70)

for group in sorted(df["main_group"].unique()):
    subset = (
        feature_summary[
            feature_summary["feature_group"] == group
        ]
        .head(10)
    )

    print(f"\n--- {group} ---")
    print(subset.to_string())


ling_features = (
    feature_summary[
        feature_summary["feature_group"] == "F. Linguistic quality"
    ]
    .copy()
)

print("\n" + "=" * 70)
print("LINGUISTIC FEATURES ONLY")
print("=" * 70)
print(ling_features.head(30).to_string())


# ============================================================
# 7. Combined across all 5 models: max-score method
# ============================================================

main_combined = (
    main_max_scores
    .groupby("main_group")["rank_score"]
    .mean()
    .reindex(main_group_order)
    .fillna(0)
)

fig, ax = plt.subplots(figsize=(10, 6))

main_combined.plot(kind="bar", ax=ax, width=0.7)

plt.ylabel("Mean maximum rank score")
plt.xlabel("")
plt.title(
    "Average Rank of the Top Feature per Group Across Models (DOI Presence Prediction)"
)
plt.xticks(rotation=30, ha="right")
plt.tight_layout()

combined_main_path = OUTPUT_DIR / "doi_main_feature_groups_importance_COMBINED_MAX_SCORE.png"

plt.savefig(combined_main_path, dpi=300, bbox_inches="tight")
plt.close()

print(f"Saved DOI combined main max-score plot:\n{combined_main_path}")


ling_combined = (
    ling_max_scores
    .groupby("linguistic_group")["rank_score"]
    .mean()
    .reindex(linguistic_order)
    .fillna(0)
)

fig, ax = plt.subplots(figsize=(10, 6))

ling_combined.plot(kind="bar", ax=ax, width=0.7)

plt.ylabel("Mean maximum rank score")
plt.xlabel("")
plt.title(
    "Average Rank of the Top Feature per Linguistic Subgroup Across Models (DOI Presence Prediction)"
)
plt.xticks(rotation=30, ha="right")
plt.tight_layout()

combined_ling_path = OUTPUT_DIR / "doi_linguistic_subgroup_importance_COMBINED_MAX_SCORE.png"

plt.savefig(combined_ling_path, dpi=300, bbox_inches="tight")
plt.close()

print(f"Saved DOI combined linguistic max-score plot:\n{combined_ling_path}")