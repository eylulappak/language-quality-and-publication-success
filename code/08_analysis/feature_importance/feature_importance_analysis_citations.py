import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# ============================================================
# 1. Output directory
# ============================================================

OUTPUT_DIR = Path("results/feature_importance_plots")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# 2. Citation-count feature lists
# ============================================================

xgboost_gain = [
    "publication_source_arXiv",
    "has_MAG",
    "publication_source_Missing",
    "primary_category_Computer_Science",
    "is_conference",
    "max_author_hindex",
    "publication_source_Physical_Review_Letters",
    "reference_count",
    "is_journal_article",
    "publication_source_International_Conference_on_Learning_Representations",
    "publication_source_Neural_Information_Processing_Systems",
    "mean_author_hindex",
    "publication_source_Other",
    "publication_source_Journal_of_Physics_Conference_Series",
    "publication_source_Physical_Review_B",
    "verdict",
    "publication_source_Proceedings_of_the_International_Astronomical_Union",
    "mean_author_citation_count",
    "year",
    "paper_age",
    "publication_source_Physical_Review_D",
    "n_characters",
    "num_authors",
    "first_author_hindex",
    "is_ACL",
    "native_like_score",
    "mean_author_paper_count",
    "primary_category_Mathematics",
    "primary_category_Physics",
    "n_sentences"
]

xgboost_split = [
    "mean_author_paper_count",
    "reference_count",
    "mean_author_hindex",
    "mean_author_citation_count",
    "first_author_hindex",
    "dale_chall_mean",
    "year",
    "title_length",
    "abstract_length",
    "abstract_perplexity",
    "grammar_error_rate",
    "n_sentences",
    "last_author_hindex",
    "words_per_sentence",
    "chars_per_word",
    "n_characters",
    "max_author_hindex",
    "coleman_liau_mean",
    "linsear_write_mean",
    "num_authors",
    "grammar_edits",
    "abstract_coleman_liau_mean",
    "abstract_dale_chall_mean",
    "n_words",
    "gunning_fog_mean",
    "flesch_mean",
    "ari_mean",
    "abstract_gunning_fog_mean",
    "abstract_linsear_write_mean",
    "abstract_flesch_mean"
]

lgbm_gain = [
    "publication_source_arXiv",
    "reference_count",
    "mean_author_hindex",
    "has_MAG",
    "mean_author_paper_count",
    "mean_author_citation_count",
    "publication_source_Missing",
    "year",
    "max_author_hindex",
    "first_author_hindex",
    "num_authors",
    "is_journal_article",
    "n_sentences",
    "n_characters",
    "is_conference",
    "primary_category_Computer_Science",
    "chars_per_word",
    "native_like_score",
    "publication_source_Neural_Information_Processing_Systems",
    "last_author_hindex",
    "abstract_length",
    "n_words",
    "title_length",
    "publication_source_Physical_Review_Letters",
    "dale_chall_mean",
    "coleman_liau_mean",
    "grammar_error_rate",
    "words_per_sentence",
    "abstract_perplexity",
    "flesch_mean"
]

lgbm_split = [
    "mean_author_paper_count",
    "mean_author_hindex",
    "reference_count",
    "year",
    "mean_author_citation_count",
    "first_author_hindex",
    "dale_chall_mean",
    "title_length",
    "n_sentences",
    "num_authors",
    "abstract_length",
    "chars_per_word",
    "grammar_error_rate",
    "last_author_hindex",
    "words_per_sentence",
    "abstract_perplexity",
    "n_characters",
    "max_author_hindex",
    "coleman_liau_mean",
    "flesch_mean",
    "linsear_write_mean",
    "grammar_edits",
    "gunning_fog_mean",
    "abstract_coleman_liau_mean",
    "abstract_dale_chall_mean",
    "publication_source_arXiv",
    "is_journal_article",
    "n_words",
    "abstract_flesch_mean",
    "abstract_gunning_fog_mean"
]

logreg = [
    "year",
    "reference_count",
    "mean_author_paper_count",
    "has_MAG",
    "max_author_hindex",
    "mean_author_hindex",
    "publication_source_Physical_Review_Letters",
    "is_journal_article",
    "publication_source_Neural_Information_Processing_Systems",
    "publication_source_Physical_Review_D",
    "grammar_edits",
    "publication_source_Physical_Review_B",
    "publication_source_Physical_Review_A",
    "publication_source_The_Astrophysical_Journal",
    "publication_source_Physical_Review_C",
    "publication_source_International_Conference_on_Learning_Representations",
    "publication_source_Europhysics_Letters",
    "is_ACL",
    "publication_source_Monthly_Notices_of_the_Royal_Astronomical_Society",
    "grammar_error_rate",
    "publication_source_Journal_of_Cosmology_and_Astroparticle_Physics",
    "publication_source_International_Conference_on_Machine_Learning",
    "publication_source_New_Journal_of_Physics",
    "primary_category_Mathematics",
    "primary_category_Computer_Science",
    "publication_source_Astronomy_and_Astrophysics",
    "publication_source_The_Astrophysical_Journal_Letters",
    "first_author_hindex",
    "linsear_write_mean",
    "publication_source_Classical_and_Quantum_Gravity"
]

sources = {
    "XGBoost split": xgboost_split,
    "XGBoost gain": xgboost_gain,
    "LGBM split": lgbm_split,
    "LGBM gain": lgbm_gain,
    "Logistic Regression": logreg
}


# ============================================================
# 3. Feature group mapping
# ============================================================

def assign_main_group(feature):
    feature = feature.replace(" ", "_")

    if feature in ["year", "paper_age"]:
        return "A. Temporal"

    if feature in [
        "n_characters", "n_words", "n_sentences", "words_per_sentence",
        "chars_per_word", "title_length", "abstract_length"
    ]:
        return "B. Text length / structural"

    if (
        feature.startswith("publication_source_")
        or feature in [
            "is_journal_article", "is_conference",
            "is_review", "is_book", "has_MAG", "is_ACL",
            "is_PubMedCentral"
        ]
    ):
        return "C. Publication / indexing metadata"


    if (feature == "reference_count"):
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
    feature = feature.replace(" ", "_")

    if feature in ["grammar_edits", "grammar_error_rate"]:
        return "Grammatical accuracy"

    if feature in ["native_like_score", "confidence", "verdict"]:
        return "Native-like language use"

    if "abstract_perplexity" in feature:
        return "Perplexity"

    if feature.startswith("abstract_") and any(k in feature for k in [
        "flesch", "gunning", "coleman", "dale", "ari", "linsear", "readability"
    ]):
        return "Abstract readability"

    if any(k in feature for k in [
        "flesch", "gunning", "coleman", "dale", "ari", "linsear"
    ]):
        return "Paper readability"

    return None


# ============================================================
# 4. Create rank-weighted dataframe
# ============================================================

rows = []

for source, features in sources.items():
    n = len(features)

    for rank, feature in enumerate(features, start=1):
        clean_feature = feature.replace(" ", "_")
        rank_score = n - rank + 1

        rows.append({
            "source": source,
            "feature": clean_feature,
            "rank": rank,
            "rank_score": rank_score,
            "main_group": assign_main_group(clean_feature),
            "linguistic_group": assign_linguistic_group(clean_feature)
        })

df = pd.DataFrame(rows)


# ============================================================
# 5. Rank and frequency summaries
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

print("\n" + "=" * 80)
print("MOST CONSISTENTLY IMPORTANT FEATURES")
print("=" * 80)
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

print("\n" + "=" * 80)
print("MAIN FEATURE GROUP SUMMARY")
print("=" * 80)
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

print("\n" + "=" * 80)
print("LINGUISTIC SUBGROUP SUMMARY")
print("=" * 80)
print(ling_summary.to_string())

print("\n" + "=" * 80)
print("FEATURES APPEARING IN ALL 5 SOURCES")
print("=" * 80)
print(feature_summary[feature_summary["frequency"] == 5].to_string())

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
# 6. Main feature groups: max rank-score method
# ============================================================

main_group_order = [
    "A. Temporal",
    "B. Text length / structural",
    "C. Publication / indexing metadata",
    "D. Reference count",
    "E. Research field",
    "F. Author impact",
    "G. Linguistic quality"
]

main_max_scores = (
    df[df["main_group"].isin(main_group_order)]
    .groupby(["main_group", "source"])["rank_score"]
    .max()
    .reset_index()
)

main_pivot = (
    main_max_scores
    .pivot(
        index="main_group",
        columns="source",
        values="rank_score"
    )
    .reindex(main_group_order)
    .fillna(0)
)

fig, ax = plt.subplots(figsize=(12, 6))

main_pivot.plot(
    kind="bar",
    ax=ax,
    width=0.8
)

plt.ylabel("Maximum rank score within group")
plt.xlabel("")
plt.title(
    "Rank of the Top Feature per Group Across Models (Citation Count Prediction)"
)

plt.xticks(rotation=30, ha="right")
plt.legend(title="Importance source")
plt.tight_layout()

main_plot_path = OUTPUT_DIR / "citation_main_feature_groups_importance_MAX_SCORE.png"

plt.savefig(main_plot_path, dpi=300, bbox_inches="tight")
plt.close()

print(f"\nSaved main max-score plot:\n{main_plot_path}")


# ============================================================
# 7. Linguistic feature groups: max rank-score method
# ============================================================

linguistic_order = [
    "Paper readability",
    "Abstract readability",
    "Perplexity",
    "Grammatical accuracy",
    "Native-like language use"
]

ling_df = df[(df["linguistic_group"].notna())].copy()

ling_max_scores = (
    ling_df
    .groupby(["linguistic_group", "source"])["rank_score"]
    .max()
    .reset_index()
)

ling_pivot = (
    ling_max_scores
    .pivot(
        index="linguistic_group",
        columns="source",
        values="rank_score"
    )
    .reindex(linguistic_order)
    .fillna(0)
)

fig, ax = plt.subplots(figsize=(12, 6))

ling_pivot.plot(
    kind="bar",
    ax=ax,
    width=0.8
)

plt.ylabel("Maximum rank score within subgroup")
plt.xlabel("")
plt.title(
    "Rank of the Top Feature per Linguistic Subgroup Across Models (Citation Count Prediction)"
)

plt.xticks(rotation=30, ha="right")
plt.legend(title="Importance source")
plt.tight_layout()

ling_plot_path = OUTPUT_DIR / "citation_linguistic_subgroup_importance_MAX_SCORE.png"

plt.savefig(ling_plot_path, dpi=300, bbox_inches="tight")
plt.close()

print(f"Saved linguistic max-score plot:\n{ling_plot_path}")


# ============================================================
# 8. Combined across all 5 models
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
    "Average Rank of the Top Feature per Group Across Models (Citation Count Prediction)"
)

plt.xticks(rotation=30, ha="right")
plt.tight_layout()

combined_main_path = OUTPUT_DIR / "citation_main_feature_groups_importance_COMBINED_MAX_SCORE.png"

plt.savefig(combined_main_path, dpi=300, bbox_inches="tight")
plt.close()

print(f"Saved combined main max-score plot:\n{combined_main_path}")


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
    "Average Rank of the Top Feature per Linguistic Subgroup Across Models (Citation Count Prediction)"
)

plt.xticks(rotation=30, ha="right")
plt.tight_layout()

combined_ling_path = OUTPUT_DIR / "citation_linguistic_subgroup_importance_COMBINED_MAX_SCORE.png"

plt.savefig(combined_ling_path, dpi=300, bbox_inches="tight")
plt.close()

print(f"Saved combined linguistic max-score plot:\n{combined_ling_path}")