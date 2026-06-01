#!/usr/bin/env python3

import os
import numpy as np
import pandas as pd


INPUT_PATH = "/mnt/beegfs/work/appak/arxiv_90k_full_joined_ss_final.csv"

OUT_DIR = "/mnt/beegfs/work/appak/final_pipeline/preprocessed_90k_classifiers"
os.makedirs(OUT_DIR, exist_ok=True)

CITATION_RAW_OUT = os.path.join(
    OUT_DIR, "arxiv_90k_citation_prediction_preprocessed_before_encoding.csv"
)
CITATION_ENCODED_OUT = os.path.join(
    OUT_DIR, "arxiv_90k_citation_prediction_preprocessed_after_encoding.csv"
)
DOI_RAW_OUT = os.path.join(
    OUT_DIR, "arxiv_90k_doi_prediction_preprocessed_before_encoding.csv"
)
DOI_ENCODED_OUT = os.path.join(
    OUT_DIR, "arxiv_90k_doi_prediction_preprocessed_after_encoding.csv"
)


TOP_FIELDS = [
    "Physics",
    "Computer Science",
    "Mathematics",
    "Engineering",
    "Materials Science",
    "Environmental Science",
    "Medicine",
    "Economics",
    "Chemistry",
    "Biology",
]


def move_targets_to_end(data, target_cols):
    target_cols = [c for c in target_cols if c in data.columns]
    other_cols = [c for c in data.columns if c not in target_cols]
    return data[other_cols + target_cols]


def move_col_after(data, col_to_move, after_col):
    if col_to_move not in data.columns or after_col not in data.columns:
        return data

    cols = [c for c in data.columns if c != col_to_move]
    insert_pos = cols.index(after_col) + 1
    cols.insert(insert_pos, col_to_move)
    return data[cols]


def clean_category_fields(data):
    if "primary_category" in data.columns:
        data["primary_category"] = np.where(
            data["primary_category"].isin(TOP_FIELDS),
            data["primary_category"],
            "Other"
        )
    return data


def clean_venue(data):
    if "venue" not in data.columns:
        return data

    data["venue"] = data["venue"].fillna("Missing")
    counts = data["venue"].value_counts(dropna=False)

    keep_values = set(counts[counts >= 100].index)
    keep_values.add("Missing")
    keep_values.add("arXiv.org")

    data["venue"] = np.where(data["venue"].isin(keep_values), data["venue"], "Other")
    return data


def clean_journal_name(data):
    if "journal_name" not in data.columns:
        return data

    data["journal_name"] = data["journal_name"].fillna("Missing")

    arxiv_mask = data["journal_name"].astype(str).str.contains(
        "arxiv", case=False, na=False
    )
    data.loc[arxiv_mask, "journal_name"] = "ArXiv"

    counts = data["journal_name"].value_counts(dropna=False)

    keep_values = set(counts[counts >= 100].index)
    keep_values.add("Missing")
    keep_values.add("ArXiv")

    data["journal_name"] = np.where(
        data["journal_name"].isin(keep_values),
        data["journal_name"],
        "Other",
    )

    return data


def split_publication_types(data):
    if "publication_types" not in data.columns:
        return data

    pub = data["publication_types"].fillna("").astype(str)

    data["is_journal_article"] = pub.str.contains(
        "JournalArticle", case=False, na=False
    ).astype(int)
    data["is_conference"] = pub.str.contains(
        "Conference", case=False, na=False
    ).astype(int)
    data["is_review"] = pub.str.contains(
        "Review", case=False, na=False
    ).astype(int)
    data["is_book"] = pub.str.contains(
        "Book", case=False, na=False
    ).astype(int)

    data = data.drop(columns=["publication_types"])

    return data


df = pd.read_csv(INPUT_PATH, low_memory=False, dtype={"arxiv_id": str})

print(f"Loaded dataset: {df.shape[0]:,} rows, {df.shape[1]:,} columns")


rename_map = {
    "journal.name": "journal_name",
    "category1": "primary_category",
    "corpusid": "corpus_id",
    "citationcount": "citation_count",
    "influentialcitationcount": "influential_citation_count",
    "referencecount": "reference_count",
    "publicationtypes": "publication_types",
    "isopenaccess": "is_open_access",
    "sentences": "n_sentences",
    "paper_n_characters": "n_characters",
    "paper_n_words": "n_words",
    "errors_per_sentence": "grammar_error_rate",
}

df = df.rename(columns=rename_map)

df = df.drop(columns=["category2"], errors="ignore")


numeric_cols = [
    "year",
    "influential_citation_count",
    "reference_count",
    "title_length",
    "abstract_length",
    "has_MAG",
    "is_ACL",
    "is_PubMedCentral",
    "n_characters",
    "n_words",
    "flesch_kincaid_mean",
    "flesch_mean",
    "gunning_fog_mean",
    "coleman_liau_mean",
    "dale_chall_mean",
    "ari_mean",
    "linsear_write_mean",
    "n_tokens",
    "n_pred_tokens",
    "n_windows",
    "neg_log_likelihood",
    "perplexity",
    "native_like_score",
    "confidence",
    "verdict",
    "n_sentences",
    "grammar_edits",
    "grammar_error_rate",
    "first_author_hindex",
    "max_author_hindex",
    "mean_author_hindex",
    "num_authors",
    "last_author_hindex",
    "mean_author_paper_count",
    "mean_author_citation_count",
    "abstract_n_tokens",
    "abstract_n_pred_tokens",
    "abstract_n_windows",
    "abstract_neg_log_likelihood",
    "abstract_perplexity",
    "abstract_flesch_kincaid_mean",
    "abstract_flesch_mean",
    "abstract_gunning_fog_mean",
    "abstract_coleman_liau_mean",
    "abstract_dale_chall_mean",
    "abstract_ari_mean",
    "abstract_linsear_write_mean",
    "citation_count",
    "has_doi",
]

for col in numeric_cols:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")


citation_bins = [-0.1, 0, 4, 12, 30, np.inf]
citation_labels = ["0", "1-4", "5-12", "13-30", "31+"]

df["citation_count_bins"] = pd.cut(
    df["citation_count"],
    bins=citation_bins,
    labels=citation_labels,
    include_lowest=True,
)

citation_label_map = {
    "0": 0,
    "1-4": 1,
    "5-12": 2,
    "13-30": 3,
    "31+": 4,
}

df["citation_count_bins"] = (
    df["citation_count_bins"]
    .astype(str)
    .map(citation_label_map)
)


categorical_unknown_cols = [
    "venue",
    "journal_name",
    "primary_category",
]

for col in categorical_unknown_cols:
    if col in df.columns:
        df[col] = df[col].astype("object").fillna("Missing")


df = clean_category_fields(df)
df = clean_venue(df)
df = clean_journal_name(df)
df = split_publication_types(df)


binary_cols = [
    "verdict",
    "is_open_access",
    "is_journal_article",
    "is_conference",
    "is_review",
    "is_book",
]

for col in binary_cols:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).astype(int)


abstract_readability_cols = [
    "abstract_flesch_kincaid_mean",
    "abstract_flesch_mean",
    "abstract_gunning_fog_mean",
    "abstract_coleman_liau_mean",
    "abstract_dale_chall_mean",
    "abstract_ari_mean",
    "abstract_linsear_write_mean",
]

existing_abs_read_cols = [c for c in abstract_readability_cols if c in df.columns]

df["abstract_readability_missing"] = (
    df[existing_abs_read_cols].isna().any(axis=1).astype(int)
)

for col in existing_abs_read_cols:
    df[col] = df[col].fillna(df[col].median())


small_missing_numeric_cols = [
    "first_author_hindex",
    "last_author_hindex",
    "mean_author_citation_count",
    "mean_author_paper_count",
    "max_author_hindex",
    "mean_author_hindex",
]

for col in small_missing_numeric_cols:
    if col in df.columns:
        df[col] = df[col].fillna(df[col].median())


for col in df.select_dtypes(include=[np.number]).columns:
    if df[col].isna().sum() > 0:
        df[col] = df[col].fillna(df[col].median())


for col in df.select_dtypes(include=["object"]).columns:
    if col not in ["corpus_id", "arxiv_id"]:
        df[col] = df[col].fillna("Missing")


current_year = pd.Timestamp.now().year

df["words_per_sentence"] = df["n_words"] / df["n_sentences"].replace(0, np.nan)
df["chars_per_word"] = df["n_characters"] / df["n_words"].replace(0, np.nan)
df["log_perplexity"] = np.log(df["perplexity"].replace(0, np.nan))
df["abstract_log_perplexity"] = np.log(
    df["abstract_perplexity"].replace(0, np.nan)
)
df["paper_age"] = current_year - df["year"]

engineered_cols = [
    "words_per_sentence",
    "chars_per_word",
    "abstract_log_perplexity",
    "log_perplexity",
    "paper_age",
]

for col in engineered_cols:
    df[col] = df[col].replace([np.inf, -np.inf], np.nan)
    df[col] = df[col].fillna(df[col].median())


# Remove redundant perplexity-related columns.
redundant_perplexity_cols = [
    "neg_log_likelihood",
    "n_tokens",
    "n_pred_tokens",
    "abstract_neg_log_likelihood",
    "abstract_n_tokens",
    "abstract_n_pred_tokens",
]

df = df.drop(columns=[c for c in redundant_perplexity_cols if c in df.columns])

# Remove redundant chunk/window count columns
window_cols_to_drop = [
    "n_windows",
    "abstract_n_windows",
]

df = df.drop(
    columns=[c for c in window_cols_to_drop if c in df.columns]
)

df = move_col_after(df, "paper_age", "year")
df = move_col_after(df, "chars_per_word", "n_words")
df = move_col_after(df, "words_per_sentence", "n_sentences")
df = move_col_after(df, "log_perplexity", "perplexity")
df = move_col_after(df, "abstract_log_perplexity", "abstract_perplexity")
df = move_col_after(df, "abstract_readability_missing", "abstract_linsear_write_mean")

for pub_flag in ["is_book", "is_review", "is_conference", "is_journal_article"]:
    df = move_col_after(df, pub_flag, "reference_count")


citation_df = df.copy()
doi_df = df.copy()


citation_drop_cols = [
    "influential_citation_count",
    "title",
]

citation_df = citation_df.drop(
    columns=[c for c in citation_drop_cols if c in citation_df.columns]
)

doi_drop_cols = [
    "venue",
    "journal_name",
    "is_journal_article",
    "is_conference",
    "is_review",
    "is_book",
    "is_ACL",
    "is_PubMedCentral",
    "has_MAG",
    "citation_count",
    "citation_count_bins",
    "influential_citation_count",
    "title",
]

doi_df = doi_df.drop(
    columns=[c for c in doi_drop_cols if c in doi_df.columns]
)


citation_df = move_targets_to_end(citation_df, ["citation_count_bins"])
doi_df = move_targets_to_end(doi_df, ["has_doi"])


citation_df.to_csv(CITATION_RAW_OUT, index=False)
doi_df.to_csv(DOI_RAW_OUT, index=False)

print(f"\nSaved citation pre-encoding dataset to:\n{CITATION_RAW_OUT}")
print(f"Saved DOI pre-encoding dataset to:\n{DOI_RAW_OUT}")


print("\nMissing values in citation prediction dataset:")
print(citation_df.isna().sum().sort_values(ascending=False))

print("\nMissing values in DOI prediction dataset:")
print(doi_df.isna().sum().sort_values(ascending=False))


one_hot_cols = [
    "primary_category",
    "venue",
    "journal_name",
]


def one_hot_encode_dataset(data):
    cols_to_encode = [c for c in one_hot_cols if c in data.columns]

    encoded = pd.get_dummies(
        data,
        columns=cols_to_encode,
        dummy_na=False,
        drop_first=True,
    )

    encoded = encoded.replace({True: 1, False: 0})

    return encoded


citation_encoded = one_hot_encode_dataset(citation_df)
doi_encoded = one_hot_encode_dataset(doi_df)

citation_encoded = move_targets_to_end(citation_encoded, ["citation_count_bins"])
doi_encoded = move_targets_to_end(doi_encoded, ["has_doi"])

citation_encoded.to_csv(CITATION_ENCODED_OUT, index=False)
doi_encoded.to_csv(DOI_ENCODED_OUT, index=False)

print(f"\nSaved citation encoded dataset to:\n{CITATION_ENCODED_OUT}")
print(f"Saved DOI encoded dataset to:\n{DOI_ENCODED_OUT}")

print("\nDone.")