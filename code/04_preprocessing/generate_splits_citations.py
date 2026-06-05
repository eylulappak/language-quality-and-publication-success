"""
Splits the citation prediction dataset into stratified train/val/test sets (60/20/20).
Stratification is by citation count bin to preserve class distribution across splits.
Input:  data/90k_arxiv_citation_prediction_full.csv
Output: data/90k_arxiv_citation_prediction_splits/
"""
from sklearn.model_selection import train_test_split
import pandas as pd
import os

CITATION_CSV = "data/90k_arxiv_citation_prediction_full.csv"

citation_df = pd.read_csv(CITATION_CSV, dtype={"arxiv_id": str})

# Bin edges chosen to match the threshold used for the binary prediction task (>5) and capture natural citation tiers
citation_df["citation_bin"] = pd.cut(
    citation_df["citation_count"],
    bins=[-1, 0, 4, 12, 30, float("inf")],
    labels=["0", "1-4", "5-12", "13-30", "31+"]
)

train_val_df, test_df = train_test_split(
    citation_df,
    test_size=0.2,
    # stratify on citation_bin so all five citation tiers are proportionally represented in every split
    stratify=citation_df["citation_bin"],
    random_state=42
)

train_df, val_df = train_test_split(
    train_val_df,
    test_size=0.25,  # 0.25 of 80% = 20% of full data
    stratify=train_val_df["citation_bin"],
    random_state=42
)

output_dir = "data/90k_arxiv_citation_prediction_splits"
os.makedirs(output_dir, exist_ok=True)

train_df.to_csv(f"{output_dir}/90k_arxiv_citation_prediction_train.csv", index=False)
val_df.to_csv(f"{output_dir}/90k_arxiv_citation_prediction_val.csv", index=False)
test_df.to_csv(f"{output_dir}/90k_arxiv_citation_prediction_test.csv", index=False)