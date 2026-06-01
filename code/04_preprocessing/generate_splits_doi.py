from sklearn.model_selection import train_test_split
import pandas as pd 

DOI_CSV = "data/90k_arxiv_doi_prediction_full.csv"
doi_df = pd.read_csv(DOI_CSV, dtype={"arxiv_id": str})

train_val_df, test_df = train_test_split(
    doi_df,
    test_size=0.2,
    stratify=doi_df["has_doi"],
    random_state=42
)

train_df, val_df = train_test_split(
    train_val_df,
    test_size=0.25,  # 0.25 of 80% = 20% of full data
    stratify=train_val_df["has_doi"],
    random_state=42
)

import os
os.makedirs("data/90k_arxiv_doi_prediction_splits", exist_ok=True)
pd.DataFrame(train_df).to_csv("data/90k_arxiv_doi_prediction_splits/90k_arxiv_doi_prediction_train.csv", index=False)
pd.DataFrame(val_df).to_csv("data/90k_arxiv_doi_prediction_splits/90k_arxiv_doi_prediction_val.csv", index=False)
pd.DataFrame(test_df).to_csv("data/90k_arxiv_doi_prediction_splits/90k_arxiv_doi_prediction_test.csv", index=False)