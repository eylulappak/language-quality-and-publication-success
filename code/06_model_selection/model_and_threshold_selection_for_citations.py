import os
import warnings
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, roc_auc_score
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")

TRAIN_CSV = "/mnt/beegfs/work/appak/final_pipeline/preprocessed_90k_classifiers/new_splits/arxiv_90k_citation_prediction_train.csv"
VAL_CSV = "/mnt/beegfs/work/appak/final_pipeline/preprocessed_90k_classifiers/new_splits/arxiv_90k_citation_prediction_val.csv"

OUTPUT_DIR = "/mnt/beegfs/work/appak/final_pipeline/preprocessed_90k_classifiers/model_selection"
os.makedirs(OUTPUT_DIR, exist_ok=True)

train_df = pd.read_csv(TRAIN_CSV, dtype={"arxiv_id": str})
val_df = pd.read_csv(VAL_CSV, dtype={"arxiv_id": str})

thresholds = [0, 1, 5, 10, 20, 50, 100]

drop_cols = [
    "arxiv_id",
    "corpus_id",
    "citation_count",
    "citation_bin",
]

feature_cols = [
    col for col in train_df.columns
    if col not in drop_cols
]

X_train = train_df[feature_cols].select_dtypes(include=["number"])
X_val = val_df[X_train.columns]

models = {
    "logistic_regression": Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("model", LogisticRegression(
            max_iter=2000,
            class_weight="balanced",
            random_state=42
        ))
    ]),

    "lightgbm": Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("model", LGBMClassifier(
            n_estimators=500,
            learning_rate=0.05,
            num_leaves=31,
            random_state=42,
            class_weight="balanced",
            verbose=-1
        ))
    ]),

    "xgboost": Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("model", XGBClassifier(
            n_estimators=500,
            learning_rate=0.05,
            max_depth=6,
            subsample=0.8,
            colsample_bytree=0.8,
            eval_metric="logloss",
            random_state=42,
            n_jobs=-1
        ))
    ])
}

results = []

for t in thresholds:
    y_train = (train_df["citation_count"] > t).astype(int)
    y_val = (val_df["citation_count"] > t).astype(int)

    print(f"\nThreshold: citation_count > {t}")
    print(f"Train positive rate: {y_train.mean():.4f}")
    print(f"Val positive rate:   {y_val.mean():.4f}")

    for model_name, model in models.items():
        model.fit(X_train, y_train)

        y_pred = model.predict(X_val)
        y_proba = model.predict_proba(X_val)[:, 1]

        result = {
            "threshold_t": t,
            "task": f"citation_count > {t}",
            "model": model_name,
            "val_accuracy": accuracy_score(y_val, y_pred),
            "val_balanced_accuracy": balanced_accuracy_score(y_val, y_pred),
            "val_roc_auc": roc_auc_score(y_val, y_proba),
            "train_positive_rate": y_train.mean(),
            "val_positive_rate": y_val.mean(),
            "n_train": len(y_train),
            "n_val": len(y_val),
            "n_features": X_train.shape[1],
        }

        results.append(result)

        print(
            f"{model_name}: "
            f"Acc={result['val_accuracy']:.4f}, "
            f"BalAcc={result['val_balanced_accuracy']:.4f}, "
            f"ROC-AUC={result['val_roc_auc']:.4f}"
        )

results_df = pd.DataFrame(results)

results_df = results_df.sort_values(
    by=["val_roc_auc", "val_balanced_accuracy", "val_accuracy"],
    ascending=False
)

results_path = f"{OUTPUT_DIR}/citation_threshold_model_selection_validation_results.csv"
results_df.to_csv(results_path, index=False)

print("\nBest model-threshold combination:")
print(results_df.head(1).T)

print(f"\nSaved validation model-selection results to:\n{results_path}")