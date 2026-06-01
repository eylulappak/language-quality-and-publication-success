# Language Quality and Publication Success

This repository contains the code and data for the thesis:

> **Language Quality and Publication Success: A Computational Study of Scientific Papers**

---

## Thesis

The thesis PDF is located at:

```
thesis/thesis.pdf
```

---

## Folder Structure

```
language-quality-and-publication-success/
│
├── thesis/          # Thesis PDF and source files
├── data/            # Datasets used in the thesis
├── code/            # Source code
├── results/         # Experimental results and generated outputs
├── README.md
└── HANDBOOK.pdf
```

---

## Datasets

| File | Description |
|---|---|
| `data/full_arxiv_before_sampling.csv` | Full arXiv metadata pool before stratified sampling |
| `data/90k_arxiv_metadata_from_semantic_scholar.csv` | 90k sampled papers with Semantic Scholar metadata and abstracts |
| `data/90k_arxiv_citation_prediction_full.csv` | Fully merged dataset for citation count prediction (all features) |
| `data/90k_arxiv_doi_prediction_full.csv` | Fully merged dataset for DOI presence prediction (all features) |
| `data/200-paper-benchmark_validation_of_linguistic_features.csv` | 200-paper benchmark with ground-truth labels and computed features, used for feature validation |
| `data/90k_arxiv_citation_prediction_splits/` | Train / validation / test splits for citation prediction |
| `data/90k_arxiv_doi_prediction_splits/` | Train / validation / test splits for DOI prediction |
| `data/200-paper-benchmark/` | Raw PDFs, JSONs, TXTs, grammar outputs, and LLM (Qwen) outputs for the 200-paper benchmark |

---

## Source Code

### `code/01_data_collection/`

| Script | Description |
|---|---|
| `stratified_sampler.py` | Draws a stratified 90k sample from the full arXiv pool, balanced by year, citation count, research field, and DOI presence |
| `clean_abstracts.py` | Cleans and deduplicates abstracts: removes HTML/LaTeX artifacts, filters low-quality abstracts, removes near-duplicates |
| `pdf2txt_with_fitz.py` | Converts a single PDF paper to a structured JSON using PyMuPDF |
| `json2txt.py` | Converts paper JSONs to plain-text files; strips references, repairs hyphenated line breaks |

### `code/02_feature_extraction/`

| Script | Description |
|---|---|
| `readability_on_arxiv_abstracts.py` | Computes seven readability metrics (Flesch, Gunning Fog, Dale-Chall, etc.) on abstract text |
| `readability_on_arxiv_papers.py` | Computes the same readability metrics on full paper text via chunked aggregation |
| `perplexity_arxiv_abstracts.py` | Computes GPT-2 perplexity on abstract text using non-overlapping token windows |
| `detect_grammar.py` | Counts grammar errors in paper text using a T5-based grammar error correction model |
| `qwen_native_like_prediction.py` | Assigns a native-like language score (0–10) and binary verdict to each paper using Qwen2.5-14B |

### `code/03_validation_of_language_quality_features/`

| Script | Description |
|---|---|
| `validation_of_language_quality_features.py` | Validates that language quality features discriminate native-like from non-native-like papers on the 200-paper benchmark using Mann-Whitney U tests and rank-biserial correlations |

### `code/04_preprocessing/`

| Script | Description |
|---|---|
| `preprocess.py` | Merges all features, handles imputation, feature engineering, and one-hot encoding; produces separate datasets for citation and DOI prediction |
| `generate_splits_citations.py` | Generates stratified train/val/test splits for citation prediction |
| `generate_splits_doi.py` | Generates stratified train/val/test splits for DOI prediction |

### `code/05_exploratory_and_correlation_analysis/`

| Script | Description |
|---|---|
| `eda_and_correlation_analysis.py` | Descriptive statistics, Mann-Whitney U group tests, Spearman correlations, and distribution plots for all feature groups |

### `code/06_model_selection/`

| Script | Description |
|---|---|
| `model_and_threshold_selection_for_citations.py` | Evaluates three model types across seven citation thresholds on the validation set to select the best threshold and model |
| `model_selection_for_doi.py` | Evaluates three model types for DOI prediction on the validation set |

### `code/07_models/`

| Script | Description |
|---|---|
| `citation_prediction/lightgbm_citation_prediction.py` | Trains and evaluates LightGBM for citation count prediction (citation\_count > 5) |
| `citation_prediction/xgboost_citation_prediction.py` | Trains and evaluates XGBoost for citation count prediction |
| `citation_prediction/logistic_regression_citation_prediction.py` | Trains and evaluates logistic regression (with p-values) for citation count prediction |
| `doi_prediction/lightgbm_doi_prediction.py` | Trains and evaluates LightGBM for DOI presence prediction |
| `doi_prediction/xgboost_doi_prediction.py` | Trains and evaluates XGBoost for DOI presence prediction |
| `doi_prediction/logistic_regression_doi_prediction.py` | Trains and evaluates logistic regression (with p-values) for DOI presence prediction |

### `code/08_analysis/`

| Script | Description |
|---|---|
| `ablation/ablations_for_citation_prediction.py` | Runs 25 ablation configurations for citation prediction to quantify each feature group's contribution |
| `ablation/ablations_for_doi_prediction.py` | Runs 23 ablation configurations for DOI prediction |
| `feature_importance/feature_importance_analysis_citations.py` | Aggregates feature importance across XGBoost, LightGBM, and logistic regression for citation prediction |
| `feature_importance/feature_importance_analysis_doi.py` | Aggregates feature importance across models for DOI prediction |

### `code/09_cluster_jobs/`

SLURM batch scripts used to run computationally intensive feature extraction on an HPC cluster. Not needed for local execution.

| Script | Description |
|---|---|
| `run_grammar_detection.sbatch` | Launches `detect_grammar.py` as a 50-shard GPU array job |
| `run_qwen_awq.sbatch` | Launches `qwen_native_like_prediction.py` as a batched GPU array job |

---

## Scripts by Experiment

### Experiment 1 — Feature validation (200-paper benchmark)

```
code/03_validation_of_language_quality_features/validation_of_language_quality_features.py
```

Validates that each language quality feature (perplexity, readability, grammar error rate, native-like language use) is significantly higher or lower in native-like papers compared to non-native-like papers.

---

### Experiment 2 — Exploratory analysis

```
code/05_exploratory_and_correlation_analysis/eda_and_correlation_analysis.py
```

Examines distributions and Spearman correlations between language quality features and publication success outcomes (citation count and DOI presence) across 90k papers.

---

### Experiment 3 — Citation count prediction

```
code/06_model_selection/model_and_threshold_selection_for_citations.py
code/07_models/citation_prediction/lightgbm_citation_prediction.py
code/07_models/citation_prediction/xgboost_citation_prediction.py
code/07_models/citation_prediction/logistic_regression_citation_prediction.py
```

Predicts whether a paper receives more than 5 citations. Three classifiers are trained and evaluated; the threshold and model type are selected via `model_and_threshold_selection_for_citations.py`.

---

### Experiment 4 — DOI presence prediction

```
code/06_model_selection/model_selection_for_doi.py
code/07_models/doi_prediction/lightgbm_doi_prediction.py
code/07_models/doi_prediction/xgboost_doi_prediction.py
code/07_models/doi_prediction/logistic_regression_doi_prediction.py
```

Predicts whether a paper has been assigned a DOI (a proxy for formal publication). Three classifiers are trained and evaluated.

---

### Experiment 5 — Ablation study

```
code/08_analysis/ablation/ablations_for_citation_prediction.py
code/08_analysis/ablation/ablations_for_doi_prediction.py
```

Systematically removes feature groups (and tests group-only models) to quantify each group's contribution to predictive performance. Reports delta ROC-AUC and delta balanced accuracy against the full model.

---

### Experiment 6 — Feature importance analysis

```
code/08_analysis/feature_importance/feature_importance_analysis_citations.py
code/08_analysis/feature_importance/feature_importance_analysis_doi.py
```

Aggregates feature importance rankings from XGBoost (gain/split), LightGBM (gain/split), and logistic regression to identify the most consistently important features across all models.

---

## Execution

See [HANDBOOK.pdf] for step-by-step instructions, installation commands, and runtime requirements.
