# Language Quality and Publication Success

This repository contains the code and data for the thesis:

> **Language Quality and Publication Success: A Computational Study of Scientific Papers**

---

## Quick Start

Requires Python 3.10+. Create a virtual environment and install dependencies:

```bash
python -m venv .venv
source .venv/bin/activate        # Linux/macOS
# .venv\Scripts\activate         # Windows

pip install -r requirements.txt
```

See [HANDBOOK.md](HANDBOOK.md) for full installation notes (GPU/CUDA setup, optional packages) and step-by-step execution instructions.

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

## Reproducing the Main Results

To reproduce the experiments reported in the thesis:

1. Install dependencies from `requirements.txt`
2. Use the provided train/validation/test splits in `data/`
3. Run the model selection scripts in `code/06_model_selection/`
4. Run the final model scripts in `code/07_models/`
5. Run the ablation and feature importance analyses in `code/08_analysis/`

Detailed instructions are provided in [HANDBOOK.md](HANDBOOK.md).

---

## External Dependencies Not Included

The following resources are not distributed with this submission:

- Raw paper PDFs used for full-text feature extraction
- Qwen2.5-14B-Instruct-AWQ model weights
- GPT-2 model weights
- HPC cluster environment

All scripts required to reproduce the feature extraction process are included.

---

## Execution

See [HANDBOOK.md](HANDBOOK.md) for source code descriptions, scripts-by-experiment mapping, step-by-step execution instructions, and runtime requirements.
