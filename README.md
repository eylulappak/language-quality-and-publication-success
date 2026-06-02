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

See [HANDBOOK.pdf](HANDBOOK.pdf) for full installation notes (GPU/CUDA setup, optional packages) and step-by-step execution instructions.

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

## Reproducing the Main Results

To reproduce the experiments reported in the thesis:

1. Install dependencies from `requirements.txt`
2. Use the provided train/validation/test splits in `data/`
3. Run the model selection scripts in `code/06_model_selection/`
4. Run the final model scripts in `code/07_models/`
5. Run the ablation and feature importance analyses in `code/08_analysis/`

Detailed instructions are provided in [HANDBOOK.pdf](HANDBOOK.pdf).

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

See [HANDBOOK.pdf](HANDBOOK.pdf) for data and source code descriptions, step-by-step execution instructions, and runtime requirements.
