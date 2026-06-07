# Quant Trading Prediction Research

This repository contains a lightweight reproduction scaffold for stock-return
prediction experiments with Qlib Alpha158 features and an MDGNN-style model.

## Components

- `scripts/export_alpha158_qlib.py` exports Qlib Alpha158 features and labels to
  parquet files.
- `src/mdgnn_lite/` contains a PyTorch dataset, relation-graph loader,
  MDGNN-lite model, and training entrypoint.

Local PDFs and generated datasets are ignored by git.

## Linux Setup

Use Python 3.10 or 3.11.

```bash
python3.10 -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -r requirements.txt
```

## Download Qlib China Data

Qlib package versions differ in their data-download entrypoints. The most stable
path is to use the official repository script:

```bash
git clone https://github.com/microsoft/qlib.git /tmp/qlib
cd /tmp/qlib
python scripts/get_data.py qlib_data \
  --target_dir ~/.qlib/qlib_data/cn_data \
  --region cn
```

## Export Alpha158

From this repository:

```bash
python scripts/export_alpha158_qlib.py \
  --provider-uri ~/.qlib/qlib_data/cn_data \
  --instruments csi300 \
  --start 2018-01-01 \
  --end 2023-12-31 \
  --fit-start 2018-01-01 \
  --fit-end 2019-12-31 \
  --out-dir data/alpha158
```

## Train MDGNN-lite

```bash
python -m src.mdgnn_lite.train \
  --features data/alpha158/features_alpha158_csi300_20180101_20231231.parquet \
  --labels data/alpha158/labels_alpha158_csi300_20180101_20231231.parquet \
  --window 10 \
  --batch-size 8 \
  --epochs 20 \
  --out checkpoints/mdgnn_lite.pt
```

The trainer can also consume relation graphs:

- `.npy` / `.npz`: `[num_relations, num_stocks, num_stocks]`
- `.csv`: `src,dst,relation,weight`

When no relation graph is provided, it uses identity edges so the data/model
pipeline can be tested first.
