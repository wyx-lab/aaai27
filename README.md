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
bash scripts/dump_alpha158.sh
```

Override defaults with environment variables:

```bash
INSTRUMENTS=csi300 START=2018-01-01 END=2023-12-31 bash scripts/dump_alpha158.sh
```

## Train MDGNN-lite

```bash
bash scripts/train_mdgnn_lite.sh
```

Override defaults with environment variables:

```bash
EPOCHS=50 BATCH_SIZE=4 WINDOW=20 bash scripts/train_mdgnn_lite.sh
```

Use explicit train/validation target-date ranges:

```bash
TRAIN_START=2018-01-01 TRAIN_END=2022-12-31 \
VALID_START=2023-01-01 VALID_END=2023-12-31 \
bash scripts/train_mdgnn_lite.sh
```

Training logs include dataset range, train/validation ranges, batch tensor
shapes, loss function, validation loss, validation IC, and validation RankIC.

For numerical stability, features are standardized and clipped by default, and
labels are clipped before training:

```bash
FEATURE_CLIP=10.0 LABEL_CLIP=0.2 STANDARDIZE=1 bash scripts/train_mdgnn_lite.sh
```

The trainer can also consume relation graphs:

- `.npy` / `.npz`: `[num_relations, num_stocks, num_stocks]`
- `.csv`: `src,dst,relation,weight`

When no relation graph is provided, it uses identity edges so the data/model
pipeline can be tested first.

Example with relations:

```bash
RELATIONS=data/relations/csi300_relations.npy bash scripts/train_mdgnn_lite.sh
```

Debug NaN losses:

```bash
DEBUG=1 DEBUG_BATCHES=3 bash scripts/train_mdgnn_lite.sh
```
