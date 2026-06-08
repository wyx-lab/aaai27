# Quant Trading Prediction Research

This repository contains a Qlib workflow for stock-return prediction experiments
with Alpha158 features and an MDGNN-style model.

## Components

- `configs/mdgnn_alpha158.yaml` runs Qlib Alpha158 + MDGNN-lite with qrun.
- `src/mdgnn_lite/qlib_model.py` implements a Qlib `Model` wrapper.
- `scripts/run_qlib_mdgnn.sh` starts the Qlib workflow.
- `src/mdgnn_lite/` contains a PyTorch dataset, relation-graph loader,
  MDGNN-lite model, and legacy standalone training entrypoint.

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

## Run Qlib MDGNN

The main entrypoint uses Qlib's DatasetH and Alpha158 handler, but does not use
Qlib workflow recorder. It directly fits the model and writes prediction scores
to parquet.

```bash
bash scripts/run_qlib_mdgnn.sh
```

The config uses:

- Alpha158 features
- CSI500 universe with `SH000905` benchmark
- five-day tradable return label
- `RobustZScoreNorm` + `Fillna` for features
- `DropnaLabel` + `CSRankNorm` for labels
- no Qlib recorder / Mongo / MLflow workflow dependency

Predictions are saved to:

```text
predictions/mdgnn_scores.parquet
```

The configured label is:

```text
LABEL0 = Ref($close, -LABEL_HORIZON) / Ref($close, -1) - 1
```

With the default `LABEL_HORIZON=5`, this is the tradable forward return from
T+1 close to T+5 close.

The default date range is `2018-01-01` to `2020-12-31` because the commonly
downloaded public Qlib China data package often ends around 2020. If your data
provider has newer bars, update `configs/mdgnn_alpha158.yaml`.

## Legacy Parquet Flow

The previous standalone PyTorch path is still available:

```bash
bash scripts/dump_alpha158.sh
bash scripts/train_mdgnn_lite.sh
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
TRAIN_START=2018-01-01 TRAIN_END=2019-12-31 \
VALID_START=2020-01-01 VALID_END=2020-06-30 \
bash scripts/train_mdgnn_lite.sh
```

Training logs include dataset range, train/validation ranges, batch tensor
shapes, loss function, validation loss, validation IC, and validation RankIC.

For numerical stability, features are normalized with training windows only and
clipped by default. Labels are clipped before training:

```bash
FEATURE_NORM=zscore FEATURE_CLIP=10.0 LABEL_CLIP=0.2 bash scripts/train_mdgnn_lite.sh
```

`FEATURE_NORM` supports `zscore`, `robust`, and `none`.

The trainer can also consume relation graphs:

- `.npy` / `.npz`: `[num_relations, num_stocks, num_stocks]`
- `.csv`: `src,dst,relation,weight`

When no relation graph is provided, it uses identity edges so the data/model
pipeline can be tested first.

Example with relations:

```bash
RELATIONS=data/relations/csi500_relations.npy bash scripts/train_mdgnn_lite.sh
```

Debug NaN losses:

```bash
DEBUG=1 DEBUG_BATCHES=3 bash scripts/train_mdgnn_lite.sh
```
