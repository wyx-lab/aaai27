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

## Build 2018-2025 Data From AKShare

AKShare does not require a token. Dump raw A-share data into Qlib format:

```bash
START=20180101 END=20251231 bash scripts/dump_akshare_qlib.sh
```

The dump writes Qlib instrument lists for `all`, `csi300`, and `csi500`.
If you already have CSV bars and only need to refresh these lists, run:

```bash
python scripts/write_akshare_instruments.py --csv-dir data/akshare_qlib_csv
```

If Qlib reports `instrument not exists`, copy the lists into the dumped provider
directory:

```bash
python scripts/write_akshare_instruments.py \
  --csv-dir data/akshare_qlib_csv \
  --qlib-dir ~/.qlib/qlib_data/akshare_cn_data
```

If a training segment is empty, inspect the provider and segments:

```bash
python scripts/check_qlib_segments.py --config configs/master_alpha158_akshare.yaml
```

If EastMoney/AKShare throttles or proxy disconnects, slow down requests:

```bash
SLEEP=1.5 RETRIES=5 RETRY_SLEEP=8 START=20180101 END=20251231 bash scripts/dump_akshare_qlib.sh
```

On Windows with Git Bash, use:

```bash
pip install -r requirements.txt
LIMIT=20 SKIP_DUMP=1 bash scripts/dump_akshare_qlib_windows.sh
```

Remove `SKIP_DUMP=1` after the CSV download works and Qlib is installed.

Smoke test with a small number of stocks:

```bash
LIMIT=20 START=20180101 END=20251231 bash scripts/dump_akshare_qlib.sh
```

Then run MDGNN with the AKShare-backed provider:

```bash
CONFIG=configs/mdgnn_alpha158_akshare.yaml bash scripts/run_qlib_mdgnn.sh
```

Switch the stock universe by editing one field in the AKShare config:

```yaml
instruments: csi300
```

or:

```yaml
instruments: csi500
```

Check the actual Qlib Alpha158 feature dimension:

```bash
python scripts/check_alpha158_features.py --config configs/mdgnn_alpha158_akshare.yaml
```

Summarize Alpha158 feature ranges after the configured model-side feature
normalization:

```bash
python scripts/summarize_alpha158_ranges.py --config configs/mdgnn_alpha158_akshare.yaml
```

Or:

```bash
bash scripts/summarize_alpha158_ranges.sh
```

## Build 2018-2025 Data From TuShare

Set your token and dump raw TuShare data into Qlib format:

```bash
export TUSHARE_TOKEN=your_token_here
START=20180101 END=20251231 bash scripts/dump_tushare_qlib.sh
```

Smoke test with a small number of stocks:

```bash
LIMIT=20 START=20180101 END=20251231 bash scripts/dump_tushare_qlib.sh
```

Then run MDGNN with the TuShare-backed provider:

```bash
CONFIG=configs/mdgnn_alpha158_tushare.yaml bash scripts/run_qlib_mdgnn.sh
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
- `Fillna` for features in Qlib
- `DropnaLabel` for labels in Qlib
- time-series z-score feature normalization fitted on the train segment
- robust median/IQR label normalization fitted on the train segment
- MSE loss on normalized five-day returns
- MASTER-style feature gate, temporal attention, and cross-stock attention
- prediction score uses the raw model output
- no Qlib recorder / Mongo / MLflow workflow dependency

Predictions and evaluation outputs are saved to:

```text
predictions/mdgnn_scores.parquet
predictions/mdgnn_metrics.json
predictions/mdgnn_daily_metrics.parquet
```

The evaluation reports test IC, RankIC, equal-weight Top-K return, Precision@K,
long-short Top-K minus Bottom-K return, annualized return, Sharpe, max drawdown,
win rate, turnover, number of valid test days, and number of evaluated
observations.
Use `TOPK=50 bash scripts/run_qlib_mdgnn.sh` to change the portfolio size.

Run flat factor baselines to sanity-check Alpha158 and labels:

```bash
MODEL=mlp EPOCHS=5 bash scripts/run_factor_baselines.sh
MODEL=xgboost bash scripts/run_factor_baselines.sh
```

Outputs are written under:

```text
predictions/baselines/
```

Run the standalone MASTER-style reproduction:

```bash
bash scripts/run_master_qlib.sh
```

`configs/master_alpha158_akshare.yaml` uses the same AKShare Alpha158 dataset,
train-fitted time-series feature normalization, robust label normalization, and
portfolio evaluation outputs as the MDGNN workflow.

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
