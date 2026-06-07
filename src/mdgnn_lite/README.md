# MDGNN-lite with Qlib Alpha158

This module expects Alpha158 features exported by `scripts/export_alpha158_qlib.py`.

## Inputs

Feature parquet:

- pandas DataFrame
- MultiIndex: `datetime`, `instrument`
- columns: Alpha158 feature columns

Label parquet:

- pandas DataFrame
- MultiIndex: `datetime`, `instrument`
- first column is used as the regression target by default

Relation graph:

- optional
- `.npy` / `.npz`: `[num_relations, num_stocks, num_stocks]` or `[num_stocks, num_stocks]`
- `.csv`: edge list with `src,dst,relation,weight`

If no relation graph is passed, the trainer uses identity edges. That is useful for
checking the data pipeline before adding industry or correlation relations.

## Train

```bash
python -m src.mdgnn_lite.train \
  --features data/alpha158/features_alpha158_csi300_20180101_20231231.parquet \
  --labels data/alpha158/labels_alpha158_csi300_20180101_20231231.parquet \
  --window 10 \
  --batch-size 8 \
  --epochs 20 \
  --out checkpoints/mdgnn_lite.pt
```

With a relation graph:

```bash
python -m src.mdgnn_lite.train \
  --features data/alpha158/features_alpha158_csi300_20180101_20231231.parquet \
  --labels data/alpha158/labels_alpha158_csi300_20180101_20231231.parquet \
  --relations data/relations/csi300_relations.npy
```

## Model

`MDGNNLite` is a practical approximation of the paper's idea:

1. GRU temporal encoder over Alpha158 windows.
2. Multi-relation graph convolution over stocks.
3. Transformer encoder for cross-stock context.
4. Regression head outputs a score for each stock/date.

Use the scores for IC/RankIC and Top-K portfolio backtests.
