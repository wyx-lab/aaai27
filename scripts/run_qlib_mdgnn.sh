#!/usr/bin/env bash
set -euo pipefail

CONFIG="${CONFIG:-configs/mdgnn_alpha158_akshare.yaml}"
OUT="${OUT:-predictions/mdgnn_scores.parquet}"
METRICS_OUT="${METRICS_OUT:-predictions/mdgnn_metrics.json}"
DAILY_OUT="${DAILY_OUT:-predictions/mdgnn_daily_metrics.parquet}"
TOPK="${TOPK:-20}"
PYTHONPATH="${PYTHONPATH:-.}"
export PYTHONPATH

python scripts/run_qlib_fit.py \
  --config "${CONFIG}" \
  --out "${OUT}" \
  --metrics-out "${METRICS_OUT}" \
  --daily-out "${DAILY_OUT}" \
  --topk "${TOPK}"
