#!/usr/bin/env bash
set -euo pipefail

CONFIG="${CONFIG:-configs/mdgnn_alpha158_akshare.yaml}"
MODEL="${MODEL:-mlp}"
OUT_DIR="${OUT_DIR:-predictions/baselines}"
TOPK="${TOPK:-20}"
EPOCHS="${EPOCHS:-20}"
BATCH_SIZE="${BATCH_SIZE:-8192}"
MAX_TRAIN_ROWS="${MAX_TRAIN_ROWS:-0}"
PYTHONPATH="${PYTHONPATH:-.}"
export PYTHONPATH

python scripts/run_factor_baselines.py \
  --config "${CONFIG}" \
  --model "${MODEL}" \
  --out-dir "${OUT_DIR}" \
  --topk "${TOPK}" \
  --epochs "${EPOCHS}" \
  --batch-size "${BATCH_SIZE}" \
  --max-train-rows "${MAX_TRAIN_ROWS}"
