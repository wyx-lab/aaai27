#!/usr/bin/env bash
set -euo pipefail

CONFIG="${CONFIG:-configs/mdgnn_alpha158_akshare.yaml}"
SEGMENT="${SEGMENT:-train}"
DATA_KEY="${DATA_KEY:-learn}"
OUT="${OUT:-reports/alpha158_feature_ranges.csv}"
PYTHONPATH="${PYTHONPATH:-.}"
export PYTHONPATH

python scripts/summarize_alpha158_ranges.py \
  --config "${CONFIG}" \
  --segment "${SEGMENT}" \
  --data-key "${DATA_KEY}" \
  --out "${OUT}"
