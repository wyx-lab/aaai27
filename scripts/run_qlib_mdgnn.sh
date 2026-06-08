#!/usr/bin/env bash
set -euo pipefail

CONFIG="${CONFIG:-configs/mdgnn_alpha158.yaml}"
OUT="${OUT:-predictions/mdgnn_scores.parquet}"
PYTHONPATH="${PYTHONPATH:-.}"
export PYTHONPATH

python scripts/run_qlib_fit.py --config "${CONFIG}" --out "${OUT}"
