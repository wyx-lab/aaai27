#!/usr/bin/env bash
set -euo pipefail

CONFIG="${CONFIG:-configs/mdgnn_alpha158.yaml}"
PYTHONPATH="${PYTHONPATH:-.}"
export PYTHONPATH

qrun "${CONFIG}"
