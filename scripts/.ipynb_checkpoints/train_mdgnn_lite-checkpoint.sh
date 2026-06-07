#!/usr/bin/env bash
INSTRUMENTS="${INSTRUMENTS:-csi300}"
START="${START:-2018-01-01}"
END="${END:-2023-12-31}"
START_COMPACT="${START//-/}"
END_COMPACT="${END//-/}"

FEATURES="${FEATURES:-data/alpha158/features_alpha158_${INSTRUMENTS}_${START_COMPACT}_${END_COMPACT}.parquet}"
LABELS="${LABELS:-data/alpha158/labels_alpha158_${INSTRUMENTS}_${START_COMPACT}_${END_COMPACT}.parquet}"
RELATIONS="${RELATIONS:-}"
WINDOW="${WINDOW:-10}"
BATCH_SIZE="${BATCH_SIZE:-8}"
EPOCHS="${EPOCHS:-20}"
LR="${LR:-0.001}"
HIDDEN_DIM="${HIDDEN_DIM:-128}"
OUT="${OUT:-checkpoints/mdgnn_lite.pt}"
DEBUG="${DEBUG:-0}"
DEBUG_BATCHES="${DEBUG_BATCHES:-2}"

cmd=(
  python -m src.mdgnn_lite.train
  --features "${FEATURES}"
  --labels "${LABELS}"
  --window "${WINDOW}"
  --batch-size "${BATCH_SIZE}"
  --epochs "${EPOCHS}"
  --lr "${LR}"
  --hidden-dim "${HIDDEN_DIM}"
  --out "${OUT}"
)

if [[ -n "${RELATIONS}" ]]; then
  cmd+=(--relations "${RELATIONS}")
fi

if [[ "${DEBUG}" == "1" ]]; then
  cmd+=(--debug --debug-batches "${DEBUG_BATCHES}")
fi

"${cmd[@]}"
