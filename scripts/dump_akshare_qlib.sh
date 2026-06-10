#!/usr/bin/env bash
set -euo pipefail

START="${START:-20180101}"
END="${END:-20251231}"
RAW_DIR="${RAW_DIR:-data/akshare_raw}"
CSV_DIR="${CSV_DIR:-data/akshare_qlib_csv}"
QLIB_DIR="${QLIB_DIR:-$HOME/.qlib/qlib_data/akshare_cn_data}"
QLIB_REPO_DIR="${QLIB_REPO_DIR:-/tmp/qlib}"
ADJUST="${ADJUST:-qfq}"
LIMIT="${LIMIT:-}"
SLEEP="${SLEEP:-0.5}"
RETRIES="${RETRIES:-3}"
RETRY_SLEEP="${RETRY_SLEEP:-3.0}"

download_cmd=(
  python scripts/download_akshare_daily.py
  --start "${START}"
  --end "${END}"
  --raw-dir "${RAW_DIR}"
  --csv-dir "${CSV_DIR}"
  --adjust "${ADJUST}"
  --sleep "${SLEEP}"
  --retries "${RETRIES}"
  --retry-sleep "${RETRY_SLEEP}"
)

if [[ -n "${LIMIT}" ]]; then
  download_cmd+=(--limit "${LIMIT}")
fi

"${download_cmd[@]}"

python scripts/write_akshare_instruments.py \
  --csv-dir "${CSV_DIR}" \
  --start "${START}" \
  --end "${END}"

if [[ ! -d "${QLIB_REPO_DIR}/.git" ]]; then
  git clone https://github.com/microsoft/qlib.git "${QLIB_REPO_DIR}"
fi

python "${QLIB_REPO_DIR}/scripts/dump_bin.py" dump_all \
  --data_path "${CSV_DIR}" \
  --qlib_dir "${QLIB_DIR}" \
  --date_field_name date \
  --symbol_field_name symbol \
  --include_fields open,high,low,close,volume,factor

echo "qlib_dir=${QLIB_DIR}"
