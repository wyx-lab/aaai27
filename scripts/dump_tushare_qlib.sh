#!/usr/bin/env bash
set -euo pipefail

START="${START:-20180101}"
END="${END:-20251231}"
RAW_DIR="${RAW_DIR:-data/tushare_raw}"
CSV_DIR="${CSV_DIR:-data/tushare_qlib_csv}"
QLIB_DIR="${QLIB_DIR:-$HOME/.qlib/qlib_data/tushare_cn_data}"
QLIB_REPO_DIR="${QLIB_REPO_DIR:-/tmp/qlib}"
LIMIT="${LIMIT:-}"

download_cmd=(
  python scripts/download_tushare_daily.py
  --start "${START}"
  --end "${END}"
  --raw-dir "${RAW_DIR}"
  --csv-dir "${CSV_DIR}"
)

if [[ -n "${LIMIT}" ]]; then
  download_cmd+=(--limit "${LIMIT}")
fi

"${download_cmd[@]}"

if [[ ! -d "${QLIB_REPO_DIR}/.git" ]]; then
  git clone https://github.com/microsoft/qlib.git "${QLIB_REPO_DIR}"
fi

python "${QLIB_REPO_DIR}/scripts/dump_bin.py" dump_all \
  --csv_path "${CSV_DIR}" \
  --qlib_dir "${QLIB_DIR}" \
  --include_fields open,high,low,close,volume,factor

echo "qlib_dir=${QLIB_DIR}"
