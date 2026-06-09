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

download_cmd=(
  python scripts/download_akshare_daily.py
  --start "${START}"
  --end "${END}"
  --raw-dir "${RAW_DIR}"
  --csv-dir "${CSV_DIR}"
  --adjust "${ADJUST}"
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
