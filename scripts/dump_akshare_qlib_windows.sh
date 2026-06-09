#!/usr/bin/env bash
set -euo pipefail

# Run from Git Bash / MSYS2 / WSL in the repository root.
# Use SKIP_DUMP=1 when you only want to download CSVs on Windows.

START="${START:-20180101}"
END="${END:-20251231}"
RAW_DIR="${RAW_DIR:-data/akshare_raw}"
CSV_DIR="${CSV_DIR:-data/akshare_qlib_csv}"
QLIB_DIR="${QLIB_DIR:-data/akshare_cn_data}"
QLIB_REPO_DIR="${QLIB_REPO_DIR:-.cache/qlib}"
ADJUST="${ADJUST:-qfq}"
LIMIT="${LIMIT:-}"
SLEEP="${SLEEP:-1.5}"
RETRIES="${RETRIES:-5}"
RETRY_SLEEP="${RETRY_SLEEP:-8.0}"
PYTHON="${PYTHON:-python}"
SKIP_DUMP="${SKIP_DUMP:-0}"

download_cmd=(
  "${PYTHON}" scripts/download_akshare_daily.py
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

if [[ "${SKIP_DUMP}" == "1" ]]; then
  echo "skip dump_bin; csv_dir=${CSV_DIR}"
  exit 0
fi

if [[ ! -d "${QLIB_REPO_DIR}/.git" ]]; then
  git clone https://github.com/microsoft/qlib.git "${QLIB_REPO_DIR}"
fi

"${PYTHON}" "${QLIB_REPO_DIR}/scripts/dump_bin.py" dump_all \
  --data_path "${CSV_DIR}" \
  --qlib_dir "${QLIB_DIR}" \
  --date_field_name date \
  --symbol_field_name symbol \
  --include_fields open,high,low,close,volume,factor

echo "qlib_dir=${QLIB_DIR}"
