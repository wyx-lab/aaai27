#!/usr/bin/env bash
set -euo pipefail

PROVIDER_URI="${PROVIDER_URI:-$HOME/.qlib/qlib_data/cn_data}"
INSTRUMENTS="${INSTRUMENTS:-csi300}"
START="${START:-2018-01-01}"
END="${END:-2023-12-31}"
FIT_START="${FIT_START:-2018-01-01}"
FIT_END="${FIT_END:-2019-12-31}"
OUT_DIR="${OUT_DIR:-data/alpha158}"
QLIB_REPO_DIR="${QLIB_REPO_DIR:-/tmp/qlib}"
SKIP_DOWNLOAD="${SKIP_DOWNLOAD:-0}"

if [[ "${SKIP_DOWNLOAD}" != "1" && ! -d "${PROVIDER_URI}" ]]; then
  if [[ ! -d "${QLIB_REPO_DIR}/.git" ]]; then
    git clone https://github.com/microsoft/qlib.git "${QLIB_REPO_DIR}"
  fi

  python "${QLIB_REPO_DIR}/scripts/get_data.py" qlib_data \
    --target_dir "${PROVIDER_URI}" \
    --region cn
fi

python scripts/export_alpha158_qlib.py \
  --provider-uri "${PROVIDER_URI}" \
  --instruments "${INSTRUMENTS}" \
  --start "${START}" \
  --end "${END}" \
  --fit-start "${FIT_START}" \
  --fit-end "${FIT_END}" \
  --out-dir "${OUT_DIR}"

