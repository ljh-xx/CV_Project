#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -lt 1 ]; then
  echo "Usage: bash run_reproduce.sh /path/to/DLUT_VLG_2026_本科生/data [--cuda]"
  exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PACKAGE_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
DATA_ROOT="$1"
CUDA_FLAG="${2:-}"

PYTHON_BIN="${PYTHON_BIN:-python3}"
MODEL_FILE="${PACKAGE_ROOT}/模型文件/best_model.pth"
RESULT_DIR="${PACKAGE_ROOT}/Score"

cd "${SCRIPT_DIR}"
"${PYTHON_BIN}" generate_score.py \
  --data-root "${DATA_ROOT}" \
  --model-file "${MODEL_FILE}" \
  --result "${RESULT_DIR}" \
  --batch-size 256 \
  --workers 8 \
  ${CUDA_FLAG}

echo "Generated score files in ${RESULT_DIR}"
