#!/usr/bin/env bash
set -euo pipefail
DATA_ROOT="${1:?Usage: bash scripts/infer_task1.sh /path/to/data [--cuda]}"
CUDA_FLAG="${2:-}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SOURCE_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
PACKAGE_ROOT="$(cd "${SOURCE_DIR}/.." && pwd)"
python3 "${SOURCE_DIR}/infer.py" --task Task1 --data-root "${DATA_ROOT}" --model-file "${PACKAGE_ROOT}/模型文件/best_model.pth" --result "${PACKAGE_ROOT}/Score" ${CUDA_FLAG}
