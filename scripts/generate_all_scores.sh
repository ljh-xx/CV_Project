#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -lt 1 ]; then
  echo "Usage: bash scripts/generate_all_scores.sh /path/to/DLUT_VLG_2026_本科生/data [--cuda]"
  exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SOURCE_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
bash "${SOURCE_DIR}/run_reproduce.sh" "$@"
