#!/usr/bin/env bash
set -euo pipefail

mkdir -p Score_val_mixed logs

LARGE="vit_large_patch14_reg4_dinov2.lvd142m"
BASE="vit_base_patch14_reg4_dinov2.lvd142m"
COMMON=(--split val --result Score_val_mixed --write-score --workers 8)

.venv/bin/python teacher_proto.py --task Task1 --teacher "${LARGE}" "${COMMON[@]}" \
  --batch-size 16 --center none --score-scale 5 --transductive-iters 1 --query-weight 0.25 \
  | tee logs/final_mixed_val_task1.log

.venv/bin/python teacher_proto.py --task Task2 --teacher "${BASE}" "${COMMON[@]}" \
  --batch-size 32 --center support --score-scale 30 --transductive-iters 1 --query-weight 0.5 \
  | tee logs/final_mixed_val_task2.log

.venv/bin/python teacher_proto.py --task Task3 --teacher "${LARGE}" "${COMMON[@]}" \
  --batch-size 16 --center none --score-scale 30 --transductive-iters 1 --query-weight 0.1 \
  | tee logs/final_mixed_val_task3.log

.venv/bin/python teacher_proto.py --task Task4 --teacher "${LARGE}" "${COMMON[@]}" \
  --batch-size 16 --center support --score-scale 5 --transductive-iters 1 --query-weight 1.0 \
  | tee logs/final_mixed_val_task4.log
