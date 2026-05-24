#!/usr/bin/env bash
set -euo pipefail

COMMON=(
  --split val
  --pretrained results/se_resnet18_bs_512_lr_0.1/model_best.pth
  --result results/distill_val_hard
  --epochs 120
  --batch-size 256
  --query-batch-size 256
  --lr 0.01
  --soft-weight 1.0
  --hard-weight 0.5
  --pseudo-hard-weight 1.0
  --temperature 1.0
  --query-augment none
  --workers 8
)

.venv/bin/python distill_task_student.py --data data/Task1 --dataset Task1 \
  --num-classes 102 --teacher-score Score_val_mixed/Task1_score.csv "${COMMON[@]}" \
  | tee logs/distill_task1_val_hard_120e.log

.venv/bin/python distill_task_student.py --data data/Task2 --dataset Task2 \
  --num-classes 100 --teacher-score Score_val_mixed/Task2_score.csv "${COMMON[@]}" \
  | tee logs/distill_task2_val_hard_120e.log

.venv/bin/python distill_task_student.py --data data/Task4 --dataset Task4 \
  --num-classes 37 --teacher-score Score_val_mixed/Task4_score.csv "${COMMON[@]}" \
  | tee logs/distill_task4_val_hard_120e.log
