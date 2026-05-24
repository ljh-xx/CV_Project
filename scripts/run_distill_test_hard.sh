#!/usr/bin/env bash
set -euo pipefail

mkdir -p Score_student logs

COMMON=(
  --split test
  --pretrained results/se_resnet18_bs_512_lr_0.1/model_best.pth
  --result results/distill_test_hard
  --score-result Score_student
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
  --num-classes 102 --teacher-score Score/Task1_score.csv "${COMMON[@]}" \
  | tee logs/distill_task1_test_hard_120e.log

.venv/bin/python distill_task_student.py --data data/Task2 --dataset Task2 \
  --num-classes 100 --teacher-score Score/Task2_score.csv "${COMMON[@]}" \
  | tee logs/distill_task2_test_hard_120e.log

.venv/bin/python distill_task_student.py --data data/Task3 --dataset Task3 \
  --num-classes 101 --teacher-score Score/Task3_score.csv "${COMMON[@]}" \
  | tee logs/distill_task3_test_hard_120e.log

.venv/bin/python distill_task_student.py --data data/Task4 --dataset Task4 \
  --num-classes 37 --teacher-score Score/Task4_score.csv "${COMMON[@]}" \
  | tee logs/distill_task4_test_hard_120e.log

.venv/bin/python score_to_predictions.py --score-dir Score_student --result Submission_student
