@echo off
chcp 65001 >nul
echo ==============================================
echo  自动蒸馏脚本 - Task2/Task3/Task4
echo  参数完全保留，按顺序自动执行
echo ==============================================
echo.

echo [1/3] 开始运行 Task2 蒸馏训练...
echo.
python distill_task_student.py --data ./data/Task2 --dataset Task2 --num-classes 100 --teacher-score ./Score_teacher_train_kfold/Task2_score.csv --split train --pretrained ./results/pretrain/model_best.pth --result results/distill --epochs 80 --batch-size 64 --query-batch-size 256 --lr 0.01 --soft-weight 1.0 --hard-weight 0.5 --pseudo-hard-weight 1.0 --temperature 1.0 --query-augment none --workers 4 --cuda
echo.
echo ✅ Task2 蒸馏训练完成！
echo.

echo [2/3] 开始运行 Task3 蒸馏训练...
echo.
python distill_task_student.py --data ./data/Task3 --dataset Task3 --num-classes 101 --teacher-score ./Score_teacher_train_kfold/Task3_score.csv --split train --pretrained ./results/pretrain/model_best.pth --result results/distill --epochs 80 --batch-size 64 --query-batch-size 256 --lr 0.01 --soft-weight 1.0 --hard-weight 0.5 --pseudo-hard-weight 1.0 --temperature 1.0 --query-augment none --workers 4 --cuda
echo.
echo ✅ Task3 蒸馏训练完成！
echo.

echo [3/3] 开始运行 Task4 蒸馏训练...
echo.
python distill_task_student.py --data ./data/Task4 --dataset Task4 --num-classes 37 --teacher-score ./Score_teacher_train_kfold/Task4_score.csv --split train --pretrained ./results/pretrain/model_best.pth --result results/distill --epochs 80 --batch-size 64 --query-batch-size 256 --lr 0.01 --soft-weight 1.0 --hard-weight 0.5 --pseudo-hard-weight 1.0 --temperature 1.0 --query-augment none --workers 4 --cuda
echo.
echo ✅ Task4 蒸馏训练完成！
echo.

echo ==============================================
echo  🎉 所有任务蒸馏训练全部完成！
echo  模型已保存到 ./results/distill/ 目录
echo ==============================================
pause