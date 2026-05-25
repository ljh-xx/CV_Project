# Few-shot Image Recognition — DLUT VLG 2026

## 概述

本方案基于 SE-ResNet18 架构，采用"强预训练 + 知识蒸馏 + 原型推理"的纯归纳式管线，解决 16-shot 小样本图像分类任务。

**核心思路**：用公开预训练 DINOv2 作为教师模型生成软标签（仅对验证集），通过知识蒸馏提升学生 backbone 的特征质量；最终推理时丢弃 FC 层，改用原型分类对测试集进行一次性前向推理。

**合规性**：教师模型使用公开预训练权重（规则明确允许），学生模型仅使用官方提供的 DLUT 数据进行训练，测试集从未参与任何训练过程。

---

## 1. 环境配置

```bash
python -m venv .venv
source .venv/bin/activate        # Linux / macOS
# 或 .venv\Scripts\activate      # Windows
pip install -r requirements.txt
```

核心依赖：PyTorch 2.x, torchvision, timm, pandas, thop, tqdm

GPU 推荐，CPU 也可运行但较慢。

---

## 2. 数据集

数据根目录需包含：

```text
data/
├── pretrain/
│   ├── train/        # 200 类，每类 200 张，共 40000 张
│   └── val/          # 200 类，每类 50 张，共 10000 张
├── Task1/
│   ├── train/        # 102 类，每类 16 张
│   ├── val/
│   └── test/
├── Task2/
│   ├── train/        # 100 类，每类 16 张
│   ├── val/
│   └── test/
├── Task3/
│   ├── train/        # 101 类，每类 16 张
│   ├── val/
│   └── test/
└── Task4/
    ├── train/        # 37 类，每类 16 张
    ├── val/
    └── test/
```

---

## 3. 整体管线

```
┌─────────────────────────────────────────────────────────┐
│  Step 1: 骨干网络预训练（官方 pretrain 数据，200 类）         │
│    pretrain.py / main.py  →  results/pretrain/model_best.pth │
├─────────────────────────────────────────────────────────┤
│  Step 2: 教师软标签生成（仅对 val 集）                       │
│    teacher_proto.py  →  Score_teacher_val/Taskx_score.csv   │
├─────────────────────────────────────────────────────────┤
│  Step 3: 知识蒸馏（学生在 val 上模仿教师，test 不参与）        │
│    distill_task_student.py  →  results/distill_val/Taskx/   │
├─────────────────────────────────────────────────────────┤
│  Step 4: 原型推理（蒸馏 backbone + 原型分类 → test 一次性前向） │
│    infer_proto.py  →  Score_student/Taskx_score.csv         │
├─────────────────────────────────────────────────────────┤
│  Step 5: 格式转换                                           │
│    score_to_predictions.py  →  Submission_student/          │
└─────────────────────────────────────────────────────────┘
```

所有训练仅使用官方数据。测试集仅在 Step 4 中参与一次性前向推理（无梯度更新、无标签使用）。

---

## 4. 复现步骤

以下命令默认在 `code/` 目录下执行，以 Windows 为例。Linux/macOS 将路径分隔符改为 `/` 即可。

### Step 1: 骨干网络预训练

使用强数据增广（CutOut + MixUp + CutMix + RandAugment）在 200 类预训练集上训练 SE-ResNet18：

```powershell
python pretrain.py --data-root ./data/pretrain --arch se_resnet18 `
    --epochs 300 --batch-size 64 --lr 0.05 --workers 4 --cuda
```

或使用 demo 中更完整的训练脚本（含 MixUp/CutMix/LabelSmoothing 等）。输出为 `results/pretrain/model_best.pth`。

### Step 2: 教师软标签生成

用冻结的公开 DINOv2 模型提取特征，在验证集上通过原型分类生成软标签。此处教师不使用任何 DLUT 标签进行训练：

```powershell
# 生成教师对 val 集的软标签（供蒸馏使用）
python teacher_proto.py --data-root ./data --task all --split val `
    --write-score --result ./Score_teacher_val --cuda --batch-size 64 --workers 4
```

教师模型从 timm 自动下载（首次运行需联网），之后缓存到本地。

### Step 3: 知识蒸馏

学生模型（SE-ResNet18）以预训练 checkpoint 初始化，在验证集上学习教师的软标签分布。**测试集不参与此步骤**：

```powershell
# 对四个任务分别蒸馏（可并行或逐个运行）
python distill_task_student.py --data ./data/Task1 --dataset Task1 --num-classes 102 `
    --teacher-score ./Score_teacher_val/Task1_score.csv --split val `
    --pretrained ./results/pretrain/model_best.pth --result results/distill_val `
    --epochs 30 --batch-size 64 --query-batch-size 256 --lr 0.01 `
    --soft-weight 1.0 --hard-weight 0.5 --pseudo-hard-weight 1.0 `
    --temperature 1.0 --query-augment none --workers 4 --cuda

python distill_task_student.py --data ./data/Task2 --dataset Task2 --num-classes 100 `
    --teacher-score ./Score_teacher_val/Task2_score.csv --split val `
    --pretrained ./results/pretrain/model_best.pth --result results/distill_val `
    --epochs 30 --batch-size 64 --query-batch-size 256 --lr 0.01 `
    --soft-weight 1.0 --hard-weight 0.5 --pseudo-hard-weight 1.0 `
    --temperature 1.0 --query-augment none --workers 4 --cuda

python distill_task_student.py --data ./data/Task3 --dataset Task3 --num-classes 101 `
    --teacher-score ./Score_teacher_val/Task3_score.csv --split val `
    --pretrained ./results/pretrain/model_best.pth --result results/distill_val `
    --epochs 30 --batch-size 64 --query-batch-size 256 --lr 0.01 `
    --soft-weight 1.0 --hard-weight 0.5 --pseudo-hard-weight 1.0 `
    --temperature 1.0 --query-augment none --workers 4 --cuda

python distill_task_student.py --data ./data/Task4 --dataset Task4 --num-classes 37 `
    --teacher-score ./Score_teacher_val/Task4_score.csv --split val `
    --pretrained ./results/pretrain/model_best.pth --result results/distill_val `
    --epochs 30 --batch-size 64 --query-batch-size 256 --lr 0.01 `
    --soft-weight 1.0 --hard-weight 0.5 --pseudo-hard-weight 1.0 `
    --temperature 1.0 --query-augment none --workers 4 --cuda
```

**损失函数说明**：

| 损失 | 数据来源 | 作用 |
|---|---|---|
| `hard_loss` | downstream train 集真实标签 | 保持分类能力 |
| `distill_loss`（KL 散度） | 教师对 val 集的软标签 | 模仿教师判断分布 |
| `pseudo_loss`（交叉熵） | 教师 argmax 作为伪标签 | 辅助收敛 |

### Step 4: 原型推理

取蒸馏后学生的 backbone（丢弃 FC 层），对测试集进行纯归纳式原型分类：

```powershell
# 一次性对所有四个任务推理
python infer_proto.py --data-root ./data --checkpoint-dir ./results/distill_val `
    --result ./Score_student --task all --cuda --batch-size 256 --transductive-iters 0
```

**原理**：用 downstream train 集（16-shot）通过 backbone 提取特征并构建类原型，测试图片仅做一次前向传播后与原型进行余弦相似度匹配。无梯度更新，无测试标签使用。

### Step 5: 格式转换

```powershell
python score_to_predictions.py --score-dir Score_student --result Submission_student
```

生成 `Submission_student/test_predictions.csv`（Kaggle 提交用）和各 Task 的预测文件。

---

## 5. 最终提交文件

```text
Score_student/
├── Task1_score.csv    # 1793 行 × 102 列
├── Task2_score.csv    # 1649 行 × 100 列
├── Task3_score.csv    # 2020 行 × 101 列
├── Task4_score.csv    # 1850 行 × 37 列
```

格式：首列为 `TaskX_00000` 式的图片索引，后续列为 `score_0` 至 `score_N` 的 softmax 概率，每行求和为 1。

注意：最终提交的是 `Taskx_score.csv`，而非 `test_predictions.csv`。

---

## 6. 模型规格

| 项目 | 数值 | 限制 |
|---|---|---|
| 架构 | SE-ResNet18 | ResNet/ViT 系列 |
| 参数量 | 11.311 M | < 15 M |
| 计算量 | 0.289 GFLOPs | < 2 GFLOPs |
| 图像尺寸 | 84 × 84 | — |

可通过 `model_params_flops.py` 验证：

```bash
python model_params_flops.py
```

---

## 7. 关键脚本说明

| 脚本 | 用途 |
|---|---|
| `pretrain.py` | 在官方 pretrain 数据上训练 200 类 SE-ResNet18 骨干网络 |
| `teacher_proto.py` | 用冻结的公开 DINOv2 + 原型分类生成教师软标签 |
| `distill_task_student.py` | 知识蒸馏：学生模仿教师软标签（仅使用 val 集） |
| `infer_proto.py` | 蒸馏 backbone + 原型分类进行测试集推理 |
| `score_to_predictions.py` | Score CSV → argmax 预测 CSV |
| `model_params_flops.py` | 统计模型参数量和计算量 |
| `compare_preds.py` | 比较两个预测文件的一致性 |

---

## 8. 教师模型说明

教师使用 `vit_small_patch14_dinov2.lvd142m`（从 timm 加载公开预训练权重）。DINOv2 未在 DLUT 数据上训练，仅作为冻结的特征提取器，配合原型分类生成软标签。

蒸馏规则允许教师模型使用公开预训练权重，最终提交的学生模型不包含任何公开预训练权重。

教师仅对验证集生成软标签供学生蒸馏使用，不参与最终推理。

---

## 9. 随机种子

```text
2026
```
