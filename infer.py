import argparse
import json
from pathlib import Path

import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms

from models import model_dict
from utils.test_imagefolder import TestImageFolder


TASKS = {
    "Task1": 102,
    "Task2": 100,
    "Task3": 101,
    "Task4": 37,
}

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def parse_args():
    parser = argparse.ArgumentParser(description="Generate one task score CSV.")
    parser.add_argument("--task", required=True, choices=sorted(TASKS))
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--model-file", default="../模型文件/best_model.pth")
    parser.add_argument("--result", default="../Score")
    parser.add_argument("--arch", default="se_resnet18", choices=sorted(model_dict))
    parser.add_argument("--batch-size", default=256, type=int)
    parser.add_argument("--workers", default=8, type=int)
    parser.add_argument("--cuda", action="store_true")
    return parser.parse_args()


def eval_transform():
    return transforms.Compose(
        [
            transforms.Resize(84),
            transforms.CenterCrop(84),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )


def load_task_state(model_file, task):
    checkpoint = torch.load(model_file, map_location="cpu")
    if "tasks" in checkpoint:
        return checkpoint["tasks"][task]["state_dict"]
    return checkpoint.get("state_dict", checkpoint)


def write_score(task, probs, result_dir):
    result_dir.mkdir(parents=True, exist_ok=True)
    csv_file = result_dir / f"{task}_score.csv"
    index = [f"{task}_{i:05d}" for i in range(probs.size(0))]
    columns = [f"score_{i}" for i in range(probs.size(1))]
    pd.DataFrame(probs.numpy(), index=index, columns=columns).to_csv(csv_file)
    return csv_file


def main():
    args = parse_args()
    use_cuda = args.cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    num_classes = TASKS[args.task]

    model = model_dict[args.arch](num_classes=num_classes)
    model.load_state_dict(load_task_state(args.model_file, args.task))
    model.to(device)
    model.eval()

    test_root = Path(args.data_root) / args.task / "test"
    dataset = TestImageFolder(str(test_root), transform=eval_transform())
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=use_cuda,
    )

    outputs = []
    with torch.no_grad():
        for images, _ in loader:
            images = images.to(device, non_blocking=True)
            outputs.append(F.softmax(model(images), dim=1).cpu())
    csv_file = write_score(args.task, torch.cat(outputs, dim=0), Path(args.result))
    print(json.dumps({"task": args.task, "score_file": str(csv_file)}, sort_keys=True))


if __name__ == "__main__":
    main()
