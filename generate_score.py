import argparse
import subprocess
import sys
from pathlib import Path


TASKS = ("Task1", "Task2", "Task3", "Task4")


def parse_args():
    parser = argparse.ArgumentParser(description="Generate all Taskx_score.csv files.")
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--model-file", default="../模型文件/best_model.pth")
    parser.add_argument("--result", default="../Score")
    parser.add_argument("--batch-size", default=256, type=int)
    parser.add_argument("--workers", default=8, type=int)
    parser.add_argument("--cuda", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    script = Path(__file__).resolve().with_name("infer.py")
    for task in TASKS:
        cmd = [
            sys.executable,
            str(script),
            "--task",
            task,
            "--data-root",
            args.data_root,
            "--model-file",
            args.model_file,
            "--result",
            args.result,
            "--batch-size",
            str(args.batch_size),
            "--workers",
            str(args.workers),
        ]
        if args.cuda:
            cmd.append("--cuda")
        subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
