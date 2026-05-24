import argparse
from pathlib import Path

import pandas as pd


TASKS = ("Task1", "Task2", "Task3", "Task4")


def parse_args():
    parser = argparse.ArgumentParser(description="Convert score CSVs to argmax predictions.")
    parser.add_argument("--score-dir", default="Score")
    parser.add_argument("--result", default="Submission")
    return parser.parse_args()


def main():
    args = parse_args()
    score_dir = Path(args.score_dir)
    result_dir = Path(args.result)
    result_dir.mkdir(parents=True, exist_ok=True)

    frames = []
    for task in TASKS:
        score_path = score_dir / f"{task}_score.csv"
        scores = pd.read_csv(score_path, index_col=0)
        pred = scores.to_numpy().argmax(axis=1)
        frame = pd.DataFrame({"Id": scores.index, "Prediction": pred})
        frame.to_csv(result_dir / f"{task}_predictions.csv", index=False)
        frames.append(frame)

    pd.concat(frames, ignore_index=True).to_csv(
        result_dir / "test_predictions.csv", index=False
    )


if __name__ == "__main__":
    main()
