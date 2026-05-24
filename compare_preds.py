"""Compare two prediction CSVs, treating best as ground truth."""
import argparse
import pandas as pd


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--best", default="Submission_student/test_predictions_best.csv")
    parser.add_argument("--current", default="Submission_student/test_predictions.csv")
    args = parser.parse_args()

    best = pd.read_csv(args.best)
    cur = pd.read_csv(args.current)

    # Merge on Id to ensure alignment
    merged = best.merge(cur, on="Id", suffixes=("_best", "_cur"))
    match = (merged["Prediction_best"] == merged["Prediction_cur"]).sum()
    total = len(merged)
    acc = match / total * 100

    print(f"Total samples: {total}")
    print(f"Matched:       {match}")
    print(f"Agreement:     {acc:.4f}%")

    # Per-task breakdown
    for task in ["Task1", "Task2", "Task3", "Task4"]:
        sub = merged[merged["Id"].str.startswith(task)]
        m = (sub["Prediction_best"] == sub["Prediction_cur"]).sum()
        t = len(sub)
        print(f"  {task}: {m}/{t} = {m/t*100:.2f}%")


if __name__ == "__main__":
    main()
