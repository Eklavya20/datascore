import argparse

import pandas as pd

from datascore import score


def main():
    parser = argparse.ArgumentParser(description="Score a CSV dataset for ML readiness")
    parser.add_argument("csv_path", help="Path to the CSV file")
    parser.add_argument("--target", help="Target column name")
    parser.add_argument(
        "--task",
        choices=["classification", "regression"],
        help="Optional task override; otherwise inferred from the target",
    )
    args = parser.parse_args()

    df = pd.read_csv(args.csv_path)
    report = score(df, target=args.target, task=args.task)
    report.show()


if __name__ == "__main__":
    main()
