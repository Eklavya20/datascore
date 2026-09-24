import pandas as pd

from datascore.checks.completeness import check_completeness
from datascore.checks.distribution import check_distribution
from datascore.checks.integrity import check_integrity
from datascore.checks.ml_readiness import check_ml_readiness
from datascore.reporter import Report, build_report


def score(
    df: pd.DataFrame,
    target: str | None = None,
    task: str | None = None,
) -> "Report":
    if not isinstance(df, pd.DataFrame):
        raise TypeError("Input must be a pandas DataFrame")

    if df.empty:
        raise ValueError("Input DataFrame must contain at least one row and one column")

    if task not in (None, "classification", "regression"):
        raise ValueError("task must be 'classification', 'regression', or None")

    if target is None:
        if task is not None:
            raise ValueError("task can only be specified when target is provided")
        print("Warning: no target specified. ML readiness and leakage checks skipped.")
    elif target not in df.columns:
        raise ValueError(f"Target column '{target}' not found in DataFrame")
    elif df[target].dropna().empty:
        raise ValueError(f"Target column '{target}' contains no non-missing values")

    results = {
        "shape": df.shape,
        "target": target,
        "completeness": check_completeness(df),
        "integrity": check_integrity(df),
        "ml_readiness": check_ml_readiness(df, target, task=task) if target else {},
        "distribution": check_distribution(df, target=target),
    }

    return build_report(results)
