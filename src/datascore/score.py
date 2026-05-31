import pandas as pd
from datascore.checks.completeness import check_completeness
from datascore.checks.integrity import check_integrity
from datascore.checks.ml_readiness import check_ml_readiness
from datascore.checks.distribution import check_distribution
from datascore.reporter import build_report, Report


def score(df: pd.DataFrame, target: str = None) -> "Report":
    if not isinstance(df, pd.DataFrame):
        raise TypeError("Input must be a pandas DataFrame")
    
    if target is None:
        print("Warning: no target specified. ML readiness and leakage checks skipped.")

    results = {
        "shape": df.shape,
        "target": target,
        "completeness": check_completeness(df),
        "integrity": check_integrity(df),
        "ml_readiness": check_ml_readiness(df, target) if target else {},
        "distribution": check_distribution(df, target=target),
    }

    return build_report(results)