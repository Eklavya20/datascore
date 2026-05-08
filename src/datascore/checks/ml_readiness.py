import pandas as pd
import numpy as np


def check_ml_readiness(df: pd.DataFrame, target: str) -> dict:
    results = {}

    # Class imbalance
    if target in df.columns:
        counts = df[target].value_counts(normalize=True)
        minority = round(float(counts.min()), 4)
        results["class_balance_minority"] = minority
        results["class_imbalanced"] = minority < 0.15
    else:
        results["class_balance_minority"] = None
        results["class_imbalanced"] = None

    # High cardinality categoricals
    cat_cols = df.select_dtypes(include=["str", "category"]).columns.tolist()
    if target in cat_cols:
        cat_cols.remove(target)

    high_cardinality = {
        col: int(df[col].nunique())
        for col in cat_cols
        if df[col].nunique() > 20
    }
    results["high_cardinality_cols"] = high_cardinality

    return results