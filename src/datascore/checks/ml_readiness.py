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

    # Target leakage — numerical features correlated >0.9 with target
    leakage_risk = []
    if target in df.columns:
        num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if target in num_cols:
            num_cols.remove(target)
        target_series = pd.to_numeric(df[target], errors="coerce")
        for col in num_cols:
            if col == target:
                continue
            corr = abs(df[col].corr(target_series))
            if corr > 0.9:
                leakage_risk.append({"feature": col, "correlation": round(float(corr), 4)})

    results["leakage_risk_cols"] = leakage_risk

    # High cardinality categoricals
    cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
    if target in cat_cols:
        cat_cols.remove(target)

    high_cardinality = {
        col: int(df[col].nunique())
        for col in cat_cols
        if df[col].nunique() > 20
    }
    results["high_cardinality_cols"] = high_cardinality

    return results