import pandas as pd
import numpy as np
from scipy import stats


def check_distribution(df: pd.DataFrame, target: str | None = None) -> dict:
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    skewed_cols = {}
    outlier_cols = {}
    leakage_risk = []

    target_series = None
    if target is not None and target in df.columns:
        if pd.api.types.is_numeric_dtype(df[target]):
            target_series = df[target]

    for col in num_cols:
        series = df[col].dropna()

        if series.nunique() <= 1:
            continue

        # Skew
        skew = float(stats.skew(series, nan_policy="omit"))
        if abs(skew) > 1.0:
            skewed_cols[col] = round(skew, 4)

        # Outliers via IQR
        q1 = series.quantile(0.25)
        q3 = series.quantile(0.75)
        iqr = q3 - q1
        outliers = int(((series < q1 - 1.5 * iqr) | (series > q3 + 1.5 * iqr)).sum())
        if outliers > 0:
            outlier_cols[col] = outliers

        # Leakage risk
        if target_series is not None and col != target:
            aligned = df[[col, target]].dropna()
            if aligned[col].nunique() > 1 and aligned[target].nunique() > 1:
                corr = abs(aligned[col].corr(aligned[target]))
                if pd.notna(corr) and corr > 0.9:
                    leakage_risk.append({
                        "feature": col,
                        "correlation": round(float(corr), 4)
                    })

    return {
        "skewed_cols": skewed_cols,
        "outlier_cols": outlier_cols,
        "leakage_risk": leakage_risk,
    }