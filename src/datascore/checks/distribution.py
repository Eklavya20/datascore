import pandas as pd
import numpy as np
from scipy import stats


def check_distribution(df: pd.DataFrame) -> dict:
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    skewed_cols = {}
    outlier_cols = {}

    for col in num_cols:
        series = df[col].dropna()

        # Skew
        skew = float(stats.skew(series))
        if abs(skew) > 1.0:
            skewed_cols[col] = round(skew, 4)

        # Outliers via IQR
        q1 = series.quantile(0.25)
        q3 = series.quantile(0.75)
        iqr = q3 - q1
        outliers = int(((series < q1 - 1.5 * iqr) | (series > q3 + 1.5 * iqr)).sum())
        if outliers > 0:
            outlier_cols[col] = outliers

    return {
        "skewed_cols": skewed_cols,
        "outlier_cols": outlier_cols,
    }