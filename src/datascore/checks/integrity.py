import pandas as pd
import numpy as np


def check_integrity(df: pd.DataFrame) -> dict:
    # Duplicate rows
    duplicate_count = int(df.duplicated().sum())

    # Constant features
    constant_cols = [col for col in df.columns if df[col].nunique() <= 1]

    # Near constant — one value dominates >99%
    near_constant_cols = [
        col for col in df.columns
        if df[col].nunique() > 1 and df[col].value_counts(normalize=True).iloc[0] > 0.99
    ]

    # Infinite values in numerical columns
    num_df = df.select_dtypes(include=[np.number])
    infinite_count = int(np.isinf(num_df.values).sum())

    return {
        "duplicate_rows": duplicate_count,
        "constant_cols": constant_cols,
        "near_constant_cols": near_constant_cols,
        "infinite_values": infinite_count,
    }