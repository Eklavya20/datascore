import pandas as pd


def check_completeness(df: pd.DataFrame) -> dict:
    total_cells = df.shape[0] * df.shape[1]
    missing_per_col = df.isnull().sum()
    missing_rate_per_col = df.isnull().mean()

    high_missing = {
        col: round(float(rate), 4)
        for col, rate in missing_rate_per_col.items()
        if rate > 0.05
    }

    return {
        "total_missing_rate": round(float(df.isnull().sum().sum() / total_cells), 4),
        "cols_with_any_missing": int((missing_per_col > 0).sum()),
        "high_missing_cols": high_missing,  # cols with >5% missing
    }