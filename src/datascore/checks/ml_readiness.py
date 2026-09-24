import pandas as pd


VALID_TASKS = {"classification", "regression"}


def infer_task(target: pd.Series) -> str:
    """Infer classification vs regression from a target series."""
    values = target.dropna()
    if values.empty:
        raise ValueError("Target contains no non-missing values")

    dtype = values.dtype
    if (
        pd.api.types.is_bool_dtype(dtype)
        or isinstance(dtype, pd.CategoricalDtype)
        or pd.api.types.is_object_dtype(dtype)
        or pd.api.types.is_string_dtype(dtype)
        or values.nunique() <= 20
    ):
        return "classification"
    return "regression"


def check_ml_readiness(
    df: pd.DataFrame,
    target: str,
    task: str | None = None,
) -> dict:
    if target not in df.columns:
        raise ValueError(f"Target column '{target}' not found in DataFrame")

    if task is not None and task not in VALID_TASKS:
        raise ValueError("task must be 'classification', 'regression', or None")

    resolved_task = task or infer_task(df[target])
    results = {"task": resolved_task}

    # Class imbalance
    if resolved_task == "classification":
        counts = df[target].value_counts(normalize=True)
        minority = round(float(counts.min()), 4)
        results["class_balance_minority"] = minority
        results["class_imbalanced"] = minority < 0.15
    else:
        results["class_balance_minority"] = None
        results["class_imbalanced"] = None

    # High cardinality categoricals
    cat_cols = [
        col
        for col in df.columns
        if (
            pd.api.types.is_object_dtype(df[col].dtype)
            or pd.api.types.is_string_dtype(df[col].dtype)
            or isinstance(df[col].dtype, pd.CategoricalDtype)
        )
    ]
    if target in cat_cols:
        cat_cols.remove(target)

    high_cardinality = {
        col: int(df[col].nunique())
        for col in cat_cols
        if df[col].nunique() > 20
    }
    results["high_cardinality_cols"] = high_cardinality

    return results
