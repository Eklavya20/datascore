import pandas as pd

from datascore.checks.ml_readiness import check_ml_readiness


def test_object_categorical_high_cardinality_is_detected():
    df = pd.DataFrame({
        "category": [f"value-{i}" for i in range(25)],
        "target": [0, 1] * 12 + [0],
    })
    result = check_ml_readiness(df, "target")
    assert result["high_cardinality_cols"] == {"category": 25}


def test_continuous_target_is_inferred_as_regression():
    df = pd.DataFrame({
        "feature": range(30),
        "target": [value / 10 for value in range(30)],
    })
    result = check_ml_readiness(df, "target")
    assert result["task"] == "regression"
    assert result["class_balance_minority"] is None
