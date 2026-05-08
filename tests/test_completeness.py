import pandas as pd
from datascore.checks.completeness import check_completeness


def test_no_missing():
    df = pd.DataFrame({
        "a": [1, 2, 3],
        "b": ["x", "y", "z"],
    })
    result = check_completeness(df)
    assert result["total_missing_rate"] == 0.0
    assert result["cols_with_any_missing"] == 0
    assert result["high_missing_cols"] == {}


def test_detects_missing():
    df = pd.DataFrame({
        "a": [1, None, 3],
        "b": ["x", "y", None],
    })
    result = check_completeness(df)
    assert result["total_missing_rate"] > 0
    assert result["cols_with_any_missing"] == 2


def test_high_missing_threshold():
    df = pd.DataFrame({
        "a": [None] * 90 + [1] * 10,
    })
    result = check_completeness(df)
    assert "a" in result["high_missing_cols"]
    assert result["high_missing_cols"]["a"] == 0.9