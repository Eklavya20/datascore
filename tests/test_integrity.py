import pandas as pd
import numpy as np
from datascore.checks.integrity import check_integrity


def test_no_issues():
    df = pd.DataFrame({
        "a": [1, 2, 3],
        "b": ["x", "y", "z"],
    })
    result = check_integrity(df)
    assert result["duplicate_rows"] == 0
    assert result["constant_cols"] == []
    assert result["infinite_values"] == 0


def test_detects_duplicates():
    df = pd.DataFrame({
        "a": [1, 1, 2],
        "b": ["x", "x", "y"],
    })
    result = check_integrity(df)
    assert result["duplicate_rows"] == 1


def test_detects_constant_col():
    df = pd.DataFrame({
        "a": [1, 1, 1],
        "b": [1, 2, 3],
    })
    result = check_integrity(df)
    assert "a" in result["constant_cols"]


def test_detects_infinite():
    df = pd.DataFrame({
        "a": [1.0, np.inf, 3.0],
    })
    result = check_integrity(df)
    assert result["infinite_values"] == 1