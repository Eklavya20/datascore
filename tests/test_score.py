import pandas as pd
from datascore import score


def test_score_returns_report():
    df = pd.DataFrame({
        "a": [1, 2, 3],
        "b": ["x", "y", "z"],
        "target": [0, 1, 0],
    })
    report = score(df, target="target")
    assert report.score >= 0
    assert report.score <= 100
    assert report.verdict in ["READY", "NEEDS WORK", "NOT READY"]


def test_score_without_target():
    df = pd.DataFrame({
        "a": [1, 2, 3],
        "b": ["x", "y", "z"],
    })
    report = score(df)
    assert report.score >= 0


def test_score_penalises_missing():
    df = pd.DataFrame({
        "a": [None] * 50 + [1] * 50,
        "b": [1] * 100,
        "target": [0, 1] * 50,
    })
    clean_df = pd.DataFrame({
        "a": list(range(100)),
        "b": list(range(100)),
        "target": [0, 1] * 50,
    })
    dirty_report = score(df, target="target")
    clean_report = score(clean_df, target="target")
    assert dirty_report.score < clean_report.score


def test_type_error_on_non_dataframe():
    import pytest
    with pytest.raises(TypeError):
        score([1, 2, 3], target="a")