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


def test_empty_dataframe_raises():
    import pytest
    with pytest.raises(ValueError, match="at least one row and one column"):
        score(pd.DataFrame())


def test_missing_target_raises():
    import pytest
    df = pd.DataFrame({"feature": [1, 2, 3]})
    with pytest.raises(ValueError, match="Target column 'missing' not found"):
        score(df, target="missing")


def test_all_missing_target_raises():
    import pytest
    df = pd.DataFrame({"feature": [1, 2], "target": [None, None]})
    with pytest.raises(ValueError, match="contains no non-missing values"):
        score(df, target="target")


def test_invalid_task_raises():
    import pytest
    df = pd.DataFrame({"feature": [1, 2], "target": [0, 1]})
    with pytest.raises(ValueError, match="task must be"):
        score(df, target="target", task="forecasting")


def test_task_without_target_raises():
    import pytest
    df = pd.DataFrame({"feature": [1, 2]})
    with pytest.raises(ValueError, match="only be specified when target is provided"):
        score(df, task="regression")


def test_regression_target_skips_class_balance():
    df = pd.DataFrame({
        "feature": range(30),
        "target": [value / 10 for value in range(30)],
    })
    report = score(df, target="target")
    readiness = report.raw["ml_readiness"]
    assert readiness["task"] == "regression"
    assert readiness["class_balance_minority"] is None
    assert readiness["class_imbalanced"] is None


def test_task_override_supports_integer_regression_target():
    df = pd.DataFrame({
        "feature": range(10),
        "target": range(10),
    })
    report = score(df, target="target", task="regression")
    assert report.raw["ml_readiness"]["task"] == "regression"
