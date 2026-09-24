import pandas as pd

from datascore.checks.distribution import check_distribution


def test_target_is_not_reported_as_feature_skew_or_outlier():
    df = pd.DataFrame({
        "feature": range(100),
        "target": [0] * 99 + [100],
    })
    result = check_distribution(df, target="target")
    assert "target" not in result["skewed_cols"]
    assert "target" not in result["outlier_cols"]
