from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.features.build import build_smp_hourly_features
from src.features.lag_features import (
    LagSpec,
    RollingSpec,
    add_lag_features,
    add_rolling_features,
)
from src.validation.leakage_checks import assert_no_future_leakage


def test_lag_feature_matches_shifted_value(synthetic_smp_dataframe):
    df = add_lag_features(
        synthetic_smp_dataframe,
        [LagSpec("smp_krw_per_kwh", 24, "smp_lag_24h")],
        group_col="area",
    )
    # any row after index 24 should have lag value equal to smp 24 rows earlier
    by_area = df.sort_values(["area", "interval_end"]).reset_index(drop=True)
    pair = by_area[["smp_krw_per_kwh", "smp_lag_24h"]].iloc[30:60]
    expected = by_area["smp_krw_per_kwh"].iloc[6:36].to_numpy()
    np.testing.assert_allclose(pair["smp_lag_24h"].to_numpy(), expected)


def test_lag_zero_raises():
    df = pd.DataFrame({"area": ["x"], "interval_end": [pd.Timestamp("2024-01-01")], "v": [1.0]})
    with pytest.raises(ValueError, match="no-future-leakage"):
        add_lag_features(df, [LagSpec("v", 0)], group_col="area")


def test_rolling_excludes_current_hour(synthetic_smp_dataframe):
    df = add_rolling_features(
        synthetic_smp_dataframe,
        [RollingSpec("smp_krw_per_kwh", 24, "mean", "smp_rolling_24h_mean")],
        group_col="area",
    )
    by_area = df.sort_values(["area", "interval_end"]).reset_index(drop=True)
    # row at index 24 has rolling mean of rows 0..23 (24 values, all in the past)
    expected_mean = synthetic_smp_dataframe.sort_values("interval_end")["smp_krw_per_kwh"].iloc[0:24].mean()
    assert by_area["smp_rolling_24h_mean"].iloc[24] == pytest.approx(expected_mean)


def test_build_smp_hourly_features_passes_leakage_check(synthetic_smp_dataframe):
    feat = build_smp_hourly_features(synthetic_smp_dataframe, forecast_horizon_hours=24)
    feature_cols = [c for c in feat.columns if c not in {"area", "interval_end", "target_smp_t_plus_h"}]
    # no exception means the leakage check passed
    assert_no_future_leakage(
        feat,
        target_col="target_smp_t_plus_h",
        timestamp_col="interval_end",
        feature_cols=feature_cols,
        forecast_horizon_hours=24,
    )


def test_leakage_detected_when_feature_equals_future_target(synthetic_smp_dataframe):
    feat = build_smp_hourly_features(synthetic_smp_dataframe, forecast_horizon_hours=24)
    feat["leaky_future_smp"] = feat["target_smp_t_plus_h"]
    with pytest.raises(AssertionError, match="leakage"):
        assert_no_future_leakage(
            feat,
            target_col="target_smp_t_plus_h",
            timestamp_col="interval_end",
            feature_cols=["leaky_future_smp"],
            forecast_horizon_hours=24,
        )
