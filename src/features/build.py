"""Top-level feature builder for the SMP hourly target.

Input:  tidy SMP dataframe with columns
        [area, interval_end, smp_krw_per_kwh, demand_forecast_mw]
Output: feature dataframe ready for modelling, plus the target column.

This module deliberately only depends on the SMP+demand data so the MVP
pipeline can run end-to-end before generation/fuel/weather collectors are
wired. Add additional feature joins as those sources land.
"""

from __future__ import annotations

import pandas as pd

from src.features.lag_features import (
    DEFAULT_DEMAND_LAGS,
    DEFAULT_DEMAND_ROLLINGS,
    DEFAULT_SMP_LAGS,
    DEFAULT_SMP_ROLLINGS,
    add_lag_features,
    add_rolling_features,
)
from src.features.time_features import add_time_features
from src.validation.leakage_checks import assert_no_future_leakage


REQUIRED_INPUT_COLUMNS = {"area", "interval_end", "smp_krw_per_kwh", "demand_forecast_mw"}


def build_smp_hourly_features(
    smp_df: pd.DataFrame,
    *,
    forecast_horizon_hours: int = 24,
    target_col: str = "target_smp_t_plus_h",
) -> pd.DataFrame:
    """Build features for a T+`forecast_horizon_hours` SMP target.

    The target at row t is smp at t + horizon. All other features at row t are
    derived from data with interval_end ≤ t, so when the model predicts t+h
    nothing past the prediction-issue time leaks in.
    """
    missing = REQUIRED_INPUT_COLUMNS - set(smp_df.columns)
    if missing:
        raise KeyError(f"Input is missing required columns: {sorted(missing)}")
    if forecast_horizon_hours <= 0:
        raise ValueError("forecast_horizon_hours must be > 0")

    df = smp_df[list(REQUIRED_INPUT_COLUMNS)].copy()
    df["interval_end"] = pd.to_datetime(df["interval_end"])
    df = df.sort_values(["area", "interval_end"]).reset_index(drop=True)

    df = add_lag_features(df, DEFAULT_SMP_LAGS)
    df = add_rolling_features(df, DEFAULT_SMP_ROLLINGS)
    df = add_lag_features(df, DEFAULT_DEMAND_LAGS)
    df = add_rolling_features(df, DEFAULT_DEMAND_ROLLINGS)
    df = add_time_features(df)

    # target = SMP `horizon` hours into the future
    df[target_col] = (
        df.groupby("area", sort=False)["smp_krw_per_kwh"].shift(-forecast_horizon_hours)
    )

    # Drop rows that have NaNs in the feature columns OR the target.
    # Anything pre-warmup (need at least 168 hours of history) is unusable.
    feature_cols = [c for c in df.columns if c not in {"area", "interval_end", target_col}]
    df = df.dropna(subset=feature_cols + [target_col]).reset_index(drop=True)

    assert_no_future_leakage(
        df,
        target_col=target_col,
        timestamp_col="interval_end",
        feature_cols=feature_cols,
        forecast_horizon_hours=forecast_horizon_hours,
    )
    return df
