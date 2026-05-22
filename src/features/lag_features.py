"""Lag / rolling features with a strict no-future-leakage contract.

The single rule for every feature here:

    A feature stamped at `target_time` may only be a function of values whose
    `interval_end` is STRICTLY LESS than `target_time`.

The implementations use pandas `shift()` on a hourly-contiguous index so a
gap in the input becomes a NaN rather than a silent jump that could leak
future information into the past.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import pandas as pd


HOUR = pd.Timedelta(hours=1)


@dataclass(frozen=True)
class LagSpec:
    column: str
    periods: int           # number of hourly periods to shift (positive = past)
    new_name: str | None = None

    def output_name(self) -> str:
        return self.new_name or f"{self.column}_lag_{self.periods}h"


@dataclass(frozen=True)
class RollingSpec:
    column: str
    window_hours: int
    agg: str               # 'mean', 'std', 'min', 'max'
    new_name: str | None = None

    def output_name(self) -> str:
        return self.new_name or f"{self.column}_rolling_{self.window_hours}h_{self.agg}"


def reindex_hourly(df: pd.DataFrame, timestamp_col: str = "interval_end") -> pd.DataFrame:
    """Reindex to a continuous hourly grid so shift() means 'shift one hour'."""
    if df.empty:
        return df.copy()
    df = df.sort_values(timestamp_col).copy()
    df[timestamp_col] = pd.to_datetime(df[timestamp_col])
    full_index = pd.date_range(df[timestamp_col].min(), df[timestamp_col].max(), freq="1h")
    df = df.set_index(timestamp_col).reindex(full_index)
    df.index.name = timestamp_col
    return df.reset_index()


def add_lag_features(
    df: pd.DataFrame,
    specs: Iterable[LagSpec],
    *,
    group_col: str | None = "area",
    timestamp_col: str = "interval_end",
) -> pd.DataFrame:
    """Add lag features grouped by `group_col` (e.g. area).

    The data is reindexed to an hourly grid per group BEFORE shifting so a
    missing hour does not collapse and pull a future value into the gap.
    """
    if group_col is None or group_col not in df.columns:
        return _add_lag_one_group(df, specs, timestamp_col=timestamp_col)

    out_frames: list[pd.DataFrame] = []
    for group_val, group_df in df.groupby(group_col, sort=False):
        gridded = reindex_hourly(group_df.drop(columns=[group_col]), timestamp_col)
        with_lags = _add_lag_one_group(gridded, specs, timestamp_col=timestamp_col)
        with_lags[group_col] = group_val
        out_frames.append(with_lags)
    out = pd.concat(out_frames, ignore_index=True)
    cols = [group_col, timestamp_col] + [c for c in out.columns if c not in (group_col, timestamp_col)]
    return out[cols]


def _add_lag_one_group(
    df: pd.DataFrame, specs: Iterable[LagSpec], *, timestamp_col: str
) -> pd.DataFrame:
    df = df.sort_values(timestamp_col).copy()
    for spec in specs:
        if spec.column not in df.columns:
            raise KeyError(f"Lag source column missing: {spec.column}")
        if spec.periods <= 0:
            raise ValueError(
                f"LagSpec.periods must be > 0 to enforce no-future-leakage (got {spec.periods})"
            )
        df[spec.output_name()] = df[spec.column].shift(spec.periods)
    return df


def add_rolling_features(
    df: pd.DataFrame,
    specs: Iterable[RollingSpec],
    *,
    group_col: str | None = "area",
    timestamp_col: str = "interval_end",
) -> pd.DataFrame:
    """Rolling aggregates over PAST `window_hours` only.

    We shift by 1 hour BEFORE rolling so the window for target_time t
    covers [t - window_hours, t - 1h] inclusive — never t itself.
    """
    if group_col is None or group_col not in df.columns:
        return _add_rolling_one_group(df, specs, timestamp_col=timestamp_col)

    out_frames: list[pd.DataFrame] = []
    for group_val, group_df in df.groupby(group_col, sort=False):
        gridded = reindex_hourly(group_df.drop(columns=[group_col]), timestamp_col)
        with_roll = _add_rolling_one_group(gridded, specs, timestamp_col=timestamp_col)
        with_roll[group_col] = group_val
        out_frames.append(with_roll)
    out = pd.concat(out_frames, ignore_index=True)
    cols = [group_col, timestamp_col] + [c for c in out.columns if c not in (group_col, timestamp_col)]
    return out[cols]


def _add_rolling_one_group(
    df: pd.DataFrame, specs: Iterable[RollingSpec], *, timestamp_col: str
) -> pd.DataFrame:
    df = df.sort_values(timestamp_col).copy()
    for spec in specs:
        if spec.column not in df.columns:
            raise KeyError(f"Rolling source column missing: {spec.column}")
        if spec.window_hours <= 0:
            raise ValueError(f"window_hours must be > 0 (got {spec.window_hours})")
        shifted = df[spec.column].shift(1)
        rolled = shifted.rolling(window=spec.window_hours, min_periods=spec.window_hours)
        if spec.agg == "mean":
            df[spec.output_name()] = rolled.mean()
        elif spec.agg == "std":
            df[spec.output_name()] = rolled.std()
        elif spec.agg == "min":
            df[spec.output_name()] = rolled.min()
        elif spec.agg == "max":
            df[spec.output_name()] = rolled.max()
        else:
            raise ValueError(f"Unsupported agg {spec.agg!r}")
    return df


# Default feature specs for the SMP hourly target -----------------------------

DEFAULT_SMP_LAGS = [
    LagSpec("smp_krw_per_kwh", 1, "smp_lag_1h"),
    LagSpec("smp_krw_per_kwh", 2, "smp_lag_2h"),
    LagSpec("smp_krw_per_kwh", 3, "smp_lag_3h"),
    LagSpec("smp_krw_per_kwh", 24, "smp_lag_24h"),
    LagSpec("smp_krw_per_kwh", 48, "smp_lag_48h"),
    LagSpec("smp_krw_per_kwh", 168, "smp_lag_168h"),
]

DEFAULT_SMP_ROLLINGS = [
    RollingSpec("smp_krw_per_kwh", 24, "mean", "smp_rolling_24h_mean"),
    RollingSpec("smp_krw_per_kwh", 24, "std", "smp_rolling_24h_std"),
    RollingSpec("smp_krw_per_kwh", 168, "mean", "smp_rolling_7d_mean"),
    RollingSpec("smp_krw_per_kwh", 168, "std", "smp_rolling_7d_std"),
]

DEFAULT_DEMAND_LAGS = [
    LagSpec("demand_forecast_mw", 1, "demand_lag_1h"),
    LagSpec("demand_forecast_mw", 24, "demand_lag_24h"),
    LagSpec("demand_forecast_mw", 168, "demand_lag_168h"),
]

DEFAULT_DEMAND_ROLLINGS = [
    RollingSpec("demand_forecast_mw", 24, "mean", "demand_rolling_24h_mean"),
    RollingSpec("demand_forecast_mw", 168, "mean", "demand_rolling_7d_mean"),
]
