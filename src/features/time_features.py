"""Calendar / seasonality features derived from the interval-end timestamp."""

from __future__ import annotations

import holidays
import numpy as np
import pandas as pd

KR_HOLIDAYS = holidays.country_holidays("KR")


def _season(month: int) -> str:
    if month in (3, 4, 5):
        return "spring"
    if month in (6, 7, 8):
        return "summer"
    if month in (9, 10, 11):
        return "autumn"
    return "winter"


def add_time_features(df: pd.DataFrame, timestamp_col: str = "interval_end") -> pd.DataFrame:
    """Add year/month/day/hour/dow/season/holiday/peak-season flags.

    `timestamp_col` must already be hour-aligned (interval-end convention).
    """
    if timestamp_col not in df.columns:
        raise KeyError(f"{timestamp_col} not in dataframe columns")
    out = df.copy()
    ts = pd.to_datetime(out[timestamp_col])
    out["year"] = ts.dt.year
    out["month"] = ts.dt.month
    out["day"] = ts.dt.day
    # trade_hour=24 maps to 00:00 of next day, so hour 0 here represents trade_hour 24.
    out["hour"] = ts.dt.hour.where(ts.dt.hour != 0, 24)
    out["day_of_week"] = ts.dt.dayofweek
    out["is_weekend"] = out["day_of_week"].isin([5, 6])
    dates = ts.dt.date
    out["is_holiday"] = dates.map(lambda d: d in KR_HOLIDAYS)
    out["season"] = out["month"].apply(_season)
    out["is_summer"] = out["month"].isin([6, 7, 8])
    out["is_winter"] = out["month"].isin([12, 1, 2])
    out["is_peak_load_season"] = out["is_summer"] | out["is_winter"]
    out["hour_sin"] = np.sin(2 * np.pi * out["hour"] / 24)
    out["hour_cos"] = np.cos(2 * np.pi * out["hour"] / 24)
    out["dow_sin"] = np.sin(2 * np.pi * out["day_of_week"] / 7)
    out["dow_cos"] = np.cos(2 * np.pi * out["day_of_week"] / 7)
    return out
