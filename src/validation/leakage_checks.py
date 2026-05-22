"""Static, dataframe-level checks for data leakage and time integrity."""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import pandas as pd


def assert_no_duplicate_keys(
    df: pd.DataFrame, key_cols: Iterable[str], label: str = "rows"
) -> None:
    """Raise if (key_cols) is not unique."""
    key_cols = list(key_cols)
    dupes = df[df.duplicated(key_cols, keep=False)]
    if len(dupes):
        sample = dupes.head(5).to_dict(orient="records")
        raise AssertionError(f"Duplicate {label} on {key_cols}: {sample}")


def assert_hourly_contiguous(
    df: pd.DataFrame,
    *,
    timestamp_col: str = "interval_end",
    group_col: str | None = "area",
    allow_gap_minutes: int = 0,
) -> None:
    """Raise if hourly data has gaps larger than allow_gap_minutes."""
    groups = [(None, df)] if (group_col is None or group_col not in df.columns) else df.groupby(group_col)
    for group_val, group_df in groups:
        ts = pd.to_datetime(group_df[timestamp_col]).sort_values()
        if len(ts) < 2:
            continue
        diffs = ts.diff().dropna()
        bad = diffs[diffs > pd.Timedelta(hours=1) + pd.Timedelta(minutes=allow_gap_minutes)]
        if len(bad):
            first_bad = ts.iloc[bad.index[0]]
            raise AssertionError(
                f"Hourly gap > 1h detected (group={group_val}) at {first_bad}"
            )


def assert_no_future_leakage(
    df: pd.DataFrame,
    *,
    target_col: str,
    timestamp_col: str,
    feature_cols: Iterable[str],
    forecast_horizon_hours: int,
) -> None:
    """Sanity-check correlations between features and the *current-hour* SMP.

    The features at row t represent what we know at time t. The target at row
    t is SMP at t + horizon. If a feature equals the future SMP, the per-row
    relationship `feature[t] == target[t]` will hold systematically — this
    check raises when that happens.
    """
    if target_col not in df.columns:
        raise KeyError(f"target column missing: {target_col}")
    feature_cols = [c for c in feature_cols if c in df.columns]
    if not feature_cols:
        return
    if forecast_horizon_hours <= 0:
        raise ValueError("forecast_horizon_hours must be > 0")

    # A leak shows up as: feature[t] equals target[t] exactly. We compare on
    # the numeric subset; equality of arbitrary nullable strings is meaningless.
    target = pd.to_numeric(df[target_col], errors="coerce")
    suspicious: list[str] = []
    for col in feature_cols:
        feat = pd.to_numeric(df[col], errors="coerce")
        if feat.isna().all():
            continue
        # exact equality on every non-null pair is a strong leakage signal
        mask = feat.notna() & target.notna()
        if mask.sum() < 24:
            continue
        if np.array_equal(feat[mask].to_numpy(), target[mask].to_numpy()):
            suspicious.append(col)
    if suspicious:
        raise AssertionError(
            "Possible future leakage: features identical to target -> "
            f"{suspicious}"
        )


def daily_missing_hour_report(
    df: pd.DataFrame,
    *,
    timestamp_col: str = "interval_end",
    group_col: str | None = "area",
) -> pd.DataFrame:
    """Return per-day counts vs expected 24 hours."""
    df = df.copy()
    df[timestamp_col] = pd.to_datetime(df[timestamp_col])
    df["_date"] = df[timestamp_col].dt.date
    group_cols = [group_col, "_date"] if group_col and group_col in df.columns else ["_date"]
    counts = df.groupby(group_cols).size().rename("hour_count").reset_index()
    counts["missing_hours"] = 24 - counts["hour_count"]
    return counts
