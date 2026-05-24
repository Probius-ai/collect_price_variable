"""Time helpers for KPX hourly conventions.

KPX SMP uses trade hours 1..24 where hour H means the interval ending at H:00.
For example trade_hour=6 covers 05:00–06:00 (timestamp = 06:00 of trade_date).
trade_hour=24 means 23:00 of trade_date through 24:00 — which is 00:00 of
trade_date + 1 day. We normalize everything to a single timezone-naive
`interval_end` timestamp in Asia/Seoul.
"""

from __future__ import annotations

from datetime import datetime, time, timedelta, timezone
from typing import Iterable

import pandas as pd

KST = "Asia/Seoul"


def kpx_trade_hour_to_interval_end(trade_date: datetime, trade_hour: int) -> datetime:
    """Convert (date, hour 1..24) to the end-of-interval timestamp.

    trade_hour=24 means midnight of the next calendar day.
    """
    if not 1 <= int(trade_hour) <= 24:
        raise ValueError(f"trade_hour must be in 1..24, got {trade_hour}")
    if isinstance(trade_date, pd.Timestamp):
        trade_date = trade_date.to_pydatetime()
    base = datetime.combine(trade_date.date(), time(0, 0))
    return base + timedelta(hours=int(trade_hour))


def kpx_trade_hour_to_interval_start(trade_date: datetime, trade_hour: int) -> datetime:
    return kpx_trade_hour_to_interval_end(trade_date, trade_hour) - timedelta(hours=1)


def build_hourly_index(start: datetime, end_inclusive: datetime) -> pd.DatetimeIndex:
    """Build a contiguous hourly DatetimeIndex (interval-end convention) in KST.

    `start` and `end_inclusive` are treated as wall-clock KST times.
    """
    return pd.date_range(start=start, end=end_inclusive, freq="1h")


def expected_24_hours(trade_date: datetime) -> pd.DatetimeIndex:
    """Return the 24 expected interval-end timestamps for a given trade date."""
    start = kpx_trade_hour_to_interval_end(trade_date, 1)
    end = kpx_trade_hour_to_interval_end(trade_date, 24)
    return pd.date_range(start=start, end=end, freq="1h")


def assert_no_duplicate_timestamps(timestamps: Iterable, label: str = "timestamps") -> None:
    series = pd.Series(list(timestamps))
    dupes = series[series.duplicated()].unique()
    if len(dupes):
        raise ValueError(f"Duplicate {label} found: {list(dupes)[:5]}…")


def collected_now_utc() -> datetime:
    """Naive UTC `now` used for `collected_at` columns."""
    return datetime.now(timezone.utc).replace(tzinfo=None)
