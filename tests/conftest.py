"""Shared pytest fixtures."""

from __future__ import annotations

import os
import sys
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# Make `src` importable in tests without installing the package.
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Default env so settings load even when .env is absent.
os.environ.setdefault("PROJECT_TIMEZONE", "Asia/Seoul")
os.environ.setdefault("LOG_LEVEL", "WARNING")


@pytest.fixture
def synthetic_smp_dataframe() -> pd.DataFrame:
    """30 days of hourly SMP for mainland with a clean daily cycle.

    Each row has trade_date, trade_hour, area, interval_start, interval_end,
    smp_krw_per_kwh, demand_forecast_mw -- the shape produced by the
    KpxSmpCollector after parsing.
    """
    start = datetime(2024, 1, 1, 1, 0, 0)
    n_hours = 30 * 24
    timestamps = [start + timedelta(hours=i) for i in range(n_hours)]
    rng = np.random.default_rng(7)
    rows = []
    for ts in timestamps:
        # SMP has a daily 24-hour shape + small noise
        smp = 100.0 + 25.0 * np.sin(2 * np.pi * ts.hour / 24) + rng.normal(0, 3)
        demand = 60_000 + 8_000 * np.sin(2 * np.pi * (ts.hour - 4) / 24) + rng.normal(0, 500)
        trade_date = (ts - timedelta(hours=1)).date() if ts.hour == 0 else ts.date()
        trade_hour = 24 if ts.hour == 0 else ts.hour
        rows.append(
            dict(
                area="mainland",
                trade_date=trade_date,
                trade_hour=trade_hour,
                interval_start=ts - timedelta(hours=1),
                interval_end=ts,
                smp_krw_per_kwh=smp,
                demand_forecast_mw=demand,
                source_name="kpx_smp_day_ahead",
                schema_version=0,
            )
        )
    return pd.DataFrame(rows)
