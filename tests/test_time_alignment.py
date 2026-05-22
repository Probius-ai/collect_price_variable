from __future__ import annotations

from datetime import datetime, timedelta

import pandas as pd
import pytest

from src.utils.time import (
    expected_24_hours,
    kpx_trade_hour_to_interval_end,
    kpx_trade_hour_to_interval_start,
)


def test_trade_hour_24_rolls_to_next_day_midnight():
    d = datetime(2024, 3, 1)
    end = kpx_trade_hour_to_interval_end(d, 24)
    assert end == datetime(2024, 3, 2, 0, 0)


def test_trade_hour_6_covers_5_to_6():
    d = datetime(2024, 3, 1)
    assert kpx_trade_hour_to_interval_start(d, 6) == datetime(2024, 3, 1, 5, 0)
    assert kpx_trade_hour_to_interval_end(d, 6) == datetime(2024, 3, 1, 6, 0)


@pytest.mark.parametrize("bad_hour", [0, -1, 25, 100])
def test_invalid_trade_hour_raises(bad_hour):
    with pytest.raises(ValueError):
        kpx_trade_hour_to_interval_end(datetime(2024, 1, 1), bad_hour)


def test_expected_24_hours_is_contiguous_and_ends_next_midnight():
    idx = expected_24_hours(datetime(2024, 1, 1))
    assert len(idx) == 24
    diffs = pd.Series(idx).diff().dropna()
    assert (diffs == pd.Timedelta(hours=1)).all()
    assert idx[0] == datetime(2024, 1, 1, 1, 0)
    assert idx[-1] == datetime(2024, 1, 2, 0, 0)
