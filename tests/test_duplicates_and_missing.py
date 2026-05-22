from __future__ import annotations

import pandas as pd
import pytest

from src.validation.leakage_checks import (
    assert_hourly_contiguous,
    assert_no_duplicate_keys,
    daily_missing_hour_report,
)


def test_assert_no_duplicate_keys_passes_on_unique(synthetic_smp_dataframe):
    assert_no_duplicate_keys(synthetic_smp_dataframe, ["area", "interval_end"])


def test_assert_no_duplicate_keys_raises_on_dupe(synthetic_smp_dataframe):
    dup = pd.concat([synthetic_smp_dataframe, synthetic_smp_dataframe.head(3)])
    with pytest.raises(AssertionError, match="Duplicate"):
        assert_no_duplicate_keys(dup, ["area", "interval_end"])


def test_assert_hourly_contiguous_passes_on_clean(synthetic_smp_dataframe):
    assert_hourly_contiguous(synthetic_smp_dataframe)


def test_assert_hourly_contiguous_raises_on_gap(synthetic_smp_dataframe):
    # delete 5 consecutive hours to introduce a gap
    df = synthetic_smp_dataframe.drop(synthetic_smp_dataframe.index[10:15]).copy()
    with pytest.raises(AssertionError, match="gap"):
        assert_hourly_contiguous(df)


def test_daily_missing_hour_report_counts_correctly(synthetic_smp_dataframe):
    df = synthetic_smp_dataframe.drop(synthetic_smp_dataframe.index[5:8]).copy()
    report = daily_missing_hour_report(df)
    bad_day = report[report["missing_hours"] > 0]
    assert not bad_day.empty
    assert int(bad_day["missing_hours"].iloc[0]) == 3
