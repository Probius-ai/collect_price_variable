"""Monthly feature builder + monthly naive model tests (round-2 schema)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.features.build_monthly import (
    DEFAULT_MONTHLY_LAGS,
    DEFAULT_MONTHLY_ROLLINGS,
    MonthlyLagSpec,
    _exogenous_lag_1m,
    add_calendar_features,
    add_monthly_lags,
    add_monthly_rollings,
)
from src.models.metrics import compute_metrics
from src.models.naive import NaiveLag1m, SeasonalNaiveLag12m


@pytest.fixture
def synthetic_monthly_smp() -> pd.DataFrame:
    """Round-2 long-format synthetic monthly SMP for `mainland`."""
    rng = np.random.default_rng(11)
    months = pd.date_range("2015-01-01", "2024-12-01", freq="MS")
    seasonal = 30 * np.sin(2 * np.pi * months.month / 12)
    trend = np.linspace(0, 40, len(months))
    noise = rng.normal(0, 5, len(months))
    smp = 100 + seasonal + trend + noise
    return pd.DataFrame({
        "period_month": months,
        "smp_krw_per_kwh": smp,
        "area": "mainland",
    })


def test_monthly_lag_zero_raises():
    df = pd.DataFrame({"period_month": pd.date_range("2020-01-01", periods=3, freq="MS"), "v": [1, 2, 3]})
    with pytest.raises(ValueError, match="no-future-leakage"):
        add_monthly_lags(df, [MonthlyLagSpec("v", 0)])


def test_monthly_lag_1m_matches_previous_value(synthetic_monthly_smp):
    out = add_monthly_lags(
        synthetic_monthly_smp, [MonthlyLagSpec("smp_krw_per_kwh", 1, "smp_lag_1m")]
    )
    expected = synthetic_monthly_smp["smp_krw_per_kwh"].shift(1)
    pd.testing.assert_series_equal(
        out["smp_lag_1m"].reset_index(drop=True),
        expected.reset_index(drop=True),
        check_names=False,
    )


def test_monthly_rolling_excludes_current_month(synthetic_monthly_smp):
    out = add_monthly_rollings(synthetic_monthly_smp, DEFAULT_MONTHLY_ROLLINGS)
    expected = synthetic_monthly_smp["smp_krw_per_kwh"].iloc[0:12].mean()
    assert out["smp_rolling_12m_mean"].iloc[12] == pytest.approx(expected)


def test_calendar_features_have_expected_columns(synthetic_monthly_smp):
    out = add_calendar_features(synthetic_monthly_smp)
    for col in ["year", "month", "quarter", "month_sin", "month_cos", "is_summer", "is_winter"]:
        assert col in out.columns


def test_monthly_naive_models_match_lag_columns(synthetic_monthly_smp):
    df = add_monthly_lags(synthetic_monthly_smp, DEFAULT_MONTHLY_LAGS)
    df = df.dropna().reset_index(drop=True)
    target = df["smp_krw_per_kwh"]

    m1 = NaiveLag1m().fit(df, target)
    np.testing.assert_allclose(m1.predict(df).to_numpy(), df["smp_lag_1m"].to_numpy())

    m12 = SeasonalNaiveLag12m().fit(df, target)
    np.testing.assert_allclose(m12.predict(df).to_numpy(), df["smp_lag_12m"].to_numpy())


def test_seasonal_naive_beats_lag_1m_on_seasonal_data(synthetic_monthly_smp):
    df = add_monthly_lags(synthetic_monthly_smp, DEFAULT_MONTHLY_LAGS)
    df = df.dropna().reset_index(drop=True)
    df["target"] = df["smp_krw_per_kwh"].shift(-1)
    df = df.dropna(subset=["target"]).reset_index(drop=True)

    mae_lag1 = compute_metrics(df["target"], df["smp_lag_1m"]).mae
    mae_lag12 = compute_metrics(df["target"], df["smp_lag_12m"]).mae
    assert mae_lag12 < mae_lag1, (mae_lag1, mae_lag12)


def test_exogenous_lag_returns_nan_across_missing_months():
    """Round-1 regression: gaps in exogenous monthly data must surface as NaN."""
    df = pd.DataFrame(
        {
            "period_month": pd.to_datetime(["2024-01-01", "2024-02-01", "2024-04-01"]),
            "settlement_unit_price_avg": [100.0, 110.0, 130.0],
        }
    )
    out = _exogenous_lag_1m(df, value_cols=["settlement_unit_price_avg"]).set_index(
        "period_month"
    )
    assert out.loc["2024-02-01", "settlement_unit_price_avg_lag_1m"] == 100.0
    assert out.loc["2024-03-01", "settlement_unit_price_avg_lag_1m"] == 110.0
    assert pd.isna(out.loc["2024-04-01", "settlement_unit_price_avg_lag_1m"])


def test_exogenous_lag_aggregates_duplicate_year_months_deterministically():
    """Round-1 regression: duplicates are aggregated by mean, order-independent."""
    base = pd.DataFrame(
        {
            "period_month": pd.to_datetime(
                ["2024-02-01", "2024-01-01", "2024-02-01", "2024-03-01"]
            ),
            "v": [99.0, 10.0, 21.0, 30.0],
        }
    )
    expected_feb_mean = (99.0 + 21.0) / 2
    out_base = _exogenous_lag_1m(base, value_cols=["v"]).set_index("period_month")
    assert out_base.loc["2024-02-01", "v_lag_1m"] == 10.0
    assert out_base.loc["2024-03-01", "v_lag_1m"] == expected_feb_mean
    shuffled = base.iloc[::-1].reset_index(drop=True)
    out_shuf = _exogenous_lag_1m(shuffled, value_cols=["v"]).set_index("period_month")
    pd.testing.assert_frame_equal(out_base.sort_index(), out_shuf.sort_index())


def test_build_smp_monthly_features_round_trip(tmp_path, monkeypatch, synthetic_monthly_smp):
    """End-to-end: write parsed_*.parquet into the source root, then build features."""
    from src.utils import io as io_mod
    import importlib

    class _Stub:
        data_dir = tmp_path / "data"
    monkeypatch.setattr(io_mod, "get_settings", lambda: _Stub())
    bm = importlib.reload(importlib.import_module("src.features.build_monthly"))

    target_dir = (
        bm.source_root_dir("kpx_smp_monthly_kepco_file") / "2024" / "12" / "01"
    )
    target_dir.mkdir(parents=True, exist_ok=True)
    parsed = synthetic_monthly_smp.copy()
    parsed["source_id"] = "kpx_smp_monthly_kepco_file"
    parsed["collected_at"] = pd.Timestamp("2026-05-24")
    parsed["source_file"] = "synthetic.csv"
    parsed.to_parquet(target_dir / "parsed_test.parquet", index=False)

    feats, side_info = bm.build_smp_monthly_features(
        area="mainland", horizon_months=1, include_settlement=False
    )
    assert len(feats) > 24
    assert "target_smp_t_plus_h_months" in feats.columns
    assert side_info["area"] == "mainland"
    assert side_info["rows"] == len(feats)


def test_priority_dedup_prefers_lower_number(tmp_path, monkeypatch):
    """When two SMP sources cover the same (period, area), priority 1 wins."""
    from src.utils import io as io_mod
    import importlib

    class _Stub:
        data_dir = tmp_path / "data"
    monkeypatch.setattr(io_mod, "get_settings", lambda: _Stub())
    bm = importlib.reload(importlib.import_module("src.features.build_monthly"))

    common = pd.DataFrame({
        "period_month": [pd.Timestamp("2025-01-01")] * 3,
        "area": ["mainland", "jeju", "integrated"],
        "smp_krw_per_kwh": [100.0, 101.0, 102.0],
        "source_id": "kpx_smp_monthly_kepco_file",
        "collected_at": pd.Timestamp("2026-05-24"),
        "source_file": "k.csv",
    })
    other = common.copy()
    other["smp_krw_per_kwh"] = [777.0, 778.0, 779.0]
    other["source_id"] = "kpx_smp_monthly_home_avg_file"
    other["source_file"] = "h.csv"

    kdir = bm.source_root_dir("kpx_smp_monthly_kepco_file") / "2025" / "01" / "01"
    kdir.mkdir(parents=True, exist_ok=True)
    common.to_parquet(kdir / "parsed_test.parquet", index=False)
    hdir = bm.source_root_dir("kpx_smp_monthly_home_avg_file") / "2025" / "01" / "01"
    hdir.mkdir(parents=True, exist_ok=True)
    other.to_parquet(hdir / "parsed_test.parquet", index=False)

    long = bm.load_smp_monthly_long()
    main = long[(long["period_month"] == pd.Timestamp("2025-01-01"))
                & (long["area"] == "mainland")].iloc[0]
    # Priority 1 (KEPCO) should win
    assert main["source_id"] == "kpx_smp_monthly_kepco_file"
    assert main["smp_krw_per_kwh"] == 100.0
    # And the conflict must be visible in the dedup log
    log = long.attrs["_priority_dedup_log"]
    assert {"kpx_smp_monthly_kepco_file", "kpx_smp_monthly_home_avg_file"}.issubset(
        set(log["source_id"])
    )
    # The dropped row's reason must be "lower_priority"
    dropped_main = log[
        (log["area"] == "mainland")
        & (log["role"] == "dropped")
    ]
    assert len(dropped_main) == 1
    assert dropped_main.iloc[0]["reason"] == "lower_priority"


def test_revision_dedup_prefers_newer_collected_at(tmp_path, monkeypatch):
    """Same-source same-(period,area) snapshots: latest collected_at wins.

    Regression for the Codex adversarial-review finding: when KPX/EPSIS
    publishes a corrected SMP value, a later re-load creates a second
    parsed_*.parquet for the same (period_month, area). The feature builder
    must pick the NEWER snapshot and record the OLDER one as dropped with
    reason=``older_revision``.
    """
    from src.utils import io as io_mod
    import importlib

    class _Stub:
        data_dir = tmp_path / "data"
    monkeypatch.setattr(io_mod, "get_settings", lambda: _Stub())
    bm = importlib.reload(importlib.import_module("src.features.build_monthly"))

    base = pd.DataFrame({
        "period_month": [pd.Timestamp("2025-01-01")],
        "area": ["mainland"],
        "smp_krw_per_kwh": [100.0],  # value A — original publication
        "source_id": "kpx_smp_monthly_kepco_file",
        "collected_at": pd.Timestamp("2026-04-01T12:00:00"),  # older
        "source_file": "kepco_2025_01_v1.csv",
        "source_file_sha256": "a" * 64,
        "source_priority": 1,
    })
    revised = base.copy()
    revised["smp_krw_per_kwh"] = [123.45]  # value B — corrected publication
    revised["collected_at"] = pd.Timestamp("2026-05-01T09:30:00")  # newer
    revised["source_file"] = "kepco_2025_01_v2.csv"
    revised["source_file_sha256"] = "b" * 64

    # Both snapshots land in the SAME source-root subdir but under different
    # event-date partitions, mimicking two re-loads of the same period.
    root = bm.source_root_dir("kpx_smp_monthly_kepco_file")
    old_dir = root / "2025" / "01" / "01"
    new_dir = root / "2025" / "01" / "02"
    old_dir.mkdir(parents=True, exist_ok=True)
    new_dir.mkdir(parents=True, exist_ok=True)
    base.to_parquet(old_dir / "parsed_20260401T120000Z.parquet", index=False)
    revised.to_parquet(new_dir / "parsed_20260501T093000Z.parquet", index=False)

    long = bm.load_smp_monthly_long()
    row = long[
        (long["period_month"] == pd.Timestamp("2025-01-01"))
        & (long["area"] == "mainland")
    ]
    assert len(row) == 1, "Same (period, area) must collapse to a single row"
    selected = row.iloc[0]
    assert selected["smp_krw_per_kwh"] == 123.45, (
        "Newer collected_at must supersede the older revision"
    )
    assert selected["source_file_sha256"] == "b" * 64
    assert "v2" in selected["source_file"]
    assert selected["collected_at"] == pd.Timestamp("2026-05-01T09:30:00")

    log = long.attrs["_priority_dedup_log"]
    sel_rows = log[(log["area"] == "mainland") & (log["role"] == "selected")]
    drop_rows = log[(log["area"] == "mainland") & (log["role"] == "dropped")]
    assert len(sel_rows) == 1 and len(drop_rows) == 1
    assert sel_rows.iloc[0]["smp_krw_per_kwh"] == 123.45
    assert sel_rows.iloc[0]["source_file_sha256"] == "b" * 64
    assert drop_rows.iloc[0]["smp_krw_per_kwh"] == 100.0, (
        "The older value must be recorded as the dropped candidate"
    )
    assert drop_rows.iloc[0]["source_file_sha256"] == "a" * 64
    assert drop_rows.iloc[0]["collected_at"] == pd.Timestamp("2026-04-01T12:00:00")
    assert drop_rows.iloc[0]["reason"] == "older_revision"
