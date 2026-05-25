"""Round-3 tests: capacity + transaction loaders, feature joins, units, and
overlap-conditional price computation.

These tests exercise the file_sources added for KPX capacity (yearly +
monthly) and KPX transaction (volume hourly + amount daily) sources, plus
the optional feature joins that wire them into build_smp_monthly_features.

The 12 explicit assertions called out by the round-3 spec live here:

  1) capacity files unit == MW
  2) transaction volume unit == MWh
  3) transaction amount unit == KRW
  4) amount file column `전력거래금액(원)` maps to transaction_amount_krw
  5) multi-row headers are not collapsed incorrectly
  6) renewable sub-fuels are preserved
  7) monthly feature table row count does not change after optional joins
  8) join key is period_month only
"""

from __future__ import annotations

import importlib
from pathlib import Path

import pandas as pd
import pytest

from src.collectors.kpx_files import (
    CAPACITY_MONTHLY_BY_FUEL_COLUMNS,
    CAPACITY_MONTHLY_BY_GENERATION_TYPE_COLUMNS,
    CAPACITY_YEARLY_COLUMNS,
    TRANSACTION_AMOUNT_DAILY_COLUMNS,
    TRANSACTION_VOLUME_HOURLY_COLUMNS,
    KpxCapacityMonthlyByFuelHomeFileLoader,
    KpxCapacityMonthlyByGenerationTypeHomeFileLoader,
    KpxCapacityYearlyByEnergySourceHomeFileLoader,
    KpxTransactionAmountDailyByFuelFileLoader,
    KpxTransactionVolumeHourlyByFuelFileLoader,
)
from src.config.settings import get_source_config


# ---------------------------------------------------------------------------
# Helpers: locate the file in the drop directory by content (NFC-tolerant)
# ---------------------------------------------------------------------------

DROP_ROOT = Path("data/raw/manual_or_filedata")


def _file_in(source_id: str, *, contains: str | None = None) -> Path:
    """Return a file from a drop directory, optionally filtering by substring."""
    dir_ = DROP_ROOT / source_id
    if not dir_.exists():
        pytest.skip(f"Drop dir missing for {source_id}: {dir_}")
    candidates = sorted(p for p in dir_.iterdir() if p.is_file())
    if contains is not None:
        candidates = [p for p in candidates if contains in p.name]
    if not candidates:
        pytest.skip(f"No files in {dir_} matching {contains!r}")
    return candidates[0]


# ===========================================================================
# 1. Loader-level tests (multi-row header + unit + fuel mapping)
# ===========================================================================


def test_capacity_yearly_energy_source_multirow_header_loader():
    loader = KpxCapacityYearlyByEnergySourceHomeFileLoader()
    fp = _file_in("kpx_capacity_yearly_by_energy_source_home_file", contains="2024")
    df = loader.parse_file(fp)
    # Canonical schema must be present
    assert set(CAPACITY_YEARLY_COLUMNS).issubset(df.columns)
    # Year must come from filename, not file body
    assert (df["period_year"] == 2024).all()
    # Multi-row header flatten preserved level1|level2 distinction:
    # we must see at least one row whose level1 is "기력" with a non-empty level2
    steam_rows = df[df["capacity_category_level1"] == "기력"]
    assert not steam_rows.empty, "Steam (기력) rows missing — header flatten broken"
    assert (steam_rows["capacity_category_level2"] != "").any(), (
        "Multi-row header was collapsed: level2 sub-categories under 기력 lost"
    )
    # MW unit must match (numeric, int or float — pandas infers int when all
    # values happen to be whole)
    assert df["capacity_mw"].dtype.kind in {"f", "i"}


def test_capacity_monthly_generation_type_multirow_header_loader():
    loader = KpxCapacityMonthlyByGenerationTypeHomeFileLoader()
    fp = _file_in("kpx_capacity_monthly_by_generation_type_home_file")
    df = loader.parse_file(fp)
    assert set(CAPACITY_MONTHLY_BY_GENERATION_TYPE_COLUMNS).issubset(df.columns)
    # After 합계-filter, every row's metadata columns must be "합계".
    for meta_col in ("member_type", "dispatch_type", "business_type", "region"):
        assert (df[meta_col] == "합계").all(), (
            f"Default filter failed: {meta_col} contains non-합계 rows"
        )
    # capacity_type_canonical must include canonical names like 'steam_coal_bituminous'
    seen = set(df["capacity_type_canonical"].dropna().unique())
    expected = {"nuclear", "steam_coal_bituminous", "steam_gas",
                "combined_gas", "pumped_storage", "renewable", "total"}
    assert expected.issubset(seen), (
        f"Canonical generation_type mapping incomplete. Missing: {expected - seen}"
    )


def test_capacity_monthly_fuel_multirow_header_loader():
    loader = KpxCapacityMonthlyByFuelHomeFileLoader()
    fp = _file_in("kpx_capacity_monthly_by_fuel_home_file")
    df = loader.parse_file(fp)
    assert set(CAPACITY_MONTHLY_BY_FUEL_COLUMNS).issubset(df.columns)
    # Renewable sub-fuels must be preserved as distinct rows (not collapsed)
    fuels = set(df["fuel_type"].dropna().unique())
    sub_fuels = {"renewable_solar", "renewable_wind", "renewable_fuel_cell",
                 "renewable_hydro", "renewable_bio", "renewable_waste"}
    assert sub_fuels.issubset(fuels), (
        f"Renewable sub-fuels missing: {sub_fuels - fuels}. "
        "Multi-row header flatten or fuel mapping is collapsing them."
    )


def test_transaction_volume_hourly_by_fuel_loader():
    loader = KpxTransactionVolumeHourlyByFuelFileLoader()
    fp = _file_in("kpx_transaction_volume_hourly_by_fuel_file")
    df = loader.parse_file(fp)
    assert set(TRANSACTION_VOLUME_HOURLY_COLUMNS).issubset(df.columns)
    # Unit is MWh — value column must be numeric (int or float)
    assert df["transaction_volume_mwh"].dtype.kind in {"f", "i"}
    # Hour must be 1..24 (raw vendor convention preserved)
    assert df["trade_hour"].between(1, 24).all()


def test_transaction_amount_daily_by_fuel_loader():
    loader = KpxTransactionAmountDailyByFuelFileLoader()
    fp = _file_in("kpx_transaction_amount_daily_by_fuel_file")
    df = loader.parse_file(fp)
    assert set(TRANSACTION_AMOUNT_DAILY_COLUMNS).issubset(df.columns)
    # Vendor column 전력거래금액(원) maps to transaction_amount_krw, KRW unit
    assert df["transaction_amount_krw"].dtype.kind in {"f", "i"}
    # Spot-check: at least one renewable_biomass row (specific to this file's
    # fuel mapping that includes 바이오매스 → renewable_biomass)
    assert "renewable_biomass" in set(df["fuel_type"].dropna().unique()) or \
           "renewable_bio_srf" in set(df["fuel_type"].dropna().unique()), (
        "Daily amount fuel mapping missing renewable_biomass / bio_srf — "
        "vendor names like 바이오매스/바이오SRF must be preserved."
    )


# ===========================================================================
# 2. Behaviour tests
# ===========================================================================


def test_capacity_monthly_filters_only_total_rows():
    """Both monthly capacity loaders must drop non-합계 rows in the MVP path."""
    fuel_loader = KpxCapacityMonthlyByFuelHomeFileLoader()
    gen_loader = KpxCapacityMonthlyByGenerationTypeHomeFileLoader()
    fp_fuel = _file_in("kpx_capacity_monthly_by_fuel_home_file")
    fp_gen = _file_in("kpx_capacity_monthly_by_generation_type_home_file")
    df_fuel = fuel_loader.parse_file(fp_fuel)
    df_gen = gen_loader.parse_file(fp_gen)
    for df in (df_fuel, df_gen):
        for col in ("member_type", "dispatch_type", "business_type", "region"):
            non_total = df.loc[df[col] != "합계", col].unique()
            assert len(non_total) == 0, (
                f"Non-합계 rows leaked through {col} filter: {non_total[:5]}"
            )


def test_transaction_volume_hour_24_interval_end_consistency():
    """trade_hour 1..24 must produce a strictly monotonic interval_end mapping,
    where hour=24 of date d equals (d+1) 00:00 — consistent with the KPX
    SMP hourly convention.
    """
    loader = KpxTransactionVolumeHourlyByFuelFileLoader()
    fp = _file_in("kpx_transaction_volume_hourly_by_fuel_file")
    df = loader.parse_file(fp)
    sample = df[(df["trade_date"] == pd.Timestamp("2023-11-01"))
                & (df["fuel_type"] == "nuclear")].sort_values("trade_hour")
    if sample.empty:
        pytest.skip("No nuclear rows on 2023-11-01 in the sample file")
    # hour=1 → 2023-11-01 01:00; hour=24 → 2023-11-02 00:00
    h1 = sample[sample["trade_hour"] == 1].iloc[0]
    h24 = sample[sample["trade_hour"] == 24].iloc[0]
    assert h1["interval_end"] == pd.Timestamp("2023-11-01 01:00:00")
    assert h24["interval_end"] == pd.Timestamp("2023-11-02 00:00:00"), (
        "trade_hour=24 must roll over into the next day's 00:00 "
        "(matches the SMP interval_end convention)."
    )


# ===========================================================================
# 3. Feature-builder integration tests
# ===========================================================================


def test_monthly_feature_builder_adds_capacity_features():
    bm = importlib.reload(importlib.import_module("src.features.build_monthly"))
    feats, info = bm.build_smp_monthly_features(
        area="mainland", horizon_months=1,
        include_capacity=True, include_transaction=False,
    )
    # All optional capacity features carry _lag_1m (monthly) or _lag_1y (yearly)
    # to satisfy the no-future-leakage contract.
    expected = [
        "capacity_fuel_nuclear_mw_lag_1m", "capacity_fuel_lng_mw_lag_1m",
        "capacity_fuel_renewable_total_mw_lag_1m", "capacity_fuel_total_mw_lag_1m",
        "capacity_fuel_nuclear_share_lag_1m", "capacity_fuel_renewable_total_share_lag_1m",
        "capacity_type_nuclear_mw_lag_1m", "capacity_type_combined_cycle_total_mw_lag_1m",
        "capacity_type_renewable_mw_lag_1m", "capacity_type_total_mw_lag_1m",
        "capacity_yearly_nuclear_mw_lag_1y", "capacity_yearly_total_mw_lag_1y",
    ]
    for col in expected:
        assert col in feats.columns, f"Missing optional capacity feature: {col}"
    # capacity_fuel_* must NOT collide with capacity_type_* (different taxonomies)
    fuel_cols = {c for c in feats.columns if c.startswith("capacity_fuel_")}
    type_cols = {c for c in feats.columns if c.startswith("capacity_type_")}
    assert fuel_cols & type_cols == set()
    # side-info must record both the broadcast and lag flags
    yearly_info = info["optional_join_info"]["capacity_yearly"]
    assert yearly_info["feature_origin_frequency"] == "yearly_broadcast"
    assert yearly_info["lag_applied"] == "1y"
    assert info["optional_join_info"]["capacity_fuel"]["lag_applied"] == "1m"
    assert info["optional_join_info"]["capacity_type"]["lag_applied"] == "1m"


def test_monthly_feature_builder_adds_transaction_features():
    bm = importlib.reload(importlib.import_module("src.features.build_monthly"))
    feats, info = bm.build_smp_monthly_features(
        area="mainland", horizon_months=1,
        include_capacity=False, include_transaction=True,
    )
    expected = [
        "transaction_volume_nuclear_mwh_lag_1m", "transaction_volume_lng_mwh_lag_1m",
        "transaction_volume_total_mwh_lag_1m",
        "transaction_volume_renewable_share_lag_1m",
        "transaction_amount_nuclear_krw_lag_1m", "transaction_amount_total_krw_lag_1m",
    ]
    for col in expected:
        assert col in feats.columns, f"Missing optional transaction feature: {col}"
    assert info["optional_join_info"]["transaction_volume"][
        "market_participating_generators_only"
    ] is True
    # Monthly aggregation must key on trade_date so trade_hour=24 doesn't
    # spill into the next month under interval_end semantics.
    assert info["optional_join_info"]["transaction_volume"][
        "monthly_aggregation_key"
    ] == "trade_date"
    assert info["optional_join_info"]["transaction_volume"]["lag_applied"] == "1m"


def test_transaction_price_not_created_without_overlap():
    """The real on-disk fixtures have volume=2023-11..12 and amount=2024-12 only;
    no period overlap → market_trade_price_* must NOT be created and the
    info dict must record the warning."""
    bm = importlib.reload(importlib.import_module("src.features.build_monthly"))
    feats, info = bm.build_smp_monthly_features(
        area="mainland", horizon_months=1,
        include_capacity=False, include_transaction=True,
    )
    price_cols = [c for c in feats.columns if c.startswith("market_trade_price_")]
    assert price_cols == [], (
        f"market_trade_price_* leaked despite no period overlap: {price_cols}"
    )
    info_price = info["optional_join_info"]["transaction_price"]
    assert info_price["overlap_months"] == 0
    assert info_price["warning"] and "no_overlap" in info_price["warning"]


def test_transaction_price_created_when_overlap_exists(tmp_path, monkeypatch):
    """Synthetic test: same month in BOTH volume and amount → price feature
    must appear and equal amount/volume/1000."""
    from src.utils import io as io_mod
    class _Stub:
        data_dir = tmp_path / "data"
    monkeypatch.setattr(io_mod, "get_settings", lambda: _Stub())
    bm = importlib.reload(importlib.import_module("src.features.build_monthly"))

    # Build a minimal SMP source so build_smp_monthly_features doesn't fail.
    smp = pd.DataFrame({
        "period_month": pd.date_range("2022-01-01", periods=36, freq="MS"),
        "area": ["mainland"] * 36,
        "smp_krw_per_kwh": [120.0 + i for i in range(36)],
        "source_id": "kpx_smp_monthly_kepco_file",
        "collected_at": pd.Timestamp("2026-05-25"),
        "source_file": "fake_smp.csv",
        "source_priority": 1,
        "source_file_sha256": "a" * 64,
    })
    smp_dir = bm.source_root_dir("kpx_smp_monthly_kepco_file") / "2025" / "01" / "01"
    smp_dir.mkdir(parents=True, exist_ok=True)
    smp.to_parquet(smp_dir / "parsed_test.parquet", index=False)

    # Volume (hourly) — make Jan/Feb/Mar 2023 have nuclear volume = 1000 MWh
    # per hour so monthly sum = 1000 × hours_in_month for nuclear.
    # Three months so that after lag_1m we still have >=2 months of overlap.
    hours_jan = pd.date_range("2023-01-01 01:00", "2023-02-01 00:00", freq="h")
    hours_feb = pd.date_range("2023-02-01 01:00", "2023-03-01 00:00", freq="h")
    hours_mar = pd.date_range("2023-03-01 01:00", "2023-04-01 00:00", freq="h")
    rows = []
    for h in list(hours_jan) + list(hours_feb) + list(hours_mar):
        rows.append({
            "source_id": "kpx_transaction_volume_hourly_by_fuel_file",
            "trade_date": h.normalize() - (pd.Timedelta(hours=1) if h.hour == 0 else pd.Timedelta(0)),
            "trade_hour": 24 if h.hour == 0 else h.hour,
            "interval_end": h,
            "fuel_type_raw": "원자력",
            "fuel_type": "nuclear",
            "transaction_volume_mwh": 1000.0,
            "collected_at": pd.Timestamp("2026-05-25"),
            "source_file": "vol.csv",
            "source_priority": 1,
            "source_file_sha256": "v" * 64,
        })
    vol_df = pd.DataFrame(rows)
    vol_dir = bm.source_root_dir("kpx_transaction_volume_hourly_by_fuel_file") / "2023" / "02" / "28"
    vol_dir.mkdir(parents=True, exist_ok=True)
    vol_df.to_parquet(vol_dir / "parsed_test.parquet", index=False)

    # Amount (daily) — same Jan/Feb/Mar 2023, nuclear amount per day = 1e9 KRW
    days_jan = pd.date_range("2023-01-01", "2023-01-31", freq="D")
    days_feb = pd.date_range("2023-02-01", "2023-02-28", freq="D")
    days_mar = pd.date_range("2023-03-01", "2023-03-31", freq="D")
    amt_rows = []
    for d in list(days_jan) + list(days_feb) + list(days_mar):
        amt_rows.append({
            "source_id": "kpx_transaction_amount_daily_by_fuel_file",
            "trade_date": d,
            "fuel_type_raw": "원자력",
            "fuel_type": "nuclear",
            "transaction_amount_krw": 1_000_000_000.0,
            "collected_at": pd.Timestamp("2026-05-25"),
            "source_file": "amt.csv",
            "source_priority": 1,
            "source_file_sha256": "x" * 64,
        })
    amt_df = pd.DataFrame(amt_rows)
    amt_dir = bm.source_root_dir("kpx_transaction_amount_daily_by_fuel_file") / "2023" / "02" / "28"
    amt_dir.mkdir(parents=True, exist_ok=True)
    amt_df.to_parquet(amt_dir / "parsed_test.parquet", index=False)

    feats, info = bm.build_smp_monthly_features(
        area="mainland", horizon_months=1,
        include_settlement=False, include_capacity=False, include_transaction=True,
    )
    assert "market_trade_price_nuclear_krw_per_kwh_lag_1m" in feats.columns
    # After _exogenous_lag_1m, the row at period_month=Feb 2023 carries the
    # AGGREGATE-OVER-Jan value (M-1). So check Feb's price against Feb's
    # lagged volume/amount columns.
    feb_row = feats.loc[
        feats["period_month"] == pd.Timestamp("2023-02-01")
    ].iloc[0]
    expected = (
        feb_row["transaction_amount_nuclear_krw_lag_1m"]
        / feb_row["transaction_volume_nuclear_mwh_lag_1m"]
        / 1000.0
    )
    assert feb_row["market_trade_price_nuclear_krw_per_kwh_lag_1m"] == pytest.approx(expected)
    # Overlap (after _exogenous_lag_1m's reindex) should be >= 2 months.
    assert info["optional_join_info"]["transaction_price"]["overlap_months"] >= 2


def test_optional_features_carry_lag_suffix():
    """Hard contract: every optional feature column added by capacity/transaction
    must carry an explicit _lag_1m or _lag_1y suffix. Current-month values would
    violate the no-future-leakage rule documented at the top of build_monthly.py.
    """
    bm = importlib.reload(importlib.import_module("src.features.build_monthly"))
    feats, _ = bm.build_smp_monthly_features(
        area="mainland", horizon_months=1,
        include_capacity=True, include_transaction=True,
    )
    optional_prefixes = ("capacity_fuel_", "capacity_type_", "capacity_yearly_",
                         "transaction_volume_", "transaction_amount_",
                         "market_trade_price_")
    offenders = [
        c for c in feats.columns
        if any(c.startswith(p) for p in optional_prefixes)
        and not (c.endswith("_lag_1m") or c.endswith("_lag_1y"))
    ]
    assert offenders == [], (
        f"Optional features missing _lag_1m/_lag_1y suffix (potential future "
        f"leakage): {offenders}"
    )


def test_transaction_volume_monthly_aggregation_uses_trade_date(tmp_path, monkeypatch):
    """trade_hour=24 of the last day of a month must stay in that month's
    aggregate (trade_date semantics) — under interval_end semantics it would
    silently spill into the next month and bias the monthly volume."""
    from src.utils import io as io_mod
    class _Stub:
        data_dir = tmp_path / "data"
    monkeypatch.setattr(io_mod, "get_settings", lambda: _Stub())
    bm = importlib.reload(importlib.import_module("src.features.build_monthly"))

    # SMP fixture — needs >=13 months so smp_lag_12m + target_t+1 dropna
    # leaves at least Feb 2023 in baseline rows.
    smp = pd.DataFrame({
        "period_month": pd.date_range("2022-01-01", periods=24, freq="MS"),
        "area": ["mainland"] * 24,
        "smp_krw_per_kwh": [100.0 + i for i in range(24)],
        "source_id": "kpx_smp_monthly_kepco_file",
        "collected_at": pd.Timestamp("2026-05-25"),
        "source_file": "fake_smp.csv",
        "source_priority": 1,
        "source_file_sha256": "a" * 64,
    })
    smp_dir = bm.source_root_dir("kpx_smp_monthly_kepco_file") / "2025" / "01" / "01"
    smp_dir.mkdir(parents=True, exist_ok=True)
    smp.to_parquet(smp_dir / "parsed_test.parquet", index=False)

    # Volume rows:
    #   - Jan 31, hour=23 → interval_end 2023-01-31 23:00 (Jan under both)
    #   - Jan 31, hour=24 → interval_end 2023-02-01 00:00 (Feb under interval_end!)
    # + A small Feb 1 hour=1 row so _exogenous_lag_1m's reindex spans both
    #   months (otherwise the lag would have nothing to shift into Feb).
    # Under correct trade_date aggregation, the FIRST TWO rows must land in
    # Jan 2023, and the Feb hour=1 row stays in Feb.
    vol_df = pd.DataFrame([
        {
            "source_id": "kpx_transaction_volume_hourly_by_fuel_file",
            "trade_date": pd.Timestamp("2023-01-31"),
            "trade_hour": 23,
            "interval_end": pd.Timestamp("2023-01-31 23:00"),
            "fuel_type_raw": "원자력", "fuel_type": "nuclear",
            "transaction_volume_mwh": 500.0,
            "collected_at": pd.Timestamp("2026-05-25"),
            "source_file": "vol.csv", "source_priority": 1,
            "source_file_sha256": "v" * 64,
        },
        {
            "source_id": "kpx_transaction_volume_hourly_by_fuel_file",
            "trade_date": pd.Timestamp("2023-01-31"),
            "trade_hour": 24,
            "interval_end": pd.Timestamp("2023-02-01 00:00"),
            "fuel_type_raw": "원자력", "fuel_type": "nuclear",
            "transaction_volume_mwh": 700.0,
            "collected_at": pd.Timestamp("2026-05-25"),
            "source_file": "vol.csv", "source_priority": 1,
            "source_file_sha256": "v" * 64,
        },
        {
            "source_id": "kpx_transaction_volume_hourly_by_fuel_file",
            "trade_date": pd.Timestamp("2023-02-01"),
            "trade_hour": 1,
            "interval_end": pd.Timestamp("2023-02-01 01:00"),
            "fuel_type_raw": "원자력", "fuel_type": "nuclear",
            "transaction_volume_mwh": 999.0,
            "collected_at": pd.Timestamp("2026-05-25"),
            "source_file": "vol.csv", "source_priority": 1,
            "source_file_sha256": "v" * 64,
        },
    ])
    vol_dir = bm.source_root_dir("kpx_transaction_volume_hourly_by_fuel_file") / "2023" / "01" / "31"
    vol_dir.mkdir(parents=True, exist_ok=True)
    vol_df.to_parquet(vol_dir / "parsed_test.parquet", index=False)

    feats, _ = bm.build_smp_monthly_features(
        area="mainland", horizon_months=1,
        include_settlement=False, include_capacity=False, include_transaction=True,
    )
    # After lag_1m, the Jan 2023 monthly volume (500+700=1200 MWh) surfaces at
    # period_month=Feb 2023 (the lag_1m shifts the index by 1 month).
    feb_row = feats.loc[
        feats["period_month"] == pd.Timestamp("2023-02-01")
    ].iloc[0]
    assert feb_row["transaction_volume_nuclear_mwh_lag_1m"] == pytest.approx(1200.0), (
        f"trade_hour=24 of Jan 31 must aggregate into Jan's monthly volume "
        f"(trade_date semantics). Got "
        f"{feb_row['transaction_volume_nuclear_mwh_lag_1m']} — likely an "
        f"interval_end-based aggregation leaked the hour into Feb."
    )


def test_monthly_feature_join_preserves_row_count():
    """Optional capacity/transaction joins must NOT change the SMP row count."""
    bm = importlib.reload(importlib.import_module("src.features.build_monthly"))
    feats_baseline, _ = bm.build_smp_monthly_features(
        area="mainland", horizon_months=1,
        include_capacity=False, include_transaction=False,
    )
    feats_with_opt, _ = bm.build_smp_monthly_features(
        area="mainland", horizon_months=1,
        include_capacity=True, include_transaction=True,
    )
    assert len(feats_baseline) == len(feats_with_opt), (
        f"Optional joins changed row count: baseline={len(feats_baseline)} "
        f"with_optional={len(feats_with_opt)}"
    )


# ===========================================================================
# 4. Source metadata / unit assertions
# ===========================================================================


def test_source_metadata_units_are_correct():
    """sources.yaml metadata must declare correct units for each new source."""
    expected_units = {
        "kpx_capacity_yearly_by_energy_source_home_file": "MW",
        "kpx_capacity_monthly_by_generation_type_home_file": "MW",
        "kpx_capacity_monthly_by_fuel_home_file": "MW",
        "kpx_transaction_volume_hourly_by_fuel_file": "MWh",
        "kpx_transaction_amount_daily_by_fuel_file": "KRW",
    }
    for source_id, unit in expected_units.items():
        config = get_source_config(source_id)
        assert config["unit"] == unit, (
            f"{source_id} unit metadata wrong: expected={unit!r}, got={config['unit']!r}"
        )
    # Amount file vendor header is '전력거래금액(원)' → must map to transaction_amount_krw
    amount_config = get_source_config("kpx_transaction_amount_daily_by_fuel_file")
    assert amount_config["column_mapping"]["transaction_amount_krw"] == "전력거래금액(원)"
