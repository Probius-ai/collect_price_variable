"""Round-2 file loader tests.

Per spec:
    * test_monthly_smp_loader_csv
    * test_monthly_smp_loader_multirow_header
    * test_settlement_wide_to_long
    * test_rec_weekly_loader
    * test_no_filename_based_source_assumption
    * test_quarantine_filename_content_mismatch
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest


# ----------------------------- Fixtures ------------------------------------

@pytest.fixture
def isolated_paths(monkeypatch, tmp_path):
    """Route data/ writes and the sources.yaml read to a temp directory.

    We DON'T overwrite the project sources.yaml — we copy it next to the
    tmp_path config and point CONFIG_DIR at the copy. That keeps real config
    intact and lets tests freely edit per-source blocks.
    """
    from src.config import settings as settings_mod
    from src.utils import io as io_mod

    fake_data = tmp_path / "data"
    fake_data.mkdir()
    fake_outputs = tmp_path / "outputs"
    fake_outputs.mkdir()
    fake_cfg = tmp_path / "config"
    fake_cfg.mkdir()
    real_cfg = Path(__file__).resolve().parents[1] / "src" / "config" / "sources.yaml"
    (fake_cfg / "sources.yaml").write_text(
        real_cfg.read_text(encoding="utf-8"), encoding="utf-8"
    )

    monkeypatch.setattr(settings_mod, "CONFIG_DIR", fake_cfg)
    settings_mod.load_sources_config.cache_clear()
    settings_mod.get_settings.cache_clear()

    class _Stub:
        data_dir = fake_data
        outputs_dir = fake_outputs
    # Patch BOTH the io module's reference AND the settings module's cached
    # get_settings so any caller (file_loader, quarantine, dq_report, …) sees
    # the temp data_dir instead of the real project path.
    monkeypatch.setattr(io_mod, "get_settings", lambda: _Stub())
    monkeypatch.setattr(settings_mod, "get_settings", lambda: _Stub())
    return tmp_path, fake_cfg


def _drop_file(tmp: Path, source: str, name: str, content: str | bytes, *, encoding: str | None = "utf-8") -> Path:
    drop = tmp / "data" / "raw" / "manual_or_filedata" / source
    drop.mkdir(parents=True, exist_ok=True)
    path = drop / name
    if isinstance(content, bytes):
        path.write_bytes(content)
    else:
        path.write_text(content, encoding=encoding or "utf-8")
    return path


# ----------------------------- Tests ---------------------------------------

def test_monthly_smp_loader_csv(isolated_paths):
    """KEPCO official single-CSV → long format with mainland/jeju/integrated rows."""
    tmp, _ = isolated_paths
    csv = (
        "년도,월,육지계통한계가격,제주계통한계가격,통합계통한계가격\n"
        "2025,08,117.39,118.46,117.4\n"
        "2025,07,120.39,121.13,120.39\n"
    )
    drop = _drop_file(
        tmp, "kpx_smp_monthly_kepco_file", "smp_kepco_sample.csv",
        csv.encode("cp949"),  # vendor encoding
    )

    from src.collectors.kpx_files import KpxSmpMonthlyKepcoFileLoader
    loader = KpxSmpMonthlyKepcoFileLoader()
    run = loader.load_one(drop)
    assert run.status == "ok"
    parsed = pd.read_parquet(run.paths.parsed_dataframe)

    expected_cols = {"source_id", "period_month", "area", "smp_krw_per_kwh",
                     "collected_at", "source_file"}
    assert expected_cols.issubset(parsed.columns)
    # 2 months × 3 areas = 6 rows
    assert len(parsed) == 6
    assert set(parsed["area"]) == {"mainland", "jeju", "integrated"}
    # Spot-check the August integrated value
    aug_integ = parsed[
        (parsed["period_month"] == pd.Timestamp("2025-08-01"))
        & (parsed["area"] == "integrated")
    ].iloc[0]
    assert aug_integ["smp_krw_per_kwh"] == 117.4

    meta = json.loads(run.paths.metadata.read_text())
    assert meta["source_id"] == "kpx_smp_monthly_kepco_file"
    assert meta["source_priority"] == 1
    assert meta["frequency"] == "monthly"
    assert meta["unit"] == "KRW/kWh"


def test_monthly_smp_loader_multirow_header(isolated_paths):
    """HOME EPSIS export: row 0 has merged 'SMP' header, row 1 has area labels."""
    tmp, _ = isolated_paths
    # mimic the merged-cell vendor CSV: row 0 only writes 'SMP' once
    csv = (
        "기간,SMP,,,BLMP\n"
        ",육지,제주,통합,\n"
        "2026/04,118.94,117.24,118.92,0\n"
        "2026/03,110.03,106.77,109.99,0\n"
    )
    drop = _drop_file(
        tmp, "kpx_smp_monthly_home_avg_file", "home_avg_smp.csv",
        csv.encode("cp949"),
    )

    from src.collectors.kpx_files import KpxSmpMonthlyHomeAvgFileLoader
    loader = KpxSmpMonthlyHomeAvgFileLoader()
    run = loader.load_one(drop)
    parsed = pd.read_parquet(run.paths.parsed_dataframe)
    # 2 months × 3 areas
    assert len(parsed) == 6
    assert set(parsed["area"]) == {"mainland", "jeju", "integrated"}
    apr_main = parsed[
        (parsed["period_month"] == pd.Timestamp("2026-04-01"))
        & (parsed["area"] == "mainland")
    ].iloc[0]
    assert apr_main["smp_krw_per_kwh"] == 118.94


def test_settlement_wide_to_long(isolated_paths):
    """Wide-by-fuel settlement CSV → long with canonical fuel_type values."""
    tmp, _ = isolated_paths
    # Vendor header includes renewable subcategories — those must be DROPPED.
    csv = (
        "기간,원자력,석탄_유연탄,석탄_무연탄,석탄_계,유류,LNG,양수,"
        "신재생_연료전지,신재생_석탄가스화,신재생_태양,신재생_풍력,"
        "신재생_수력,신재생_해양,신재생_바이오,신재생_폐기물,"
        "신재생_계,기타,합계\n"
        "2026/04,96.4,132.4,0.0,131.0,200.1,138.2,90.0,"
        "300.0,250.0,200.0,180.0,160.0,140.0,170.0,150.0,"
        "190.0,99.0,125.5\n"
    )
    drop = _drop_file(
        tmp, "kpx_settlement_monthly_file", "settlement.csv",
        ("﻿" + csv).encode("utf-8"),  # utf-8-sig
    )

    from src.collectors.kpx_files import KpxSettlementMonthlyFileLoader
    loader = KpxSettlementMonthlyFileLoader()
    run = loader.load_one(drop)
    parsed = pd.read_parquet(run.paths.parsed_dataframe)

    expected_cols = {"source_id", "period_month", "fuel_type",
                     "settlement_unit_price_krw_per_kwh", "collected_at", "source_file"}
    assert expected_cols.issubset(parsed.columns)

    # Only the 10 canonical fuel_types should be present.
    expected_fuels = {
        "nuclear", "coal_bituminous", "coal_anthracite", "coal_total",
        "oil", "lng", "pumped_storage", "renewable", "other", "total",
    }
    assert set(parsed["fuel_type"]) == expected_fuels

    # Spot check
    ren = parsed[parsed["fuel_type"] == "renewable"].iloc[0]
    assert ren["settlement_unit_price_krw_per_kwh"] == 190.0

    # The dropped renewable subcategories must be recorded in metadata.
    meta = json.loads(run.paths.metadata.read_text())
    dropped = set(meta["dropped_columns"])
    assert {"신재생_연료전지", "신재생_태양", "신재생_풍력"}.issubset(dropped)


def test_rec_weekly_loader(isolated_paths):
    """REC weekly CSV (cp949) → long with full canonical price schema."""
    tmp, _ = isolated_paths
    csv = (
        "거래일,체결 수량,평균가,체결총액,시작가,종가,기준가,최고가,최저가\n"
        "2017-03-28,2568,119012,305624400,119900,120000,137600,120000,117500\n"
        "2017-03-30,11513,119546,1376333900,120000,120100,120000,121300,115000\n"
    )
    drop = _drop_file(
        tmp, "kpx_rec_weekly_file", "rec_weekly.csv",
        csv.encode("cp949"),
    )

    from src.collectors.kpx_files import KpxRecWeeklyFileLoader
    loader = KpxRecWeeklyFileLoader()
    run = loader.load_one(drop)
    parsed = pd.read_parquet(run.paths.parsed_dataframe)

    expected_cols = {
        "source_id", "trade_date", "rec_volume", "avg_price_krw", "total_amount_krw",
        "open_price_krw", "close_price_krw", "base_price_krw",
        "high_price_krw", "low_price_krw", "collected_at", "source_file",
    }
    assert expected_cols.issubset(parsed.columns)
    assert len(parsed) == 2
    first = parsed.iloc[0]
    assert first["trade_date"] == pd.Timestamp("2017-03-28")
    assert first["rec_volume"] == 2568
    assert first["avg_price_krw"] == 119012
    assert first["total_amount_krw"] == 305624400

    meta = json.loads(run.paths.metadata.read_text())
    assert meta["frequency"] == "weekly_file_trading_day"


def test_no_filename_based_source_assumption(isolated_paths):
    """KEPCO loader must REJECT a 'looks-like-SMP'-named CSV whose content
    is actually yearly generation MWh by fuel.

    This regression test ensures we never silently load the misnamed
    HOME_..._SMP.csv into the monthly-SMP pipeline just because the
    *filename* contains "SMP".
    """
    tmp, _ = isolated_paths
    # Header looks like 'generation by fuel'. No 년도/월 columns at all.
    csv = (
        "연도,수력,기력,복합화력,원자력,신재생,총계\n"
        "2024,8977549,162326738,123240461,188754102,56537104,594266336\n"
    )
    drop = _drop_file(
        tmp, "kpx_smp_monthly_kepco_file", "HOME_가중평균SMP.csv",
        csv.encode("cp949"),
    )

    from src.collectors.file_loader import FileLoaderError
    from src.collectors.kpx_files import KpxSmpMonthlyKepcoFileLoader
    loader = KpxSmpMonthlyKepcoFileLoader()
    with pytest.raises(FileLoaderError, match="missing in the actual file"):
        loader.load_one(drop)


def test_zero_smp_is_treated_as_missing(isolated_paths):
    """Regression: KPX vendor uses literal 0.00 as a 'no published value'
    marker (e.g. KEPCO 2015-06 single-month gap; HOME 2001-2009 mainland/jeju
    when only the legacy integrated value was tracked). These rows must NOT
    appear in the training-ready output — published SMP has never been
    0 KRW/kWh in any month since KPX launched in 2001 (min ≈ 30 KRW/kWh).

    Without the fix, the monthly SMP pipeline emitted ~210 zero-price rows
    into feature parquets and downstream training tables.
    """
    tmp, _ = isolated_paths
    csv = (
        "년도,월,육지계통한계가격,제주계통한계가격,통합계통한계가격\n"
        "2015,07,66.79,93.81,67.06\n"
        "2015,06,0.00,0.00,0.00\n"   # vendor "no data" — must be dropped
        "2015,05,68.72,75.53,68.78\n"
        "2010,01,0.00,0.00,45.0\n"   # mainland/jeju missing (pre-split era)
    )
    drop = _drop_file(
        tmp, "kpx_smp_monthly_kepco_file", "smp_with_zeros.csv",
        csv.encode("cp949"),
    )

    from src.collectors.kpx_files import KpxSmpMonthlyKepcoFileLoader
    loader = KpxSmpMonthlyKepcoFileLoader()
    run = loader.load_one(drop)
    parsed = pd.read_parquet(run.paths.parsed_dataframe)

    # 2015-06 all-zero row dropped entirely (3 areas × 1 month = 3 rows lost).
    # 2010-01 keeps only the integrated row (mainland/jeju zeros dropped).
    # 2015-05 and 2015-07 keep all 3 areas. So 3 + 3 + 1 = 7 valid rows.
    assert len(parsed) == 7
    assert (parsed["smp_krw_per_kwh"] > 0).all(), "no zero rows must remain"
    # Specifically: no row exists for 2015-06.
    assert not (parsed["period_month"] == pd.Timestamp("2015-06-01")).any()
    # 2010-01 keeps only integrated.
    jan_2010 = parsed[parsed["period_month"] == pd.Timestamp("2010-01-01")]
    assert list(jan_2010["area"]) == ["integrated"]
    assert jan_2010.iloc[0]["smp_krw_per_kwh"] == 45.0

    # Metadata must record the zero-coerced counts so the DQ report can flag.
    meta = json.loads(run.paths.metadata.read_text())
    coerced = meta["zero_coerced_rows"]
    # 2015-06 contributes 1 to each of mainland/jeju/integrated; 2010-01
    # contributes 1 each to mainland and jeju.
    assert coerced["mainland"] == 2
    assert coerced["jeju"] == 2
    assert coerced["integrated"] == 1


def test_quarantine_filename_content_mismatch(isolated_paths):
    """The quarantine flow moves misnamed files aside and writes a manifest."""
    tmp, _ = isolated_paths
    # Drop two misnamed files at the location the quarantine flow scans.
    for fname in (
        "HOME_전력거래_계통한계가격_가중평균SMP.csv",
        "HOME_전력거래_계통한계가격_가중평균SMP (2).csv",
    ):
        _drop_file(tmp, "incoming", fname, "anything", encoding="utf-8")

    from src.collectors.quarantine import (
        quarantine_dir,
        quarantine_filename_content_mismatch,
    )
    entries = quarantine_filename_content_mismatch("kpx_generation_yearly")
    assert len(entries) == 2
    for e in entries:
        assert e.status == "pending_schema_verification"
        assert e.reason == "filename_content_mismatch"
        assert Path(e.quarantined_path).exists()
        # The original drop should have been moved (no longer at source).
        assert not Path(e.original_path).exists()

    # A manifest JSON must exist in the quarantine root.
    manifests = list(quarantine_dir().rglob("manifest_*.json"))
    assert manifests, "expected a quarantine manifest"
    payload = json.loads(manifests[0].read_text(encoding="utf-8"))
    assert {e["original_path"] for e in payload} == {e.original_path for e in entries}
