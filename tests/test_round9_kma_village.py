"""Round-9 tests: KMA Vilage short-term forecast collector + grid mapping loader.

The KMA mid-term forecast service (round 8) is still pending approval, but
the user confirmed VilageFcstInfoService_2.0 returns NORMAL_SERVICE. This
round wires that working endpoint plus the official grid (nx, ny) ↔
admin-region mapping spreadsheet.
"""

from __future__ import annotations

from datetime import datetime

import pandas as pd
import pytest

from src.collectors.base import FetchResult
from src.collectors.kma_village_fcst import (
    DEFAULT_REGIONS,
    KmaVilageFcstCollector,
    _latest_base_datetime,
)
from src.collectors.kpx_files import (
    KMA_GRID_XY_MAPPING_COLUMNS,
    KmaGridXyMappingFileLoader,
)


# ---------------------------------------------------------------------------
# 1. _latest_base_datetime picks the most-recent fully-published slot
# ---------------------------------------------------------------------------

def test_latest_base_datetime_after_publish_window_uses_current_slot():
    """At 14:30 KST, the 14:10 publication of base_time=1400 is live."""
    now = datetime(2026, 5, 26, 14, 30)
    bd, bt = _latest_base_datetime(now)
    assert bd == "20260526"
    assert bt == "1400"


def test_latest_base_datetime_inside_publish_window_uses_prior_slot():
    """At 14:05 KST, the 14:10 publication isn't live yet — must use 11:00 slot."""
    now = datetime(2026, 5, 26, 14, 5)
    bd, bt = _latest_base_datetime(now)
    assert bd == "20260526"
    assert bt == "1100"


def test_latest_base_datetime_before_first_slot_rolls_to_previous_day():
    """At 01:30 KST (before any of today's slots), use yesterday's 2300 slot."""
    now = datetime(2026, 5, 26, 1, 30)
    bd, bt = _latest_base_datetime(now)
    assert bd == "20260525"
    assert bt == "2300"


# ---------------------------------------------------------------------------
# 2. Vilage forecast parser
# ---------------------------------------------------------------------------

def test_village_fcst_parser_returns_canonical_schema():
    items = [
        {
            "_region_name": "Seoul",
            "baseDate": "20260526", "baseTime": "0200",
            "nx": 60, "ny": 127,
            "fcstDate": "20260526", "fcstTime": "0300",
            "category": "TMP", "fcstValue": "21",
        },
        {
            "_region_name": "Seoul",
            "baseDate": "20260526", "baseTime": "0200",
            "nx": 60, "ny": 127,
            "fcstDate": "20260526", "fcstTime": "1500",
            "category": "TMX", "fcstValue": "27",
        },
        # Non-numeric category (SKY=1/3/4) — numeric coercion must still produce a float
        {
            "_region_name": "Seoul",
            "baseDate": "20260526", "baseTime": "0200",
            "nx": 60, "ny": 127,
            "fcstDate": "20260526", "fcstTime": "0900",
            "category": "SKY", "fcstValue": "1",
        },
    ]
    fr = FetchResult(raw_text="", raw_suffix=".json",
                     parsed={"items": items}, request={})
    df = KmaVilageFcstCollector().parse(fr)
    expected_cols = {
        "source_id", "region_name", "base_date", "base_time", "nx", "ny",
        "fcst_date", "fcst_time", "category", "fcst_value", "fcst_value_numeric",
    }
    assert expected_cols.issubset(df.columns)
    tmp_row = df[df["category"] == "TMP"].iloc[0]
    assert tmp_row["fcst_value_numeric"] == pytest.approx(21.0)
    assert tmp_row["region_name"] == "Seoul"
    assert tmp_row["nx"] == 60 and tmp_row["ny"] == 127
    # The SKY=1 row preserves the original string AND the parsed numeric
    sky_row = df[df["category"] == "SKY"].iloc[0]
    assert sky_row["fcst_value"] == "1"
    assert sky_row["fcst_value_numeric"] == 1.0


def test_village_fcst_parser_handles_empty_response():
    fr = FetchResult(raw_text="", raw_suffix=".json", parsed={"items": []}, request={})
    df = KmaVilageFcstCollector().parse(fr)
    assert df.empty
    assert "fcst_value_numeric" in df.columns


# ---------------------------------------------------------------------------
# 3. KMA grid mapping loader
# ---------------------------------------------------------------------------

def test_kma_grid_mapping_loader_real_file_resolves_seoul_busan_daegu():
    """Grid mapping for the three default city centres must match the
    DEFAULT_REGIONS the village_fcst collector uses. If KMA renames a
    region or revises grid coordinates, this test catches it before the
    forecast collector silently asks for the wrong cell."""
    from pathlib import Path
    fp = Path("data/raw/manual_or_filedata/kma_grid_xy_mapping_file/kma_grid_xy_mapping.xlsx")
    if not fp.exists():
        pytest.skip(f"Grid mapping spreadsheet not present at {fp}")
    loader = KmaGridXyMappingFileLoader()
    df = loader.parse_file(fp)
    assert set(KMA_GRID_XY_MAPPING_COLUMNS).issubset(df.columns)
    # Coordinates the round-9 collector defaults use.
    expected = {
        "Seoul": (60, 127, "서울특별시"),
        "Busan": (98, 76, "부산광역시"),
        "Daegu": (89, 90, "대구광역시"),
    }
    for region in DEFAULT_REGIONS:
        nm = region["name"]
        if nm not in expected:
            continue
        exp_nx, exp_ny, sido = expected[nm]
        # Find at least one row for that sido with matching (nx, ny)
        match = df[
            (df["level1"] == sido)
            & (df["nx"] == exp_nx)
            & (df["ny"] == exp_ny)
        ]
        assert not match.empty, (
            f"DEFAULT_REGIONS has {nm}=(nx={exp_nx}, ny={exp_ny}) but "
            f"grid mapping has no row with that combination for {sido}"
        )
