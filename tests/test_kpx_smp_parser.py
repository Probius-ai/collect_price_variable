"""Parser tests for KpxSmpCollector that don't touch the network.

We patch the source config + a fetched FetchResult to make sure that:

1. When column_mapping still has TBD entries, parsing refuses to invent.
2. When mapping is filled, the parser renames the vendor keys correctly,
   constructs interval_start/end with the 1..24 convention, and validates.
"""

from __future__ import annotations

import json

import pytest

from src.collectors.base import CollectorError, FetchResult
from src.collectors.kpx_smp import KpxSmpCollector


VENDOR_PAYLOAD = {
    "response": {
        "header": {"resultCode": "00", "resultMsg": "NORMAL_SERVICE"},
        "body": {
            "items": {
                "item": [
                    {
                        "areaNm": "mainland",
                        "trDay": "20240301",
                        "trHour": "1",
                        "smpPrice": "112.34",
                        "demandFcst": "61200.0",
                    },
                    {
                        "areaNm": "mainland",
                        "trDay": "20240301",
                        "trHour": "24",
                        "smpPrice": "120.10",
                        "demandFcst": "58000.0",
                    },
                ]
            }
        },
    }
}


def _fetch_result() -> FetchResult:
    raw_text = json.dumps(VENDOR_PAYLOAD)
    return FetchResult(
        raw_text=raw_text,
        raw_suffix=".json",
        parsed=VENDOR_PAYLOAD,
        request={"url": "https://example/test", "params": {}},
    )


def test_parse_refuses_when_mapping_has_tbd():
    collector = KpxSmpCollector.__new__(KpxSmpCollector)
    collector.source_name = "kpx_smp_day_ahead"
    collector.config = {
        "column_mapping": {"area": "TBD_AFTER_FIRST_RESPONSE"},
        "verified_columns": ["TBD_AFTER_FIRST_RESPONSE"],
    }
    collector.defaults = {}
    collector.log = __import__("logging").getLogger("test")
    with pytest.raises(CollectorError, match="TBD"):
        collector.parse(_fetch_result())


def test_parse_with_filled_mapping_yields_canonical_columns():
    collector = KpxSmpCollector.__new__(KpxSmpCollector)
    collector.source_name = "kpx_smp_day_ahead"
    collector.config = {
        "column_mapping": {
            "area": "areaNm",
            "trade_date": "trDay",
            "trade_hour": "trHour",
            "smp_krw_per_kwh": "smpPrice",
            "demand_forecast_mw": "demandFcst",
        },
        "verified_columns": ["areaNm", "trDay", "trHour", "smpPrice", "demandFcst"],
        "schema_version": 1,
    }
    collector.defaults = {}
    collector.log = __import__("logging").getLogger("test")
    df = collector.parse(_fetch_result())
    assert {"area", "trade_date", "trade_hour", "smp_krw_per_kwh", "demand_forecast_mw",
            "interval_start", "interval_end"}.issubset(df.columns)
    assert df["smp_krw_per_kwh"].tolist() == [112.34, 120.10]
    assert df["interval_end"].iloc[0].hour == 1
    # trade_hour=24 should map to next-day midnight 00:00
    assert df["interval_end"].iloc[1].hour == 0
    assert df["interval_end"].iloc[1].date().isoformat() == "2024-03-02"


def test_iter_items_handles_flat_response():
    from src.collectors.kpx_smp import _iter_items
    flat = [{"foo": 1}, {"foo": 2}]
    assert list(_iter_items(flat)) == flat
