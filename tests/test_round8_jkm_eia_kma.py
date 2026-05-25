"""Round-8 tests: JKM (daily + forward curve) + EIA STEO + KMA mid-term.

Pins:
  * JKM daily loader parses MM/DD/YYYY US dates correctly
  * JKM futures-curve contract-month decoder handles both label + code paths
  * JKM forward-curve features only emit a row when quote_date ≤ info_cutoff
    (leak-safety)
  * JKM daily-volatility features lag_1m discipline + grid extension
  * EIA STEO API config + parser shape (offline parse of canned response)
  * KMA mid-term parser handles the standard items[*] response shape
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from src.collectors.eia_steo import EiaSteoCollector
from src.collectors.kma_mid_temperature import KmaMidTemperatureCollector
from src.collectors.kpx_files import (
    JKM_DAILY_HISTORY_COLUMNS,
    JKM_FUTURES_CURVE_COLUMNS,
    JkmLngDailyHistoryFileLoader,
    JkmLngFuturesCurveFileLoader,
    _parse_jkm_contract_month,
)


# ---------------------------------------------------------------------------
# 1. JKM daily history loader
# ---------------------------------------------------------------------------

def test_jkm_daily_history_loader_parses_us_date_format(tmp_path):
    csv = tmp_path / "jkm_daily_history.csv"
    csv.write_text(
        '"Date","Price","Open","High","Low","Vol.","Change %"\n'
        '"05/22/2026","18.810","18.810","18.810","18.810","","-0.58%"\n'
        '"05/21/2026","18.920","18.920","18.920","18.920","","0.08%"\n'
        '"01/28/2013","18.900","18.900","18.900","18.900","","0.75%"\n',
        encoding="utf-8-sig",
    )
    loader = JkmLngDailyHistoryFileLoader()
    df = loader.parse_file(csv)
    assert set(JKM_DAILY_HISTORY_COLUMNS).issubset(df.columns)
    # Date format MM/DD/YYYY correctly parsed
    assert df["trade_date"].min() == pd.Timestamp("2013-01-28")
    assert df["trade_date"].max() == pd.Timestamp("2026-05-22")
    # Change % strips the '%' sign and is numeric
    assert df["change_pct"].dtype.kind in {"f", "i"}
    assert pd.notna(df.loc[df["trade_date"] == pd.Timestamp("2026-05-22"),
                           "change_pct"].iloc[0])


# ---------------------------------------------------------------------------
# 2. CME forward-curve contract-month decoder
# ---------------------------------------------------------------------------

def test_jkm_contract_month_decoder_prefers_label_when_present():
    """label='JUL 2026' is unambiguous → use it; ignore the (single-digit-year) code."""
    qd = pd.Timestamp("2026-05-24")
    out = _parse_jkm_contract_month("JUL 2026", "JKMN6", qd)
    assert out == pd.Timestamp("2026-07-01")


def test_jkm_contract_month_decoder_falls_back_to_code():
    """Without a label, decode JKMN6 with the quote-date decade anchor."""
    qd = pd.Timestamp("2026-05-24")
    out = _parse_jkm_contract_month("", "JKMN6", qd)
    assert out == pd.Timestamp("2026-07-01")


def test_jkm_contract_month_decoder_rolls_decade_when_naive_is_past():
    """A single-digit year 4 → if quote_date is 2026-05, code JKMN4 naively
    decodes to 2024-07 (past!) — must roll forward to 2034-07."""
    qd = pd.Timestamp("2026-05-24")
    out = _parse_jkm_contract_month("", "JKMN4", qd)
    assert out == pd.Timestamp("2034-07-01")


def test_jkm_futures_curve_loader_emits_canonical_columns(tmp_path):
    csv = tmp_path / "lng_jkm_quotes.csv"
    csv.write_text(
        'Month,Code,Prior_Settle,Volume,Time_CT,Date\n'
        '"JUL 2026","JKMN6",18.810,0,"08:05:57","24 May 2026"\n'
        '"AUG 2026","JKMQ6",18.370,0,"08:06:13","24 May 2026"\n',
        encoding="utf-8",
    )
    loader = JkmLngFuturesCurveFileLoader()
    df = loader.parse_file(csv)
    assert set(JKM_FUTURES_CURVE_COLUMNS).issubset(df.columns)
    assert df["quote_date"].iloc[0] == pd.Timestamp("2026-05-24")
    assert df["contract_month"].iloc[0] == pd.Timestamp("2026-07-01")
    assert df["prior_settle"].iloc[0] == pytest.approx(18.810)


# ---------------------------------------------------------------------------
# 3. JKM forward features — leak-safety canary
# ---------------------------------------------------------------------------

def test_jkm_forward_features_drop_rows_whose_info_cutoff_precedes_snapshot(tmp_path, monkeypatch):
    """A snapshot quote_date=2026-05-24 must NOT be visible at SMP rows with
    info_cutoff (end-of-M) earlier than 2026-05-24. Without this filter,
    we would let post-snapshot information leak into historical SMP
    forecasts."""
    from src.utils import io as io_mod
    class _Stub:
        data_dir = tmp_path / "data"
    monkeypatch.setattr(io_mod, "get_settings", lambda: _Stub())
    import importlib
    bm = importlib.reload(importlib.import_module("src.features.build_monthly"))
    from src.utils.io import source_root_dir

    d = source_root_dir("jkm_lng_futures_curve_file") / "2026" / "05" / "26"
    d.mkdir(parents=True, exist_ok=True)
    snap = pd.DataFrame({
        "source_id": "jkm_lng_futures_curve_file",
        "quote_date": [pd.Timestamp("2026-05-24")] * 2,
        "contract_month_label": ["JUL 2026", "AUG 2026"],
        "contract_month": [pd.Timestamp("2026-07-01"), pd.Timestamp("2026-08-01")],
        "contract_code": ["JKMN6", "JKMQ6"],
        "prior_settle": [18.81, 18.37],
        "volume": [0, 0],
        "quote_time_ct": ["08:05:57", "08:06:13"],
        "collected_at": pd.Timestamp("2026-05-26"),
        "source_file": "lng_jkm_quotes.csv",
        "source_priority": 1,
        "source_file_sha256": "a" * 64,
    })
    snap.to_parquet(d / "parsed_test.parquet", index=False)

    # SMP periods spanning before AND after the snapshot date
    pm = pd.Series(pd.date_range("2026-03-01", "2026-07-01", freq="MS"))
    out, info = bm.build_jkm_forward_features(pm)

    # Rows BEFORE info_cutoff ≥ 2026-05-24 must NOT appear in out
    out_months = set(out["period_month"].dt.strftime("%Y-%m"))
    # 2026-03 (info_cutoff 2026-03-31) < 2026-05-24 → no row
    # 2026-04 (info_cutoff 2026-04-30) < 2026-05-24 → no row
    # 2026-05 (info_cutoff 2026-05-31) ≥ 2026-05-24 → row present
    # 2026-06 (info_cutoff 2026-06-30) ≥ 2026-05-24 → row present
    # 2026-07 (info_cutoff 2026-07-31) ≥ 2026-05-24 → row present
    assert "2026-03" not in out_months, (
        "JKM forward leaks: row 2026-03 emitted using snapshot 2026-05-24"
    )
    assert "2026-04" not in out_months
    assert "2026-05" in out_months
    # At SMP row 2026-06, M+1=2026-07 contract should be the next_month feature
    row_jun = out[out["period_month"] == "2026-06-01"].iloc[0]
    assert row_jun["jkm_forward_next_month_usd_per_mmbtu"] == pytest.approx(18.81)
    assert row_jun["jkm_forward_M_plus_2_usd_per_mmbtu"] == pytest.approx(18.37)


# ---------------------------------------------------------------------------
# 4. JKM daily volatility features — lag_1m + grid extension
# ---------------------------------------------------------------------------

def test_jkm_daily_volatility_lag_1m_uses_only_prior_month(tmp_path, monkeypatch):
    from src.utils import io as io_mod
    class _Stub:
        data_dir = tmp_path / "data"
    monkeypatch.setattr(io_mod, "get_settings", lambda: _Stub())
    import importlib
    bm = importlib.reload(importlib.import_module("src.features.build_monthly"))
    from src.utils.io import source_root_dir

    # 2-month synthetic daily JKM: Jan 2024 mean=10, Feb 2024 mean=20.
    days_jan = pd.date_range("2024-01-01", "2024-01-31", freq="D")
    days_feb = pd.date_range("2024-02-01", "2024-02-29", freq="D")
    rows = []
    for d in days_jan:
        rows.append({"trade_date": d, "close": 10.0})
    for d in days_feb:
        rows.append({"trade_date": d, "close": 20.0})
    daily = pd.DataFrame(rows)
    daily["source_id"] = "jkm_lng_daily_history_file"
    daily["open"] = daily["close"]; daily["high"] = daily["close"]
    daily["low"] = daily["close"]; daily["volume"] = float("nan")
    daily["change_pct"] = 0.0
    daily["collected_at"] = pd.Timestamp("2026-05-26")
    daily["source_file"] = "jkm.csv"
    daily["source_priority"] = 1
    daily["source_file_sha256"] = "b" * 64

    d_dir = source_root_dir("jkm_lng_daily_history_file") / "2024" / "02" / "29"
    d_dir.mkdir(parents=True, exist_ok=True)
    daily.to_parquet(d_dir / "parsed_test.parquet", index=False)

    pm = pd.Series(pd.date_range("2024-01-01", "2024-03-01", freq="MS"))
    out, _ = bm.build_jkm_daily_volatility_features(pm)

    # At row 2024-02: lag_1m = Jan stats (mean=10)
    row_feb = out[out["period_month"] == "2024-02-01"].iloc[0]
    assert row_feb["jkm_daily_mean_lag_1m"] == pytest.approx(10.0)

    # At row 2024-03: lag_1m = Feb stats (mean=20). Row 2024-03 MUST exist
    # thanks to grid extension (raw covers Jan-Feb; output extends to Mar
    # so the latest usable lag is exposed).
    out_months = set(out["period_month"].dt.strftime("%Y-%m"))
    assert "2024-03" in out_months, (
        "Grid extension missing — row 2024-03 dropped, losing lag_1m=Feb"
    )
    row_mar = out[out["period_month"] == "2024-03-01"].iloc[0]
    assert row_mar["jkm_daily_mean_lag_1m"] == pytest.approx(20.0)


# ---------------------------------------------------------------------------
# 5. EIA STEO config + parser
# ---------------------------------------------------------------------------

def test_eia_steo_parser_handles_canned_response():
    """Offline parse — no network — using a tiny EIA response shape."""
    canned_text = """{"response": {"data": [
        {"period": "2024-01", "seriesId": "BREPUUS", "value": "80.5", "unit": "dollars per barrel"},
        {"period": "2024-02", "seriesId": "BREPUUS", "value": "82.0", "unit": "dollars per barrel"},
        {"period": "2024-01", "seriesId": "WTIPUUS", "value": "75.0", "unit": "dollars per barrel"}
    ]}}"""
    from src.collectors.base import FetchResult
    import json
    fr = FetchResult(raw_text=canned_text, raw_suffix=".json",
                     parsed=json.loads(canned_text), request={})
    collector = EiaSteoCollector()
    df = collector.parse(fr)
    assert len(df) == 3
    assert set(df.columns) >= {"source_id", "period_month", "series_id", "value", "units"}
    assert df["period_month"].iloc[0] == pd.Timestamp("2024-01-01")
    assert sorted(df["series_id"].unique()) == ["BREPUUS", "WTIPUUS"]


# ---------------------------------------------------------------------------
# 6. KMA mid-term parser
# ---------------------------------------------------------------------------

def test_kma_mid_temperature_parser_explodes_3_to_10_day_lead_columns():
    """KMA returns one record per (regId, tm_fc) with taMin3..taMin10 +
    taMax3..taMax10. Parser must explode these to one row per (regId,
    forecast_date) with lead_days populated."""
    from src.collectors.base import FetchResult
    item = {
        "_reg_id": "11B10101",
        "_tm_fc": "202605260600",
        "taMin3": 12, "taMax3": 21,
        "taMin4": 13, "taMax4": 22,
        "taMin5": 14, "taMax5": 23,
        "taMin10": 18, "taMax10": 26,
    }
    fr = FetchResult(raw_text="", raw_suffix=".json",
                     parsed={"items": [item]}, request={})
    collector = KmaMidTemperatureCollector()
    df = collector.parse(fr)
    assert {"source_id", "reg_id", "tm_fc", "forecast_date",
            "lead_days", "ta_min", "ta_max"}.issubset(df.columns)
    # 4 explicit lead days emitted (3, 4, 5, 10)
    assert sorted(df["lead_days"].tolist()) == [3, 4, 5, 10]
    # forecast_date for lead=3 = 2026-05-26 + 3d = 2026-05-29
    row_l3 = df[df["lead_days"] == 3].iloc[0]
    assert row_l3["forecast_date"] == pd.Timestamp("2026-05-29")
    assert row_l3["ta_min"] == 12 and row_l3["ta_max"] == 21


def test_kma_encoded_key_is_NOT_double_encoded_in_request_url():
    """Round-8 leakage canary: when the user has set
    KMA_PUBLIC_API_KEY_ENCODED (URL-encoded form), the request URL must
    contain the key VERBATIM, not URL-encoded a second time.

    requests.get(params={'serviceKey': encoded_key}) would dutifully
    URL-encode the encoded key — the '%' in '%2F' would become '%25',
    producing '%252F' on the wire. Servers reject with 401.

    We verify by intercepting the constructed URL via PreparedRequest —
    no network, no key disclosure.
    """
    from src.collectors.kma_mid_temperature import _get_with_data_go_kr_key
    import requests as _rq

    # Synthetic key in the EXACT shape data.go.kr issues (raw has '/+=', encoded has '%2F%2B%3D')
    raw_key = "abc/def+xyz="
    encoded_key = "abc%2Fdef%2Bxyz%3D"

    captured: dict = {}
    original_send = _rq.Session.send

    def _capture(self, request, **kw):
        captured["url"] = request.url
        # Don't actually send — return a fake 200
        from unittest.mock import MagicMock
        m = MagicMock(); m.status_code = 200; m.ok = True
        m.json.return_value = {"response": {"body": {"items": {"item": []}}}}
        m.text = "{}"
        return m

    base = "http://example.invalid/api"
    others = {"tmFc": "202605260600", "regId": "11B10101"}

    # ── Case 1: RAW key via params= → requests encodes ONCE
    try:
        _rq.Session.send = _capture
        captured.clear()
        _get_with_data_go_kr_key(base, raw_key, is_encoded=False, other_params=others)
    finally:
        _rq.Session.send = original_send
    raw_url = captured["url"]
    # Raw '/+=' must have become '%2F%2B%3D' (single encoding) — NOT '%252F'
    assert "%2F" in raw_url and "%2B" in raw_url and "%3D" in raw_url, (
        f"raw-key path: requests should single-encode '/+=' → '%2F%2B%3D'. URL: {raw_url}"
    )
    assert "%252F" not in raw_url, (
        f"raw-key path: double-encoding detected ('%2F' became '%252F'). URL: {raw_url}"
    )

    # ── Case 2: ENCODED key via manual URL → must appear VERBATIM
    try:
        _rq.Session.send = _capture
        captured.clear()
        _get_with_data_go_kr_key(base, encoded_key, is_encoded=True, other_params=others)
    finally:
        _rq.Session.send = original_send
    enc_url = captured["url"]
    # The exact 'abc%2Fdef%2Bxyz%3D' substring must be present — verbatim
    assert "abc%2Fdef%2Bxyz%3D" in enc_url, (
        f"encoded-key path: encoded key was mangled. URL: {enc_url}"
    )
    # And NOT double-encoded
    assert "abc%252Fdef%252Bxyz%253D" not in enc_url, (
        f"encoded-key path: DOUBLE-encoded the encoded key. URL: {enc_url}"
    )


def test_kma_resolve_service_key_prefers_raw_when_both_available(monkeypatch):
    """If both raw and encoded keys are configured, prefer raw (params= path
    is sturdier vs future requests changes)."""
    from src.collectors.kma_mid_temperature import _resolve_data_go_kr_service_key
    settings = SimpleNamespace(
        kma_public_api_key="raw-key-here",
        kma_public_api_key_encoded="raw%2Dkey%2Dhere",
    )
    key, is_enc = _resolve_data_go_kr_service_key(settings)
    assert key == "raw-key-here" and is_enc is False

    # Only encoded set → use encoded
    settings2 = SimpleNamespace(
        kma_public_api_key=None,
        kma_public_api_key_encoded="abc%2F",
    )
    key, is_enc = _resolve_data_go_kr_service_key(settings2)
    assert key == "abc%2F" and is_enc is True


def test_kma_raw_envelope_preserves_verbatim_server_payload_and_redacts_url():
    """Round-9.1: KMA collectors must save verbatim server responses (so
    the saved raw_text can be re-parsed if our extractor is wrong, AND
    so audits can see `response.header.resultCode/resultMsg` etc.).

    Earlier rounds synthesized a JSON with only the `items` we extracted
    — that lost the response envelope. This canary verifies:
      * the envelope schema is `{responses: [...], request_log: {...}}`
      * each response's `body` is the verbatim server payload (no parsing)
      * URLs are redacted (serviceKey replaced with `<REDACTED>`)
    """
    import json
    from src.collectors.kma_mid_temperature import (
        build_raw_responses_envelope, redact_service_key_in_url,
    )

    # URL redaction unit test
    url_a = "http://x/api?serviceKey=ABC%2FDEF&regId=11B10101&tmFc=202605260600"
    out_a = redact_service_key_in_url(url_a)
    assert "ABC%2FDEF" not in out_a
    assert "<REDACTED>" in out_a
    # Other params survive
    assert "regId=11B10101" in out_a
    assert "tmFc=202605260600" in out_a
    # Trailing-key edge case (serviceKey is last param)
    url_b = "http://x/api?regId=11B10101&serviceKey=ABC%2FDEF"
    out_b = redact_service_key_in_url(url_b)
    assert "ABC%2FDEF" not in out_b
    assert "<REDACTED>" in out_b

    # Envelope shape — verbatim body preserved character-for-character
    verbatim_body_1 = (
        '{"response":{"header":{"resultCode":"00","resultMsg":"NORMAL_SERVICE"},'
        '"body":{"dataType":"JSON","items":{"item":['
        '{"baseDate":"20260526","baseTime":"0600","category":"TMP",'
        '"fcstDate":"20260526","fcstTime":"0700","fcstValue":"21","nx":60,"ny":127}'
        ']},"pageNo":1,"numOfRows":10,"totalCount":1}}}'
    )
    raw_resp = [
        {"region_name": "Seoul", "nx": 60, "ny": 127,
         "status_code": 200, "url": "<REDACTED>", "body": verbatim_body_1},
    ]
    request_log = {"base_url": "http://x/api", "base_date": "20260526",
                   "service_key_form": "raw"}
    envelope = build_raw_responses_envelope(raw_resp, request_log)
    parsed_env = json.loads(envelope)
    assert set(parsed_env.keys()) == {"responses", "request_log"}
    assert len(parsed_env["responses"]) == 1
    saved_body = parsed_env["responses"][0]["body"]
    # CRITICAL: the saved body MUST be character-for-character what the
    # server returned. NO re-parsing, NO key reordering, NO synthesised
    # subset.
    assert saved_body == verbatim_body_1
    # And the resultCode/resultMsg envelope IS recoverable from the saved body
    inner = json.loads(saved_body)
    assert inner["response"]["header"]["resultCode"] == "00"
    assert inner["response"]["header"]["resultMsg"] == "NORMAL_SERVICE"


def test_kma_mid_temperature_parser_handles_empty_response():
    from src.collectors.base import FetchResult
    fr = FetchResult(raw_text="", raw_suffix=".json", parsed={"items": []}, request={})
    df = KmaMidTemperatureCollector().parse(fr)
    assert df.empty
    assert {"source_id", "reg_id", "tm_fc", "forecast_date", "ta_min", "ta_max"} <= set(df.columns)


def test_kma_error_payloads_are_persisted_before_raise(tmp_path, monkeypatch):
    """Round-9.2: data.go.kr error envelopes (4xx/5xx/HTML SERVICE_KEY pages,
    HTTP-200 with non-"00" resultCode) MUST be persisted to disk verbatim
    before fetch_one raises.

    Earlier rounds: BaseCollector.collect() persists raw_text only on success,
    and KMA fetch_one raised before returning a FetchResult — so the error
    body was lost. This canary patches the HTTP layer to return a 403 with
    SERVICE_KEY_NO_PERMISSION_ERROR text and asserts a verbatim raw file
    appears under data/raw/kma/mid_temperature/<date>/response_*.json
    *before* the exception escapes.
    """
    import json
    from unittest.mock import patch

    from src.collectors.base import CollectorError
    from src.collectors import kma_mid_temperature as mid_mod

    # Redirect data_dir so the test doesn't touch the real raw tree
    settings = SimpleNamespace(
        data_dir=tmp_path,
        kma_public_api_key="raw-key-here",
        kma_public_api_key_encoded=None,
    )
    monkeypatch.setattr("src.collectors.kma_mid_temperature.get_settings", lambda: settings)
    monkeypatch.setattr("src.utils.io.get_settings", lambda: settings)
    # Disable tenacity sleep so the retry doesn't hold the test for 30s
    monkeypatch.setattr(mid_mod.KmaMidTemperatureCollector.fetch_one.retry, "wait", lambda *a, **k: 0)

    verbatim_error_body = (
        '<OpenAPI_ServiceResponse><cmmMsgHeader>'
        '<errMsg>SERVICE ERROR</errMsg>'
        '<returnAuthMsg>SERVICE_KEY_IS_NOT_REGISTERED_ERROR</returnAuthMsg>'
        '<returnReasonCode>30</returnReasonCode>'
        '</cmmMsgHeader></OpenAPI_ServiceResponse>'
    )

    class _FakeResp:
        status_code = 403
        ok = False
        text = verbatim_error_body

        class request:  # noqa: D401 — match `r.request.url`
            url = "http://x/api?serviceKey=raw-key-here&regId=11B10101&tmFc=202605260600"

        def json(self):
            raise ValueError("not JSON")

    def _fake_get(*a, **k):
        return _FakeResp()

    with patch("src.collectors.kma_mid_temperature.requests.get", _fake_get):
        coll = KmaMidTemperatureCollector()
        with pytest.raises(CollectorError) as ei:
            coll.fetch_one(reg_ids=["11B10101"], tm_fc="202605260600")

    # Error message names the persisted path so an operator can grep
    assert "raw saved:" in str(ei.value)

    # Find every saved raw envelope (one per retry attempt — non-transient,
    # so exactly one. The retry decorator only retries TransientCollectorError.)
    saved_files = sorted((tmp_path / "raw" / "kma" / "mid_temperature").rglob("response_*.json"))
    assert saved_files, "No raw envelope was persisted before the raise"

    parsed = json.loads(saved_files[-1].read_text(encoding="utf-8"))
    assert set(parsed.keys()) == {"responses", "request_log"}
    assert len(parsed["responses"]) == 1
    resp = parsed["responses"][0]
    # Status + verbatim body preserved
    assert resp["status_code"] == 403
    assert resp["body"] == verbatim_error_body
    # URL redacted (serviceKey value never lands on disk)
    assert "raw-key-here" not in saved_files[-1].read_text(encoding="utf-8")
    assert "<REDACTED>" in resp["url"]


def test_kma_response_body_scrubs_echoed_service_key(tmp_path, monkeypatch):
    """Round-9.4: data.go.kr error envelopes frequently echo the submitted
    serviceKey value inside `<returnAuthMsg>` / `<resultMsg>` / `<reasonMsg>`
    fields. The URL-level redaction protects the captured request URL, but
    if the body text contains the same key value we'd persist it verbatim,
    silently defeating the URL redaction. This canary pins that both the
    RAW and the URL-ENCODED forms of the live service key are scrubbed
    from the body before the envelope lands on disk.
    """
    import json
    from unittest.mock import patch
    from urllib.parse import quote

    from src.collectors.base import CollectorError
    from src.collectors import kma_mid_temperature as mid_mod

    fake_raw_key = "ABC/DEF+GHI=secretServiceKeyValueZZZ"
    fake_encoded_key = quote(fake_raw_key, safe="")

    settings = SimpleNamespace(
        data_dir=tmp_path,
        kma_public_api_key=fake_raw_key,
        kma_public_api_key_encoded=None,
    )
    monkeypatch.setattr("src.collectors.kma_mid_temperature.get_settings", lambda: settings)
    monkeypatch.setattr("src.utils.io.get_settings", lambda: settings)
    monkeypatch.setattr(
        mid_mod.KmaMidTemperatureCollector.fetch_one.retry, "wait", lambda *a, **k: 0
    )

    # data.go.kr echoes the key in BOTH raw and URL-encoded forms across
    # different endpoints — we test both end up redacted.
    echoed_body = (
        '<OpenAPI_ServiceResponse><cmmMsgHeader>'
        '<errMsg>SERVICE ERROR</errMsg>'
        '<returnAuthMsg>SERVICE_KEY_IS_NOT_REGISTERED_ERROR</returnAuthMsg>'
        '<returnReasonCode>30</returnReasonCode>'
        f'<reasonMsg>Provided serviceKey={fake_raw_key} was not registered.</reasonMsg>'
        f'<reasonMsgEncoded>Encoded form: {fake_encoded_key}</reasonMsgEncoded>'
        '</cmmMsgHeader></OpenAPI_ServiceResponse>'
    )

    class _FakeResp:
        status_code = 403
        ok = False
        text = echoed_body

        class request:
            url = f"http://x/api?serviceKey={fake_raw_key}&regId=11B10101&tmFc=202605260600"

        def json(self):
            raise ValueError("not JSON")

    with patch("src.collectors.kma_mid_temperature.requests.get",
               lambda *a, **k: _FakeResp()):
        coll = mid_mod.KmaMidTemperatureCollector()
        with pytest.raises(CollectorError):
            coll.fetch_one(reg_ids=["11B10101"], tm_fc="202605260600")

    saved = sorted((tmp_path / "raw" / "kma" / "mid_temperature").rglob("response_*.json"))
    assert saved, "Error envelope was not persisted"
    on_disk = saved[-1].read_text(encoding="utf-8")

    # NEITHER form of the key may appear anywhere in the saved file
    assert fake_raw_key not in on_disk, (
        f"Raw service key leaked into persisted envelope: {saved[-1]}"
    )
    assert fake_encoded_key not in on_disk, (
        f"URL-encoded service key leaked into persisted envelope: {saved[-1]}"
    )
    # And we DID redact (placeholder shows up in the saved JSON)
    assert "<REDACTED>" in on_disk

    # The rest of the error envelope is preserved (we didn't nuke the body)
    parsed = json.loads(on_disk)
    body_text = parsed["responses"][0]["body"]
    assert "SERVICE_KEY_IS_NOT_REGISTERED_ERROR" in body_text
    assert "returnReasonCode" in body_text


def test_kma_exception_messages_do_not_echo_service_key(tmp_path, monkeypatch):
    """Round-9.5: even after on-disk envelope scrubbing, the *exception
    message* itself embedded `r.text[:200]` raw — and
    `BaseCollector.collect()` logs the exception via
    `self.log.exception(exc)`. So an upstream error body that echoes the
    serviceKey would surface in the log file (and in any caller's
    captured `str(exc)`). This canary pins that the body excerpt is
    scrubbed before it enters the exception message.
    """
    from unittest.mock import patch
    from urllib.parse import quote
    from src.collectors.base import CollectorError
    from src.collectors import kma_mid_temperature as mid_mod

    fake_raw_key = "ABC/DEF+SECRETkmaKEYvalueZZZ12345"
    fake_encoded_key = quote(fake_raw_key, safe="")

    settings = SimpleNamespace(
        data_dir=tmp_path,
        kma_public_api_key=fake_raw_key,
        kma_public_api_key_encoded=None,
    )
    monkeypatch.setattr("src.collectors.kma_mid_temperature.get_settings", lambda: settings)
    monkeypatch.setattr("src.utils.io.get_settings", lambda: settings)
    monkeypatch.setattr(
        mid_mod.KmaMidTemperatureCollector.fetch_one.retry, "wait", lambda *a, **k: 0
    )

    echoed_body = (
        f'<errMsg>Bad key {fake_raw_key} or encoded {fake_encoded_key}</errMsg>'
    )

    class _FakeResp:
        status_code = 403
        ok = False
        text = echoed_body

        class request:
            url = f"http://x/api?serviceKey={fake_raw_key}"

        def json(self):
            raise ValueError("not JSON")

    with patch("src.collectors.kma_mid_temperature.requests.get",
               lambda *a, **k: _FakeResp()):
        coll = mid_mod.KmaMidTemperatureCollector()
        with pytest.raises(CollectorError) as ei:
            coll.fetch_one(reg_ids=["11B10101"], tm_fc="202605260600")

    msg = str(ei.value)
    assert fake_raw_key not in msg, f"raw key leaked into exception: {msg!r}"
    assert fake_encoded_key not in msg, f"encoded key leaked into exception: {msg!r}"
    assert "<REDACTED>" in msg
    # The status code + saved-path hint survive
    assert "403" in msg
    assert "raw saved:" in msg


def test_eia_steo_exception_and_raw_text_scrub_api_key(monkeypatch):
    """EIA STEO returns error bodies that include the submitted key value.
    Both the raised exception message AND the persisted raw_text must be
    scrubbed (the raw_text gets written to disk via the success path's
    `FetchResult.raw_text`, but a partial-fetch path could still see it
    in unit tests).
    """
    from unittest.mock import patch
    from src.collectors.base import CollectorError
    from src.collectors import eia_steo as eia_mod

    fake_key = "SECRETeiaAPIkeyVALUE0123456789abcdef"
    settings = SimpleNamespace(eia_api_key=fake_key)
    monkeypatch.setattr("src.collectors.eia_steo.get_settings", lambda: settings)
    monkeypatch.setattr(
        eia_mod.EiaSteoCollector.fetch_one.retry, "wait", lambda *a, **k: 0
    )

    echoed_body = (
        f'{{"error":"INVALID_API_KEY","description":"Submitted api_key={fake_key} was not recognised"}}'
    )

    class _FakeResp:
        status_code = 403
        ok = False
        text = echoed_body

        def json(self):
            import json as _j
            return _j.loads(echoed_body)

    with patch("src.collectors.eia_steo.requests.get",
               lambda *a, **k: _FakeResp()):
        coll = eia_mod.EiaSteoCollector()
        with pytest.raises(CollectorError) as ei:
            coll.fetch_one()

    msg = str(ei.value)
    assert fake_key not in msg, f"EIA api_key leaked into exception: {msg!r}"
    assert "<REDACTED>" in msg
    assert "403" in msg


def test_kma_request_exception_does_not_leak_key_in_traceback_chain(
    tmp_path, monkeypatch
):
    """Round-9.6: `requests.RequestException`s frequently embed the full
    request URL in `str(e)`. Two leak vectors at once:
      (a) our own f-string in `raise TransientCollectorError(...)`
      (b) the CAUSE CHAIN that `logger.exception` walks — by default
          `raise X from e` preserves `__cause__`, so `traceback.format_exception`
          on the outer exception prints the inner exception's str() too.

    Pin that neither the exception message NOR the formatted traceback
    contains the live serviceKey.
    """
    import traceback
    from unittest.mock import patch
    from urllib.parse import quote
    import requests
    from src.collectors.base import TransientCollectorError
    from src.collectors import kma_mid_temperature as mid_mod

    fake_raw_key = "ABCSECRETkma1234567890XYZ"
    fake_encoded_key = quote(fake_raw_key, safe="")

    settings = SimpleNamespace(
        data_dir=tmp_path,
        kma_public_api_key=fake_raw_key,
        kma_public_api_key_encoded=None,
    )
    monkeypatch.setattr("src.collectors.kma_mid_temperature.get_settings", lambda: settings)
    monkeypatch.setattr("src.utils.io.get_settings", lambda: settings)
    monkeypatch.setattr(
        mid_mod.KmaMidTemperatureCollector.fetch_one.retry, "wait", lambda *a, **k: 0
    )

    # Simulate urllib3 surfacing the full URL in a ConnectionError —
    # very common on DNS/SSL failures.
    upstream_msg = (
        f"HTTPSConnectionPool(host='apis.data.go.kr'): "
        f"Max retries exceeded with url: /1360000/.../getMidTa?"
        f"serviceKey={fake_raw_key}&regId=11B10101 "
        f"(also tried encoded form serviceKey={fake_encoded_key})"
    )

    def _raise_conn_err(*a, **k):
        raise requests.exceptions.ConnectionError(upstream_msg)

    with patch("src.collectors.kma_mid_temperature.requests.get", _raise_conn_err):
        coll = mid_mod.KmaMidTemperatureCollector()
        with pytest.raises(TransientCollectorError) as ei:
            coll.fetch_one(reg_ids=["11B10101"], tm_fc="202605260600")

    # (a) our exception message is scrubbed
    msg = str(ei.value)
    assert fake_raw_key not in msg, f"raw key leaked into TransientCollectorError: {msg!r}"
    assert fake_encoded_key not in msg, f"encoded key leaked into TransientCollectorError: {msg!r}"
    assert "ConnectionError" in msg  # type info preserved for debugging
    assert "<REDACTED>" in msg

    # (b) the FORMATTED TRACEBACK (what `logger.exception` writes to disk)
    # must not contain the live key either — this exercises the `from None`
    # discipline that breaks the __cause__ chain.
    formatted = "".join(traceback.format_exception(
        type(ei.value), ei.value, ei.value.__traceback__
    ))
    assert fake_raw_key not in formatted, (
        "raw key leaked into formatted traceback (the cause chain was not "
        "suppressed):\n" + formatted
    )
    assert fake_encoded_key not in formatted, (
        "encoded key leaked into formatted traceback:\n" + formatted
    )
    # And we explicitly broke the cause chain
    assert ei.value.__cause__ is None
    assert ei.value.__suppress_context__ is True


def test_kma_resultmsg_does_not_leak_key_when_server_echoes_it(
    tmp_path, monkeypatch
):
    """data.go.kr returns HTTP 200 with a non-`00` `resultCode` for many
    error conditions, and the `resultMsg` text frequently embeds the
    submitted serviceKey value (e.g.
    `"Provided serviceKey=ABC... is not registered"`). Pin that this text
    is scrubbed BEFORE it enters the exception message.
    """
    import json
    from unittest.mock import patch
    from src.collectors.base import CollectorError
    from src.collectors import kma_mid_temperature as mid_mod

    fake_key = "SECRETkmaKEYvalue1234567890ABCDEF"
    settings = SimpleNamespace(
        data_dir=tmp_path,
        kma_public_api_key=fake_key,
        kma_public_api_key_encoded=None,
    )
    monkeypatch.setattr("src.collectors.kma_mid_temperature.get_settings", lambda: settings)
    monkeypatch.setattr("src.utils.io.get_settings", lambda: settings)
    monkeypatch.setattr(
        mid_mod.KmaMidTemperatureCollector.fetch_one.retry, "wait", lambda *a, **k: 0
    )

    # HTTP 200, but resultCode != "00" and resultMsg echoes the key
    body = {
        "response": {
            "header": {
                "resultCode": "30",
                "resultMsg": f"SERVICE_KEY_IS_NOT_REGISTERED_ERROR (key={fake_key})",
            },
            "body": {"items": {"item": []}},
        }
    }
    body_text = json.dumps(body)

    class _FakeResp:
        status_code = 200
        ok = True
        text = body_text

        class request:
            url = f"http://x/api?serviceKey={fake_key}&regId=11B10101"

        def json(self):
            return body

    with patch("src.collectors.kma_mid_temperature.requests.get",
               lambda *a, **k: _FakeResp()):
        coll = mid_mod.KmaMidTemperatureCollector()
        with pytest.raises(CollectorError) as ei:
            coll.fetch_one(reg_ids=["11B10101"], tm_fc="202605260600")

    msg = str(ei.value)
    assert fake_key not in msg, f"resultMsg key leaked into exception: {msg!r}"
    assert "<REDACTED>" in msg
    assert "SERVICE_KEY_IS_NOT_REGISTERED_ERROR" in msg  # signal text preserved


def test_eia_request_exception_scrubs_url_encoded_api_key(monkeypatch):
    """Round-9.7: `requests.get(params={"api_key": api_key, ...})`
    URL-encodes the value before transmission. urllib3 connection errors
    surface the full URL — so a plain `str.replace(raw_key, ...)` would
    miss the URL-encoded form that's actually in `str(e)`.

    Pin that the multi-encoding scrub (raw + URL-encoded + URL-decoded
    forms via `src.utils.redact.secret_variants`) catches both.
    """
    import traceback
    from unittest.mock import patch
    from urllib.parse import quote
    import requests
    from src.collectors.base import TransientCollectorError
    from src.collectors import eia_steo as eia_mod

    # Key with characters that get URL-encoded (`/`, `+`, `=`) — uncommon
    # for real EIA keys but possible in principle, and exactly what
    # data.go.kr keys look like (same scrubber, same threat).
    fake_key = "AbCd/EfGh+IjKl=mn0123456789secret"
    encoded_key = quote(fake_key, safe="")
    assert encoded_key != fake_key  # sanity — they MUST differ for the test

    settings = SimpleNamespace(eia_api_key=fake_key)
    monkeypatch.setattr("src.collectors.eia_steo.get_settings", lambda: settings)
    monkeypatch.setattr(
        eia_mod.EiaSteoCollector.fetch_one.retry, "wait", lambda *a, **k: 0
    )

    # urllib3 surfaces the URL with the URL-ENCODED key
    upstream_msg = (
        f"HTTPSConnectionPool(host='api.eia.gov'): "
        f"Max retries exceeded with url: /v2/steo/data/?"
        f"api_key={encoded_key}&frequency=monthly"
    )

    def _raise_conn_err(*a, **k):
        raise requests.exceptions.ConnectionError(upstream_msg)

    with patch("src.collectors.eia_steo.requests.get", _raise_conn_err):
        coll = eia_mod.EiaSteoCollector()
        with pytest.raises(TransientCollectorError) as ei:
            coll.fetch_one()

    msg = str(ei.value)
    assert fake_key not in msg, f"raw EIA api_key leaked: {msg!r}"
    assert encoded_key not in msg, (
        f"URL-encoded EIA api_key leaked: {msg!r}\n"
        "(the plain str.replace(raw_key, ...) approach misses this form; "
        "use src.utils.redact.secret_variants instead)"
    )
    assert "<REDACTED>" in msg
    assert "ConnectionError" in msg  # type info preserved

    # Cause chain suppressed → formatted traceback also clean
    formatted = "".join(traceback.format_exception(
        type(ei.value), ei.value, ei.value.__traceback__
    ))
    assert fake_key not in formatted
    assert encoded_key not in formatted
    assert ei.value.__cause__ is None
    assert ei.value.__suppress_context__ is True


def test_eia_response_body_scrub_handles_url_encoded_key(monkeypatch):
    """EIA may echo the api_key in either raw or URL-encoded form
    depending on which middleware logs it. Both must be scrubbed from
    the body excerpt that ends up in the exception message AND from the
    persisted `safe_raw_text`."""
    from unittest.mock import patch
    from urllib.parse import quote
    from src.collectors.base import CollectorError
    from src.collectors import eia_steo as eia_mod

    fake_key = "AbCd/EfGh+IjKl=mn0123456789secret"
    encoded_key = quote(fake_key, safe="")

    settings = SimpleNamespace(eia_api_key=fake_key)
    monkeypatch.setattr("src.collectors.eia_steo.get_settings", lambda: settings)
    monkeypatch.setattr(
        eia_mod.EiaSteoCollector.fetch_one.retry, "wait", lambda *a, **k: 0
    )

    body = (
        f'{{"error":"INVALID_API_KEY","description":'
        f'"Raw form: {fake_key}. Encoded form in URL: {encoded_key}"}}'
    )

    class _FakeResp:
        status_code = 403
        ok = False
        text = body

        def json(self):
            import json as _j; return _j.loads(body)

    with patch("src.collectors.eia_steo.requests.get",
               lambda *a, **k: _FakeResp()):
        coll = eia_mod.EiaSteoCollector()
        with pytest.raises(CollectorError) as ei:
            coll.fetch_one()

    msg = str(ei.value)
    assert fake_key not in msg
    assert encoded_key not in msg
    assert "INVALID_API_KEY" in msg  # signal text preserved


def test_secret_variants_emits_raw_encoded_and_decoded_forms():
    """Pin the contract of the shared `secret_variants` helper — what
    every collector's exception scrubbing relies on."""
    from src.utils.redact import secret_variants

    raw = "AbCd/EfGh+IjKl=mn0123456789secret"
    variants = set(secret_variants(raw))
    # Raw form
    assert raw in variants
    # URL-encoded form (with /+= encoded)
    assert any("%2F" in v and "%2B" in v and "%3D" in v for v in variants), variants
    # Plain ASCII value → encoded form is no-op, but still returns at minimum the raw
    plain = "alphanumonly12345abcdef"
    assert plain in secret_variants(plain)
    # Short values rejected
    assert secret_variants("short") == []
    assert secret_variants("") == []
    assert secret_variants(None) == []


def test_base_collector_scrubs_credential_kwargs_from_metadata():
    """`BaseCollector.collect(api_key=...)` previously stored kwargs
    verbatim on `CollectorRun.params`, which `to_metadata()` writes to
    `metadata_*.json`. Pin that any kwarg whose name matches the
    credential pattern is replaced with `<REDACTED>` in the stored
    params dict, while non-credential kwargs survive.
    """
    from src.collectors.base import _scrub_credential_kwargs

    raw = {
        "api_key": "SECRETkeyVALUE1234",
        "apiKey":  "SECRETkeyVALUE5678",
        "service_key": "SVCkeyABC",
        "auth": "Bearer xyz",
        "secret": "shh",
        "session_id": "sess_abc",
        "Cookie": "JSESSIONID=foo",
        "password": "hunter2",
        # NOT credential-shaped — must survive
        "reg_ids": ["11B10101"],
        "tm_fc": "202605260600",
        "frequency": "monthly",
    }
    out = _scrub_credential_kwargs(raw)
    for sensitive in (
        "api_key", "apiKey", "service_key", "auth", "secret",
        "session_id", "Cookie", "password",
    ):
        assert out[sensitive] == "<REDACTED>", (
            f"{sensitive!r} not redacted: {out[sensitive]!r}"
        )
    # Non-credential kwargs untouched
    assert out["reg_ids"] == ["11B10101"]
    assert out["tm_fc"] == "202605260600"
    assert out["frequency"] == "monthly"
    # Original dict not mutated
    assert raw["api_key"] == "SECRETkeyVALUE1234"


def test_kma_error_envelopes_dont_collide_within_one_second(tmp_path, monkeypatch):
    """Round-9.3: tenacity can retry a transient 5xx burst inside a single
    wall-second. `_stamp` (`%Y%m%dT%H%M%SZ`) is second-resolution, so
    without a disambiguator the second save silently overwrites the first.

    This canary pins `_stamp` to a constant, calls
    `persist_partial_error_envelope` twice with DIFFERENT payloads, and
    asserts both files exist on disk with distinct names and the original
    bodies preserved.
    """
    from src.collectors import kma_mid_temperature as mid_mod
    from src.collectors.kma_mid_temperature import persist_partial_error_envelope

    settings = SimpleNamespace(data_dir=tmp_path)
    monkeypatch.setattr("src.utils.io.get_settings", lambda: settings)
    # Freeze the second-resolution stamp so any unrelated jitter doesn't
    # accidentally make the two files distinct via the seconds field.
    monkeypatch.setattr("src.utils.io._stamp", lambda ts=None: "20260526T100000Z")

    # Each call has its own microsecond + payload hash, so they MUST land
    # on disk under distinct filenames.
    p1 = persist_partial_error_envelope(
        "kma_mid_temperature",
        [{"reg_id": "11B10101", "tm_fc": "202605260600",
          "status_code": 500, "url": "<REDACTED>", "body": "first error"}],
        {"base_url": "http://x/api", "attempt": 1},
    )
    p2 = persist_partial_error_envelope(
        "kma_mid_temperature",
        [{"reg_id": "11B10101", "tm_fc": "202605260600",
          "status_code": 500, "url": "<REDACTED>", "body": "second error"}],
        {"base_url": "http://x/api", "attempt": 2},
    )

    assert p1 != p2, "Same-second retries collided to the same path"
    assert p1.exists() and p2.exists()
    # Bodies survive — first save was NOT clobbered
    assert "first error" in p1.read_text(encoding="utf-8")
    assert "second error" in p2.read_text(encoding="utf-8")
    # And the filename scheme is the expected disambiguated form
    assert "__" in p1.name and p1.name.endswith(".json")


def test_kma_village_error_payloads_are_persisted_before_raise(tmp_path, monkeypatch):
    """Mirror of the mid-term canary for the village forecast collector."""
    import json
    from unittest.mock import patch

    from src.collectors.base import CollectorError
    from src.collectors import kma_village_fcst as vil_mod
    from src.collectors.kma_village_fcst import KmaVilageFcstCollector

    settings = SimpleNamespace(
        data_dir=tmp_path,
        kma_public_api_key="raw-key-here",
        kma_public_api_key_encoded=None,
    )
    monkeypatch.setattr("src.collectors.kma_mid_temperature.get_settings", lambda: settings)
    monkeypatch.setattr("src.collectors.kma_village_fcst.get_settings", lambda: settings)
    monkeypatch.setattr("src.utils.io.get_settings", lambda: settings)
    monkeypatch.setattr(vil_mod.KmaVilageFcstCollector.fetch_one.retry, "wait", lambda *a, **k: 0)

    # data.go.kr quirk: HTTP-200 with non-"00" resultCode in the JSON body
    verbatim_error_body = (
        '{"response":{"header":{"resultCode":"22",'
        '"resultMsg":"LIMITED_NUMBER_OF_SERVICE_REQUESTS_EXCEEDS_ERROR"},'
        '"body":{"dataType":"JSON","items":{"item":[]},'
        '"pageNo":1,"numOfRows":1000,"totalCount":0}}}'
    )

    class _FakeResp:
        status_code = 200
        ok = True
        text = verbatim_error_body

        class request:
            url = ("http://x/api?serviceKey=raw-key-here&base_date=20260526&"
                   "base_time=0500&nx=60&ny=127&pageNo=1&numOfRows=1000&dataType=JSON")

        def json(self):
            return json.loads(verbatim_error_body)

    with patch("src.collectors.kma_mid_temperature.requests.get", lambda *a, **k: _FakeResp()):
        coll = KmaVilageFcstCollector()
        with pytest.raises(CollectorError) as ei:
            coll.fetch_one(
                regions=[{"name": "Seoul", "nx": 60, "ny": 127}],
                base_date="20260526",
                base_time="0500",
            )

    assert "raw saved:" in str(ei.value)
    saved_files = sorted((tmp_path / "raw" / "kma" / "village_fcst").rglob("response_*.json"))
    assert saved_files, "Village forecast error envelope was not persisted before raise"

    parsed = json.loads(saved_files[-1].read_text(encoding="utf-8"))
    assert parsed["responses"][0]["body"] == verbatim_error_body
    # The non-"00" resultCode IS recoverable from the saved verbatim body
    inner = json.loads(parsed["responses"][0]["body"])
    assert inner["response"]["header"]["resultCode"] == "22"
    assert "raw-key-here" not in saved_files[-1].read_text(encoding="utf-8")
