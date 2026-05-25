"""KMA Vilage short-term forecast (단기예보) collector.

API: ``VilageFcstInfoService_2.0/getVilageFcst``
Docs: `기상청41_단기예보 조회서비스_오픈API활용가이드_241128.docx`

The service publishes 8 times/day (02/05/08/11/14/17/20/23 :10 KST). For
each (base_date, base_time, nx, ny) request it returns up to ~5 days of
3-hourly forecasts for categories TMP/TMX/TMN/SKY/PTY/POP/PCP/REH/WSD…

We pull one (nx, ny) per configured region and emit one long-format row
per (fcst_date, fcst_time, category, value).

The encoded vs raw service key dance is handled by the shared
``_resolve_data_go_kr_service_key`` + ``_get_with_data_go_kr_key`` helpers
from `kma_mid_temperature.py` — round-8 leakage fix.
"""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import Any

import pandas as pd
import requests
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from src.collectors.base import BaseCollector, CollectorError, FetchResult, TransientCollectorError
from src.collectors.kma_mid_temperature import (
    _get_with_data_go_kr_key,
    _resolve_data_go_kr_service_key,
    _service_key_variants,
    build_raw_responses_envelope,
    persist_partial_error_envelope,
    redact_secrets_in_text,
    redact_service_key_in_url,
)
from src.config.settings import get_settings


# 단기예보 발표시각 (KST). 매시각 :10 이후 호출 가능.
_VILAGE_FCST_TIMES = ["0200", "0500", "0800", "1100", "1400", "1700", "2000", "2300"]


def _latest_base_datetime(now: datetime) -> tuple[str, str]:
    """Return (base_date, base_time) for the most recent published
    short-term forecast, accounting for the :10 post-publish window.

    Per the KMA guide, the 02:00 release becomes callable at 02:10. If
    `now` is earlier than the +10-minute window, use the prior slot.
    """
    cutoff = now - timedelta(minutes=10)
    today_str = cutoff.strftime("%Y%m%d")
    yesterday_str = (cutoff - timedelta(days=1)).strftime("%Y%m%d")
    # Slots already past cutoff today (in HHMM int form)
    cutoff_hhmm = int(cutoff.strftime("%H%M"))
    past = [t for t in _VILAGE_FCST_TIMES if int(t) <= cutoff_hhmm]
    if past:
        return today_str, past[-1]
    # All today's slots still in the future → use last slot from yesterday
    return yesterday_str, _VILAGE_FCST_TIMES[-1]


# Default regions to pull. (nx, ny) from the official grid spreadsheet.
# Keep this small (3 cities) — the full grid has 3835 rows, but we only
# need a handful for an SMP monthly forecast aggregate signal.
DEFAULT_REGIONS = [
    {"name": "Seoul", "nx": 60, "ny": 127},
    {"name": "Busan", "nx": 98, "ny": 76},
    {"name": "Daegu", "nx": 89, "ny": 90},
]


class KmaVilageFcstCollector(BaseCollector):
    """Short-term (3-5 day) forecast for configured grid points."""

    source_name = "kma_village_fcst"

    @retry(
        reraise=True,
        retry=retry_if_exception_type(TransientCollectorError),
        stop=stop_after_attempt(4),
        wait=wait_exponential(multiplier=1, min=1, max=20),
    )
    def fetch_one(self, **kwargs: Any) -> FetchResult:
        settings = get_settings()
        kw_key = kwargs.get("api_key")
        if kw_key is not None:
            api_key, is_encoded = kw_key, bool(kwargs.get("is_encoded", False))
        else:
            api_key, is_encoded = _resolve_data_go_kr_service_key(settings)

        base_url = self.config.get(
            "base_url",
            "http://apis.data.go.kr/1360000/VilageFcstInfoService_2.0/getVilageFcst",
        )
        regions: list[dict[str, Any]] = (
            kwargs.get("regions")
            or self.config.get("default_regions")
            or DEFAULT_REGIONS
        )
        now = datetime.utcnow() + timedelta(hours=9)  # KST
        base_date = kwargs.get("base_date")
        base_time = kwargs.get("base_time")
        if not base_date or not base_time:
            base_date, base_time = _latest_base_datetime(now)
        # We want the full forecast horizon (typically ~290 rows per region).
        num_of_rows = int(kwargs.get("num_of_rows") or 1000)

        all_items: list[dict[str, Any]] = []
        raw_responses: list[dict[str, Any]] = []
        # Live service-key forms to scrub from response bodies — see
        # `_service_key_variants` for why this is needed.
        key_secrets = _service_key_variants(api_key, is_encoded)
        request_log = {
            "base_url": base_url,
            "base_date": base_date,
            "base_time": base_time,
            "regions": [{"name": r.get("name", "?"), "nx": r["nx"], "ny": r["ny"]} for r in regions],
            "service_key_form": "encoded" if is_encoded else "raw",
        }

        for region in regions:
            other = {
                "pageNo": 1,
                "numOfRows": num_of_rows,
                "dataType": "JSON",
                "base_date": base_date,
                "base_time": base_time,
                "nx": region["nx"],
                "ny": region["ny"],
            }
            try:
                r = _get_with_data_go_kr_key(base_url, api_key, is_encoded, other, timeout=30)
            except requests.RequestException as e:
                # urllib3/requests exceptions can embed the full URL with
                # serviceKey. Scrub our own message AND break the cause
                # chain (`from None`) so `logger.exception` doesn't print
                # the original exception's str() in the traceback.
                safe_err = redact_secrets_in_text(f"{type(e).__name__}: {e}", key_secrets)
                raise TransientCollectorError(
                    f"KMA village request failed: {safe_err}"
                ) from None
            # Capture verbatim payload BEFORE status branching — error
            # envelopes from data.go.kr are also audit-worthy. Body is
            # scrubbed of any echoed serviceKey value first.
            raw_responses.append({
                "region_name": region.get("name"),
                "nx": region["nx"], "ny": region["ny"],
                "base_date": base_date, "base_time": base_time,
                "status_code": r.status_code,
                "url": redact_service_key_in_url(r.request.url or base_url),
                "body": redact_secrets_in_text(r.text, key_secrets),
            })
            # Scrub the body excerpt BEFORE it ends up in any exception
            # message — base.py logs exceptions via self.log.exception().
            body_excerpt = redact_secrets_in_text(r.text, key_secrets)[:200]
            if r.status_code in {429, 500, 502, 503, 504}:
                saved = persist_partial_error_envelope(
                    self.source_name, raw_responses, request_log
                )
                raise TransientCollectorError(
                    f"KMA village transient {r.status_code} (raw saved: {saved})"
                )
            if not r.ok:
                saved = persist_partial_error_envelope(
                    self.source_name, raw_responses, request_log
                )
                raise CollectorError(
                    f"KMA village {r.status_code}: {body_excerpt} (raw saved: {saved})"
                )
            try:
                body = r.json()
            except ValueError as e:
                saved = persist_partial_error_envelope(
                    self.source_name, raw_responses, request_log
                )
                raise CollectorError(
                    f"KMA village: non-JSON response. "
                    f"First 200 chars: {body_excerpt} (raw saved: {saved})"
                ) from e
            header = body.get("response", {}).get("header", {})
            if header.get("resultCode") not in {"00", None}:
                saved = persist_partial_error_envelope(
                    self.source_name, raw_responses, request_log
                )
                # resultMsg from data.go.kr often embeds the submitted
                # serviceKey — scrub before raising.
                safe_code = redact_secrets_in_text(
                    str(header.get("resultCode") or ""), key_secrets
                )
                safe_msg = redact_secrets_in_text(
                    str(header.get("resultMsg") or ""), key_secrets
                )
                raise CollectorError(
                    f"KMA village result {safe_code}: {safe_msg} (raw saved: {saved})"
                )
            items = (
                body.get("response", {})
                    .get("body", {})
                    .get("items", {})
                    .get("item", [])
            )
            if isinstance(items, dict):
                items = [items]
            for it in items:
                it["_region_name"] = region.get("name")
            all_items.extend(items)

        raw_text = build_raw_responses_envelope(raw_responses, request_log)
        return FetchResult(
            raw_text=raw_text,
            raw_suffix=".json",
            parsed={"items": all_items},
            request={"url": base_url, "params": request_log},
        )

    def parse(self, fetch_result: FetchResult) -> pd.DataFrame:
        parsed = fetch_result.parsed
        items = parsed.get("items", [])
        if not items:
            return pd.DataFrame(columns=[
                "source_id", "region_name", "base_date", "base_time",
                "nx", "ny", "fcst_date", "fcst_time", "category",
                "fcst_value", "fcst_value_numeric",
            ])
        rows: list[dict[str, Any]] = []
        for it in items:
            v_raw = it.get("fcstValue")
            v_num = pd.to_numeric(v_raw, errors="coerce")
            rows.append({
                "source_id": self.source_name,
                "region_name": it.get("_region_name"),
                "base_date": str(it.get("baseDate", "")),
                "base_time": str(it.get("baseTime", "")),
                "nx": int(it["nx"]) if it.get("nx") is not None else None,
                "ny": int(it["ny"]) if it.get("ny") is not None else None,
                "fcst_date": str(it.get("fcstDate", "")),
                "fcst_time": str(it.get("fcstTime", "")),
                "category": it.get("category"),
                "fcst_value": v_raw,
                "fcst_value_numeric": v_num,
            })
        df = pd.DataFrame(rows)
        return df.sort_values(
            ["region_name", "fcst_date", "fcst_time", "category"]
        ).reset_index(drop=True)
