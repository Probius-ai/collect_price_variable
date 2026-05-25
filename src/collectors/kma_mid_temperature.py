"""KMA mid-term temperature forecast API collector.

Endpoint: http://apis.data.go.kr/1360000/MidFcstInfoService/getMidTa
Docs: https://www.data.go.kr/data/15059468/openapi.do

Each request returns 3-10-day-ahead daily min/max temperature forecasts
for one regional ID (regId). Published twice daily at 06:00 and 18:00 KST.

We pull all configured regions, normalise to one row per (forecast_date,
reg_id) with `ta_min` and `ta_max` columns, and persist with the standard
collector layout.

The forecast issuance time (`tm_fc`) is preserved so feature engineers
can compute "what was forecast for month M+1 at the end of month M"
without bringing future information in.
"""

from __future__ import annotations

import hashlib
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any
from urllib.parse import quote, unquote, urlencode

import pandas as pd
import requests
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from src.collectors.base import BaseCollector, CollectorError, FetchResult, TransientCollectorError
from src.config.settings import get_settings
from src.utils.io import write_raw_payload
from src.utils.time import collected_now_utc


def _resolve_data_go_kr_service_key(settings) -> tuple[str, bool]:
    """Return (key, is_already_encoded).

    data.go.kr issues two forms of the same key — RAW (contains '/' '+' '=')
    and URL-ENCODED (those are '%2F' '%2B' '%3D'). If we hand the encoded
    form to ``requests.get(params={'serviceKey': key})``, requests dutifully
    URL-encodes it a SECOND time (the '%' becomes '%25') and the server
    rejects with 401 — confirmed empirically in round-8 diagnostics.

    Preference order:
      1) RAW key (`KMA_PUBLIC_API_KEY`)        → safe via params=
      2) ENCODED key (`*_ENCODED`)             → must be appended to the URL
    Raw wins when both are present because the params= path is sturdier
    against future ``requests`` changes.
    """
    raw = getattr(settings, "kma_public_api_key", None)
    encoded = getattr(settings, "kma_public_api_key_encoded", None)
    if raw:
        return raw, False
    if encoded:
        return encoded, True
    raise CollectorError(
        "KMA_PUBLIC_API_KEY (raw) or KMA_PUBLIC_API_KEY_ENCODED must be set."
    )


def _get_with_data_go_kr_key(
    base_url: str,
    service_key: str,
    is_encoded: bool,
    other_params: dict,
    timeout: int = 30,
) -> requests.Response:
    """HTTP GET to data.go.kr with the correct serviceKey-encoding strategy.

    * is_encoded=False → pass via ``params=`` (requests encodes once)
    * is_encoded=True  → manually build the URL so the already-encoded
      key passes through verbatim (the rest of the params are encoded
      via ``urllib.parse.urlencode`` — single encoding too)
    """
    if not is_encoded:
        return requests.get(
            base_url,
            params={"serviceKey": service_key, **other_params},
            timeout=timeout,
        )
    other_encoded = urlencode(other_params)
    sep = "&" if other_encoded else ""
    url = f"{base_url}?serviceKey={service_key}{sep}{other_encoded}"
    return requests.get(url, timeout=timeout)


def redact_service_key_in_url(url: str) -> str:
    """Replace the serviceKey value in a URL with `<REDACTED>` for audit logs."""
    if "serviceKey=" not in url:
        return url
    head, sep, tail = url.partition("serviceKey=")
    # tail may end at the next '&' or end-of-string
    if "&" in tail:
        _key, _, rest = tail.partition("&")
        return f"{head}{sep}<REDACTED>&{rest}"
    return f"{head}{sep}<REDACTED>"


# Re-exports from `src.utils.redact` — kept at module scope so
# `kma_village_fcst.py` and the round-8/9 test suite can keep importing
# them from here. New code should prefer the canonical
# `src.utils.redact.{secret_variants, redact_secrets_in_text}`.
from src.utils.redact import (
    redact_secrets_in_text,
    secret_variants as _generic_secret_variants,
)


def _service_key_variants(key: str, is_encoded: bool) -> list[str]:
    """Return every form of the live service key worth scrubbing from a
    response body.

    Why this matters: data.go.kr error envelopes (XML/JSON
    SERVICE_KEY_NO_PERMISSION_ERROR, etc.) frequently echo the submitted
    serviceKey back inside ``<returnReasonCode>`` / ``<returnAuthMsg>`` /
    ``<resultMsg>`` fields. The URL-level redaction protects the captured
    URL, but if the server's body text contains the same key value we
    persist it verbatim — defeating the redaction.

    Thin wrapper over the generic helper — the `is_encoded` flag is
    accepted for backward compatibility with existing call sites; the
    helper now generates both raw and URL-encoded forms unconditionally.
    """
    return _generic_secret_variants(key)


def persist_partial_error_envelope(
    source_name: str,
    raw_responses: list[dict],
    request_log: dict,
) -> Path:
    """Persist the verbatim raw envelope to disk on error, BEFORE the collector
    raises and `BaseCollector.collect()` skips its normal persistence path.

    Without this, data.go.kr error responses (401/403/500/SERVICE_KEY_NO_PERMISSION
    payloads, etc.) would be lost — making post-mortem diagnosis impossible.

    `write_raw_payload`'s default stamp is second-resolution. Tenacity can fire
    multiple retries within the same wall-second on transient 5xx bursts, which
    without a disambiguator would let later saves silently overwrite earlier
    ones. We attach a microsecond + short payload-hash `name_suffix` so each
    retry's verbatim envelope lands on disk under a distinct path (and
    `write_raw_payload` refuses to overwrite when a suffix is provided).
    """
    payload = build_raw_responses_envelope(raw_responses, request_log)
    now = collected_now_utc()
    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()[:8]
    disambiguator = f"{now.microsecond:06d}_{digest}"
    return write_raw_payload(
        source_name, payload, suffix=".json", name_suffix=disambiguator
    )


def build_raw_responses_envelope(
    raw_responses: list[dict],
    request_log: dict,
) -> str:
    """Build the canonical raw-payload JSON for multi-request data.go.kr collectors.

    Schema:
        {
          "responses": [
            {"region_name": ..., "status_code": ..., "url": <REDACTED>,
             "body": "<verbatim response.text>"},
            ...
          ],
          "request_log": {...}
        }

    The `body` field is the verbatim server response (typically JSON text);
    we never re-serialize it. This preserves the full envelope including
    `response.header.resultCode/resultMsg` for audit and re-parsing.
    """
    import json as _json
    return _json.dumps(
        {"responses": raw_responses, "request_log": request_log},
        ensure_ascii=False, indent=2,
    )


# Most-recent KMA mid-term issuance immediately before `now` (06 or 18 KST).
def _latest_tm_fc(now: datetime) -> str:
    hour = now.hour
    if hour >= 18:
        return now.strftime("%Y%m%d") + "1800"
    if hour >= 6:
        return now.strftime("%Y%m%d") + "0600"
    prev = now - timedelta(days=1)
    return prev.strftime("%Y%m%d") + "1800"


class KmaMidTemperatureCollector(BaseCollector):
    """One request per regId; we concatenate before parsing."""

    source_name = "kma_mid_temperature"

    @retry(
        reraise=True,
        retry=retry_if_exception_type(TransientCollectorError),
        stop=stop_after_attempt(4),
        wait=wait_exponential(multiplier=1, min=1, max=20),
    )
    def fetch_one(self, **kwargs: Any) -> FetchResult:
        settings = get_settings()
        # Override via kwargs: callers can pass api_key + is_encoded explicitly.
        kw_key = kwargs.get("api_key")
        if kw_key is not None:
            api_key = kw_key
            is_encoded = bool(kwargs.get("is_encoded", False))
        else:
            api_key, is_encoded = _resolve_data_go_kr_service_key(settings)

        base_url = self.config.get("base_url",
            "http://apis.data.go.kr/1360000/MidFcstInfoService/getMidTa")
        reg_ids = (
            kwargs.get("reg_ids")
            or self.config.get("default_region_ids")
            or ["11B10101"]  # Seoul as a safe default
        )
        tm_fc = kwargs.get("tm_fc") or _latest_tm_fc(datetime.utcnow() + timedelta(hours=9))

        all_items: list[dict[str, Any]] = []
        raw_responses: list[dict[str, Any]] = []
        # Live service-key forms to scrub from response bodies. data.go.kr
        # error envelopes (SERVICE_KEY_IS_NOT_REGISTERED_ERROR etc.) often
        # echo the submitted serviceKey inside <returnAuthMsg>/<resultMsg>,
        # which would otherwise land on disk verbatim — defeating the
        # URL-level redaction we already do.
        key_secrets = _service_key_variants(api_key, is_encoded)
        request_log = {
            "base_url": base_url, "tm_fc": tm_fc, "reg_ids": list(reg_ids),
            "service_key_form": "encoded" if is_encoded else "raw",
        }

        for reg in reg_ids:
            other_params = {
                "pageNo": 1, "numOfRows": 10,
                "dataType": "JSON",
                "regId": reg,
                "tmFc": tm_fc,
            }
            try:
                r = _get_with_data_go_kr_key(
                    base_url, api_key, is_encoded, other_params, timeout=30,
                )
            except requests.RequestException as e:
                # urllib3 / requests exceptions can embed the full request
                # URL (with serviceKey) in `str(e)`. Two leak vectors here:
                #   (1) our own f-string — fixed by `safe_err`
                #   (2) the chained cause: `logger.exception(...)` walks
                #       `__cause__`/`__context__` and prints the ORIGINAL
                #       exception's str() in the traceback. So we use
                #       `from None` to suppress the chain. The exception
                #       type/class name still surfaces in `safe_err`.
                safe_err = redact_secrets_in_text(f"{type(e).__name__}: {e}", key_secrets)
                raise TransientCollectorError(
                    f"KMA mid request failed: {safe_err}"
                ) from None
            # Capture verbatim server payload BEFORE branching on status, so
            # audits can see error envelopes from data.go.kr too. The body is
            # scrubbed of any echoed serviceKey value first.
            raw_responses.append({
                "reg_id": reg,
                "tm_fc": tm_fc,
                "status_code": r.status_code,
                "url": redact_service_key_in_url(r.request.url or base_url),
                "body": redact_secrets_in_text(r.text, key_secrets),
            })
            # Build the body excerpt with key values scrubbed BEFORE it can
            # end up in an exception message — exceptions get logged by
            # BaseCollector.collect() via `self.log.exception(...)`, so a
            # raw `r.text[:200]` that echoes the serviceKey would leak the
            # key into the log file even though the on-disk raw envelope
            # is now scrubbed.
            body_excerpt = redact_secrets_in_text(r.text, key_secrets)[:200]
            if r.status_code in {429, 500, 502, 503, 504}:
                saved = persist_partial_error_envelope(
                    self.source_name, raw_responses, request_log
                )
                raise TransientCollectorError(
                    f"KMA mid transient {r.status_code} (raw saved: {saved})"
                )
            if not r.ok:
                saved = persist_partial_error_envelope(
                    self.source_name, raw_responses, request_log
                )
                raise CollectorError(
                    f"KMA mid {r.status_code}: {body_excerpt} (raw saved: {saved})"
                )
            try:
                body = r.json()
            except ValueError as e:
                saved = persist_partial_error_envelope(
                    self.source_name, raw_responses, request_log
                )
                raise CollectorError(
                    f"KMA mid: non-JSON response (likely SERVICE_KEY error). "
                    f"First 200 chars: {body_excerpt} (raw saved: {saved})"
                ) from e
            # data.go.kr error envelopes return HTTP 200 with a non-"00" resultCode
            # (e.g. SERVICE_KEY_IS_NOT_REGISTERED_ERROR). Persist before raising so
            # the verbatim envelope is auditable. Scrub the header fields BEFORE
            # they enter the exception message — `resultMsg` often embeds the
            # submitted serviceKey value in the error text.
            header = body.get("response", {}).get("header", {})
            if header.get("resultCode") not in {"00", None}:
                saved = persist_partial_error_envelope(
                    self.source_name, raw_responses, request_log
                )
                safe_code = redact_secrets_in_text(
                    str(header.get("resultCode") or ""), key_secrets
                )
                safe_msg = redact_secrets_in_text(
                    str(header.get("resultMsg") or ""), key_secrets
                )
                raise CollectorError(
                    f"KMA mid result {safe_code}: {safe_msg} (raw saved: {saved})"
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
                it["_reg_id"] = reg
                it["_tm_fc"] = tm_fc
            all_items.extend(items)

        raw_text = build_raw_responses_envelope(raw_responses, request_log)
        return FetchResult(
            raw_text=raw_text,
            raw_suffix=".json",
            # `parsed` is what `parse()` consumes; we keep the aggregated items
            # for backwards compatibility. The verbatim per-region bodies live
            # in raw_text under "responses[].body".
            parsed={"items": all_items},
            request={"url": base_url, "params": request_log},
        )

    def parse(self, fetch_result: FetchResult) -> pd.DataFrame:
        parsed = fetch_result.parsed
        items = parsed.get("items", [])
        if not items:
            return pd.DataFrame(columns=[
                "source_id", "reg_id", "tm_fc",
                "forecast_date", "ta_min", "ta_max",
            ])
        long_rows: list[dict[str, Any]] = []
        # KMA returns one record per (regId, tm_fc) with columns
        # taMin3..taMin10 and taMax3..taMax10 — one per forecast lead day.
        for rec in items:
            reg = rec.get("_reg_id") or rec.get("regId")
            tm_fc = rec.get("_tm_fc")
            issue_ts = pd.to_datetime(tm_fc, format="%Y%m%d%H%M", errors="coerce")
            issue_date = issue_ts.normalize() if pd.notna(issue_ts) else pd.NaT
            for lead in range(3, 11):
                ta_min_key = f"taMin{lead}"
                ta_max_key = f"taMax{lead}"
                if ta_min_key not in rec and ta_max_key not in rec:
                    continue
                forecast_date = (issue_date + pd.Timedelta(days=lead)) if pd.notna(issue_date) else pd.NaT
                long_rows.append({
                    "source_id": self.source_name,
                    "reg_id": reg,
                    "tm_fc": str(tm_fc),
                    "forecast_date": forecast_date,
                    "lead_days": lead,
                    "ta_min": pd.to_numeric(rec.get(ta_min_key), errors="coerce"),
                    "ta_max": pd.to_numeric(rec.get(ta_max_key), errors="coerce"),
                })
        df = pd.DataFrame(long_rows)
        if df.empty:
            return df
        return (
            df.dropna(subset=["forecast_date"])
              .sort_values(["reg_id", "forecast_date"])
              .reset_index(drop=True)
        )
