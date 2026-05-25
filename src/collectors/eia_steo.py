"""EIA STEO (Short-Term Energy Outlook) API collector.

Publishes monthly forecasts of energy prices including Brent crude, WTI,
Henry Hub natural gas, and other LNG-related series. We pull the
historical + forecast series for the macro variables that drive LNG and
SMP, and persist them as a long-format parquet.

API docs: https://www.eia.gov/opendata/documentation.php
STEO browser: https://www.eia.gov/outlooks/steo/data/browser/

The series IDs we care about (set in sources.yaml::sources::eia_steo):
  * STEO.BREPUUS.M   — Brent crude spot ($/bbl)
  * STEO.WTIPUUS.M   — WTI crude spot ($/bbl)
  * STEO.NGHHMCF.M   — Henry Hub natural gas spot ($/MMBtu)
"""

from __future__ import annotations

import json
from typing import Any

import pandas as pd
import requests
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from src.collectors.base import BaseCollector, CollectorError, FetchResult, TransientCollectorError
from src.config.settings import get_settings
from src.utils.redact import redact_secrets_in_text, secret_variants


# Default series we pull when sources.yaml doesn't override.
_DEFAULT_SERIES = [
    "STEO.BREPUUS.M",  # Brent ($/bbl) monthly
    "STEO.WTIPUUS.M",  # WTI ($/bbl) monthly
    "STEO.NGHHMCF.M",  # Henry Hub natural gas ($/MMBtu) monthly
]


class EiaSteoCollector(BaseCollector):
    """Pull EIA STEO monthly forecast series into a single long parquet."""

    source_name = "eia_steo"

    @retry(
        reraise=True,
        retry=retry_if_exception_type(TransientCollectorError),
        stop=stop_after_attempt(4),
        wait=wait_exponential(multiplier=1, min=1, max=30),
    )
    def fetch_one(self, **kwargs: Any) -> FetchResult:
        settings = get_settings()
        api_key = getattr(settings, "eia_api_key", None) or kwargs.get("api_key")
        if not api_key:
            raise CollectorError(
                "EIA_API_KEY is not set. Register at https://www.eia.gov/opendata/register.php "
                "and add `EIA_API_KEY=...` to .env."
            )
        series_ids = (
            self.config.get("series_ids")
            or kwargs.get("series_ids")
            or _DEFAULT_SERIES
        )
        base_url = self.config.get("base_url", "https://api.eia.gov/v2/steo/data/")

        params = {
            "api_key": api_key,
            "frequency": kwargs.get("frequency", "monthly"),
            "data[0]": "value",
            "facets[seriesId][]": series_ids,
            "sort[0][column]": "period",
            "sort[0][direction]": "asc",
            "length": kwargs.get("length", 5000),
        }
        # Build the full set of key variants to scrub. `requests.get(params=...)`
        # URL-encodes the value before it hits the wire, so any urllib3
        # exception surfacing the URL would carry the encoded form — a
        # plain `str.replace(raw_key, ...)` would miss that.
        key_variants = secret_variants(api_key)

        try:
            r = requests.get(base_url, params=params, timeout=30)
        except requests.RequestException as e:
            # Two leak vectors: our own f-string AND the cause chain that
            # logger.exception() prints. Scrub the first, suppress the
            # second with `from None`.
            inner = f"{type(e).__name__}: {e}"
            safe_err = redact_secrets_in_text(inner, key_variants)
            raise TransientCollectorError(
                f"EIA STEO request failed: {safe_err}"
            ) from None
        if r.status_code in {429, 500, 502, 503, 504}:
            raise TransientCollectorError(f"EIA STEO transient {r.status_code}")
        # Scrub the body excerpt before it lands in an exception message
        # (exceptions are logged by BaseCollector.collect via self.log.exception).
        # The EIA error body for an invalid key includes "INVALID_API_KEY"
        # and may echo the submitted key in the description field — in
        # either raw OR URL-encoded form depending on the endpoint.
        body_excerpt = redact_secrets_in_text(r.text, key_variants)[:200]
        if not r.ok:
            raise CollectorError(f"EIA STEO {r.status_code}: {body_excerpt}")

        payload = r.json()
        # Persist what we actually requested (incl. api_key REDACTED) AND
        # scrub the verbatim response body of any echoed key value.
        request_log = {**params, "api_key": "<REDACTED>"}
        safe_raw_text = redact_secrets_in_text(r.text, key_variants)
        return FetchResult(
            raw_text=safe_raw_text,
            raw_suffix=".json",
            parsed=payload,
            request={"url": base_url, "params": request_log},
        )

    def parse(self, fetch_result: FetchResult) -> pd.DataFrame:
        """Long-format frame: source_id, period_month, series_id, value, units."""
        parsed = fetch_result.parsed
        response = parsed.get("response", {})
        rows = response.get("data", [])
        if not rows:
            return pd.DataFrame(columns=[
                "source_id", "period_month", "series_id", "value", "units"
            ])
        df = pd.DataFrame(rows)
        # EIA returns 'period' as 'YYYY-MM' for monthly.
        df["period_month"] = pd.to_datetime(df["period"]).dt.to_period("M").dt.to_timestamp()
        df["series_id"] = df["seriesId"]
        df["value"] = pd.to_numeric(df["value"], errors="coerce")
        # Units may not always be present per row.
        if "unit" in df.columns:
            df["units"] = df["unit"]
        else:
            df["units"] = ""
        df["source_id"] = self.source_name
        return (
            df[["source_id", "period_month", "series_id", "value", "units"]]
            .dropna(subset=["period_month", "value"])
            .sort_values(["series_id", "period_month"])
            .reset_index(drop=True)
        )
