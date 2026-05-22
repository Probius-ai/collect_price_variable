"""KPX SMP (계통한계가격) + day-ahead demand forecast collector.

Important:
    * We do NOT hard-code vendor field names. The mapping from vendor key to
      our canonical column is read from `sources.yaml::sources.kpx_smp_day_ahead
      .column_mapping`. When that mapping still contains
      `TBD_AFTER_FIRST_RESPONSE`, callers must run the schema-discovery CLI
      first, inspect `data/raw/kpx/smp/.../response_*.json`, and fill the YAML.
    * `parse()` will refuse to materialize unverified rows so we never invent
      fields silently.

The base URL and operation path also live in sources.yaml and must be filled
once verified from the data.go.kr documentation page.
"""

from __future__ import annotations

import json
from datetime import date, datetime
from typing import Any, Iterable

import pandas as pd
import requests
import xmltodict

from src.collectors.base import (
    BaseCollector,
    CollectorError,
    FetchResult,
    TransientCollectorError,
)
from src.config.settings import get_settings
from src.utils.time import kpx_trade_hour_to_interval_end, kpx_trade_hour_to_interval_start

TBD = "TBD_AFTER_FIRST_RESPONSE"
INTERNAL_COLUMNS = ["area", "trade_date", "trade_hour", "smp_krw_per_kwh", "demand_forecast_mw"]


class KpxSmpCollector(BaseCollector):
    source_name = "kpx_smp_day_ahead"

    # ---- HTTP ---------------------------------------------------------------

    def _resolve_endpoint(self) -> str:
        base = self.config.get("base_url")
        op = self.config.get("operation")
        if not base or not op or TBD in {base, op}:
            raise CollectorError(
                "kpx_smp_day_ahead.base_url and .operation are TBD. "
                "Set them in src/config/sources.yaml from the data.go.kr page "
                "for '계통한계가격 및 수요예측(하루전 발전계획용)'."
            )
        return f"{base.rstrip('/')}/{op.lstrip('/')}"

    def _build_params(self, **caller_params: Any) -> dict[str, Any]:
        settings = get_settings()
        if not settings.kpx_public_api_key:
            raise CollectorError(
                "KPX_PUBLIC_API_KEY is not set. Copy .env.example to .env and fill it."
            )
        defaults = self.config.get("request_params", {})
        params: dict[str, Any] = {
            defaults.get("service_key", "serviceKey"): settings.kpx_public_api_key,
            defaults.get("page_no", "pageNo"): caller_params.pop("page_no", 1),
            defaults.get("num_of_rows", "numOfRows"): caller_params.pop("num_of_rows", 1000),
            defaults.get("type", "type"): self.config.get("response_format", "json"),
        }
        date_param = defaults.get("base_date_param")
        base_date = caller_params.pop("base_date", None)
        if date_param and date_param != TBD and base_date is not None:
            if isinstance(base_date, (datetime, date)):
                base_date = base_date.strftime("%Y%m%d")
            params[date_param] = base_date
        elif base_date is not None and date_param == TBD:
            self.log.warning(
                "Caller passed base_date=%s but request_params.base_date_param is still TBD. "
                "The API call will likely ignore the date filter.",
                base_date,
            )
        # any extra caller_params are passed through verbatim
        params.update(caller_params)
        return params

    def fetch_one(self, **caller_params: Any) -> FetchResult:
        url = self._resolve_endpoint()
        params = self._build_params(**caller_params)
        timeout = self.defaults.get("request_timeout_seconds", 30)
        self.log.info("GET %s params=%s", url, _redact_key(params))
        try:
            resp = requests.get(url, params=params, timeout=timeout)
        except requests.RequestException as exc:
            raise TransientCollectorError(f"request failed: {exc!r}") from exc

        if resp.status_code >= 500 or resp.status_code in (408, 429):
            raise TransientCollectorError(
                f"transient HTTP {resp.status_code}: {resp.text[:200]}"
            )
        if resp.status_code >= 400:
            raise CollectorError(f"HTTP {resp.status_code}: {resp.text[:500]}")

        text = resp.text
        content_type = resp.headers.get("Content-Type", "")
        if "json" in content_type or text.lstrip().startswith(("{", "[")):
            parsed = json.loads(text)
            suffix = ".json"
        else:
            parsed = xmltodict.parse(text)
            suffix = ".xml"

        return FetchResult(
            raw_text=text,
            raw_suffix=suffix,
            parsed=parsed,
            request={"url": url, "params": _redact_key(params)},
        )

    # ---- Parsing ------------------------------------------------------------

    def parse(self, fetch_result: FetchResult) -> pd.DataFrame:
        mapping = self.config.get("column_mapping", {})
        unresolved = [k for k, v in mapping.items() if v == TBD]
        if unresolved:
            raise CollectorError(
                "column_mapping for kpx_smp_day_ahead has unresolved TBD entries: "
                f"{unresolved}. Run the schema-discovery CLI and fill them in "
                "src/config/sources.yaml before parsing."
            )

        rows = list(_iter_items(fetch_result.parsed))
        if not rows:
            raise CollectorError(
                "No item rows found in response. Inspect the raw payload at "
                "data/raw/kpx/smp/... to confirm response structure."
            )

        df = pd.DataFrame(rows)
        # rename vendor columns → internal names; preserve unmapped columns for audit
        renames = {vendor: internal for internal, vendor in mapping.items() if vendor in df.columns}
        df = df.rename(columns=renames)

        required = {"area", "trade_date", "trade_hour", "smp_krw_per_kwh", "demand_forecast_mw"}
        missing = required - set(df.columns)
        if missing:
            raise CollectorError(
                f"After renaming, required internal columns missing: {sorted(missing)}. "
                "Check column_mapping in sources.yaml against the actual response."
            )

        df["trade_date"] = pd.to_datetime(df["trade_date"], errors="raise").dt.date
        df["trade_hour"] = df["trade_hour"].astype(int)
        df["smp_krw_per_kwh"] = pd.to_numeric(df["smp_krw_per_kwh"], errors="coerce")
        df["demand_forecast_mw"] = pd.to_numeric(df["demand_forecast_mw"], errors="coerce")
        df["interval_start"] = df.apply(
            lambda r: kpx_trade_hour_to_interval_start(
                datetime.combine(r["trade_date"], datetime.min.time()), r["trade_hour"]
            ),
            axis=1,
        )
        df["interval_end"] = df.apply(
            lambda r: kpx_trade_hour_to_interval_end(
                datetime.combine(r["trade_date"], datetime.min.time()), r["trade_hour"]
            ),
            axis=1,
        )
        df["source_name"] = self.source_name
        df["schema_version"] = self.config.get("schema_version", 0)
        return df.sort_values(["area", "interval_end"]).reset_index(drop=True)


def _iter_items(parsed: Any) -> Iterable[dict[str, Any]]:
    """Walk a (typical) data.go.kr response to find the list of item dicts.

    KPX/data.go.kr responses commonly look like:
        {"response": {"header": {...}, "body": {"items": {"item": [...]}}}}
    or {"response": {"body": {"items": [...] }}} or a flat list.
    We try several shapes; whatever we find should be a list of dicts.
    """
    candidates: list[Any] = []

    def visit(node: Any) -> None:
        if isinstance(node, dict):
            for key, value in node.items():
                if key in {"item", "items", "data"} and isinstance(value, (list, dict)):
                    candidates.append(value)
                visit(value)
        elif isinstance(node, list):
            for entry in node:
                visit(entry)

    visit(parsed)

    for cand in candidates:
        if isinstance(cand, list) and cand and isinstance(cand[0], dict):
            yield from cand
            return
        if isinstance(cand, dict):
            # XML often nests as {'item': [...]}.
            inner = cand.get("item")
            if isinstance(inner, list):
                yield from inner
                return
            if isinstance(inner, dict):
                yield inner
                return

    # Fall back: maybe the whole payload is already a list of rows.
    if isinstance(parsed, list):
        yield from (r for r in parsed if isinstance(r, dict))


def _redact_key(params: dict[str, Any]) -> dict[str, Any]:
    redacted = dict(params)
    for k in list(redacted):
        if "key" in k.lower() or "secret" in k.lower():
            redacted[k] = "***REDACTED***"
    return redacted
