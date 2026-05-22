"""BaseCollector — every collector must save raw response + parsed dataframe +
collection timestamp + source metadata, as required by Plan.md §15.1."""

from __future__ import annotations

import abc
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Optional

import pandas as pd
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from src.config.settings import get_source_config, get_source_defaults
from src.utils.io import RawSavePaths, persist_collector_output
from src.utils.logging import get_logger
from src.utils.time import collected_now_utc


class CollectorError(RuntimeError):
    """Raised on unrecoverable collector failures."""


class TransientCollectorError(CollectorError):
    """Raised on errors worth retrying (network/5xx/rate-limit)."""


@dataclass
class FetchResult:
    """Container returned by `fetch()`.

    raw_text:    the verbatim payload (JSON or XML) as a string for persistence
    raw_suffix:  '.json' or '.xml'
    parsed:      the parsed payload (dict for JSON, OrderedDict for XML)
    request:     dict capturing request URL/params for the metadata file
    """

    raw_text: str
    raw_suffix: str
    parsed: Any
    request: dict[str, Any]


@dataclass
class CollectorRun:
    run_id: str
    source: str
    started_at: datetime
    finished_at: Optional[datetime] = None
    params: dict[str, Any] = field(default_factory=dict)
    raw_paths: list[RawSavePaths] = field(default_factory=list)
    row_count: int = 0
    status: str = "running"
    error: Optional[str] = None

    def to_metadata(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "source": self.source,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "params": self.params,
            "row_count": self.row_count,
            "status": self.status,
            "error": self.error,
        }


class BaseCollector(abc.ABC):
    """Subclasses implement `fetch_one()` and `parse()` for their source."""

    source_name: str  # must match a key in sources.yaml

    def __init__(self) -> None:
        if not getattr(self, "source_name", None):
            raise ValueError(f"{type(self).__name__}.source_name must be set")
        self.config = get_source_config(self.source_name)
        self.defaults = get_source_defaults()
        self.log = get_logger(self.source_name)

    # ---- Subclass hooks -----------------------------------------------------

    @abc.abstractmethod
    def fetch_one(self, **params: Any) -> FetchResult:
        """Make a single API call. Should raise TransientCollectorError on retryable errors."""

    @abc.abstractmethod
    def parse(self, fetch_result: FetchResult) -> pd.DataFrame:
        """Convert a FetchResult into a tidy dataframe."""

    def validate_schema(self, df: pd.DataFrame) -> None:
        """Per-source schema check. Default: ensure not empty + log unknown columns."""
        if df.empty:
            raise CollectorError(f"{self.source_name}: parsed dataframe is empty.")
        verified = set(self.config.get("verified_columns") or [])
        if verified and "TBD_AFTER_FIRST_RESPONSE" not in verified:
            unknown = set(df.columns) - verified
            if unknown:
                self.log.warning(
                    "Unknown columns (not in verified_columns): %s", sorted(unknown)
                )

    # ---- Public orchestration ----------------------------------------------

    @retry(
        retry=retry_if_exception_type(TransientCollectorError),
        wait=wait_exponential(multiplier=1, min=2, max=30),
        stop=stop_after_attempt(5),
        reraise=True,
    )
    def fetch_with_retry(self, **params: Any) -> FetchResult:
        return self.fetch_one(**params)

    def collect(self, **params: Any) -> CollectorRun:
        run = CollectorRun(
            run_id=str(uuid.uuid4()),
            source=self.source_name,
            started_at=collected_now_utc(),
            params=params,
        )
        try:
            fetch_result = self.fetch_with_retry(**params)
            df = self.parse(fetch_result)
            self.validate_schema(df)
            run.row_count = len(df)

            metadata = {
                **run.to_metadata(),
                "collected_at": run.started_at,
                "request": fetch_result.request,
                "schema_version": self.config.get("schema_version"),
                "source_official_name": self.config.get("official_name"),
            }
            paths = persist_collector_output(
                self.source_name,
                raw_text=fetch_result.raw_text,
                raw_suffix=fetch_result.raw_suffix,
                parsed_df=df,
                metadata=metadata,
            )
            run.raw_paths.append(paths)
            run.status = "ok"
        except Exception as exc:  # noqa: BLE001 — we want the failure recorded
            run.status = "error"
            run.error = repr(exc)
            self.log.exception("Collector run failed: %s", exc)
            raise
        finally:
            run.finished_at = collected_now_utc()
        return run

    # ---- Schema discovery ---------------------------------------------------

    def discover_schema(self, **params: Any) -> dict[str, Any]:
        """Fetch once and report what fields appeared, without persisting parsed data.

        Use this when populating sources.yaml for a brand-new endpoint:

            python -m src.pipelines.discover_schema --source kpx_smp_day_ahead ...

        The actual CLI is in src/pipelines/discover_schema.py.
        """
        fetch_result = self.fetch_with_retry(**params)
        sample = _sample_keys(fetch_result.parsed)
        return {
            "request": fetch_result.request,
            "raw_excerpt": fetch_result.raw_text[:2000],
            "discovered_keys": sample,
        }


def _sample_keys(obj: Any, depth: int = 3) -> Any:
    """Return a structural summary of nested keys for schema discovery."""
    if depth < 0:
        return type(obj).__name__
    if isinstance(obj, dict):
        return {k: _sample_keys(v, depth - 1) for k, v in obj.items()}
    if isinstance(obj, list):
        if not obj:
            return []
        return [_sample_keys(obj[0], depth - 1), f"(+{len(obj) - 1} more)"]
    return type(obj).__name__
