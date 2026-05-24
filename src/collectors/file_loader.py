"""BaseFileLoader — file-based equivalent of BaseCollector.

The pre-approval workflow:

1. User downloads a CSV/XLSX from KPX/EPSIS into
   ``data/raw/manual_or_filedata/<source_name>/``.
2. Loader reads it via the column-mapping in ``sources.yaml::file_sources``.
3. Like the API collectors, it persists: a copy of the raw file under the
   date-partitioned ``data/raw/<ns>/<rest>/YYYY/MM/DD/`` layout, the parsed
   Parquet dataframe, and a JSON metadata sidecar carrying source_url,
   collected_at, frequency, unit, and limitations.

A future API collector for the same dataset (same canonical column names)
can drop a `parsed_*.parquet` into the same date-partitioned directory and
the downstream feature builder will pick it up without changes.
"""

from __future__ import annotations

import abc
import hashlib
import shutil
import uuid
from dataclasses import dataclass, field
from datetime import date, datetime
from pathlib import Path
from typing import Any, Iterable

import pandas as pd

from src.config.settings import get_source_config, get_source_defaults
from src.utils.io import (
    RawSavePaths,
    manual_drop_dir,
    raw_response_dir,
    write_metadata,
    write_parsed_dataframe,
)
from src.utils.logging import get_logger
from src.utils.time import collected_now_utc

TBD = "TBD_AFTER_FIRST_DOWNLOAD"


class FileLoaderError(RuntimeError):
    """Raised when a file source cannot be parsed."""


@dataclass
class FileLoadRun:
    run_id: str
    source: str
    started_at: datetime
    file_path: Path
    finished_at: datetime | None = None
    paths: RawSavePaths | None = None
    row_count: int = 0
    status: str = "running"
    error: str | None = None
    file_sha256: str | None = None

    def to_metadata(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "source": self.source,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "file_path": str(self.file_path),
            "file_sha256": self.file_sha256,
            "row_count": self.row_count,
            "status": self.status,
            "error": self.error,
        }


class BaseFileLoader(abc.ABC):
    """Subclass for each file-based dataset (one per `file_sources:` entry)."""

    source_name: str  # must match a key in sources.yaml::file_sources

    def __init__(self) -> None:
        if not getattr(self, "source_name", None):
            raise ValueError(f"{type(self).__name__}.source_name must be set")
        self.config = get_source_config(self.source_name)
        self.defaults = get_source_defaults()
        self.log = get_logger(self.source_name)

    # ---- Subclass hooks -----------------------------------------------------

    @abc.abstractmethod
    def parse_file(self, file_path: Path) -> pd.DataFrame:
        """Read the file at `file_path` and return the canonical dataframe."""

    def find_files(self) -> list[Path]:
        """Return CSV/XLSX files in the drop directory for this source."""
        directory = manual_drop_dir(self.source_name)
        directory.mkdir(parents=True, exist_ok=True)
        files = sorted(
            p for p in directory.iterdir()
            if p.is_file() and p.suffix.lower() in {".csv", ".xlsx", ".xls", ".tsv"}
        )
        return files

    # ---- Public orchestration ----------------------------------------------

    def load_one(self, file_path: Path) -> FileLoadRun:
        run = FileLoadRun(
            run_id=str(uuid.uuid4()),
            source=self.source_name,
            started_at=collected_now_utc(),
            file_path=file_path,
        )
        try:
            run.file_sha256 = _sha256(file_path)
            df = self.parse_file(file_path)
            _validate_canonical_columns(df, expected=self.required_columns())
            run.row_count = len(df)

            collected_at = run.started_at
            event_date = _event_date_for_dataframe(df) or date.today()
            target_dir = raw_response_dir(self.source_name, event_date)
            target_dir.mkdir(parents=True, exist_ok=True)

            raw_copy_path = target_dir / f"raw_{_stamp(collected_at)}{file_path.suffix.lower()}"
            shutil.copy2(file_path, raw_copy_path)

            # Stamp every parsed row with the revision-metadata the downstream
            # feature builder needs to deterministically pick the latest snapshot
            # when the same (period_month, area) is loaded more than once.
            # Compute the future parsed_path BEFORE writing so it matches the
            # path that write_parsed_dataframe will produce (same `collected_at`
            # → same `_stamp`).
            future_parsed_path = target_dir / f"parsed_{_stamp(collected_at)}.parquet"
            df = df.copy()
            df["source_file_sha256"] = run.file_sha256
            df["source_priority"] = self.config.get("source_priority")
            df["parsed_path"] = str(future_parsed_path)

            parsed_path = write_parsed_dataframe(
                self.source_name, df, event_date=event_date, collected_at=collected_at
            )
            metadata = {
                **run.to_metadata(),
                "collected_at": collected_at,
                "source_id": self.config.get("source_id", self.source_name),
                "source_priority": self.config.get("source_priority"),
                "source_official_name": self.config.get("official_name"),
                "source_url": self.config.get("source_url"),
                "frequency": self.config.get("frequency"),
                "region": self.config.get("region"),
                "output_areas": self.config.get("output_areas"),
                "unit": self.config.get("unit") or self.config.get("units"),
                "limitations": self.config.get("limitations", []),
                "schema_version": self.config.get("schema_version"),
                "raw_copy_path": str(raw_copy_path),
                "parsed_path": str(parsed_path),
                # DQ-report side-info the loader may have attached via df.attrs
                "dropped_columns": df.attrs.get("_dropped_columns", []),
                "missing_canonical_categories": df.attrs.get(
                    "_missing_canonical_fuels", []
                ),
                # Loaders that coerce vendor "0 = missing" markers report the
                # per-category count here. SMP loaders use this.
                "zero_coerced_rows": df.attrs.get("_zero_coerced_rows", {}),
            }
            meta_path = write_metadata(
                self.source_name, metadata, event_date=event_date, collected_at=collected_at
            )
            run.paths = RawSavePaths(
                raw_response=raw_copy_path, parsed_dataframe=parsed_path, metadata=meta_path
            )
            run.status = "ok"
        except Exception as exc:  # noqa: BLE001 — capture and rethrow with context
            run.status = "error"
            run.error = repr(exc)
            self.log.exception("File load failed for %s: %s", file_path, exc)
            raise
        finally:
            run.finished_at = collected_now_utc()
        return run

    def load_all(self) -> list[FileLoadRun]:
        files = self.find_files()
        if not files:
            raise FileLoaderError(
                f"No files in {manual_drop_dir(self.source_name)} for source "
                f"{self.source_name}. Drop the downloaded CSV/XLSX there first."
            )
        runs: list[FileLoadRun] = []
        for path in files:
            runs.append(self.load_one(path))
        return runs

    # ---- Helpers used by subclasses ----------------------------------------

    def required_columns(self) -> set[str]:
        """Canonical column names a loader must emit."""
        return set(self.config.get("column_mapping", {}).keys())

    def assert_no_tbd_in_config(self, *keys: str) -> None:
        for key in keys:
            value = self.config.get(key)
            if isinstance(value, str) and value == TBD:
                raise FileLoaderError(
                    f"sources.yaml::file_sources.{self.source_name}.{key} is still TBD. "
                    "Run the discover-file CLI and update sources.yaml before loading."
                )
            if isinstance(value, dict):
                for sub_key, sub_value in value.items():
                    if isinstance(sub_value, str) and sub_value == TBD:
                        raise FileLoaderError(
                            f"sources.yaml::file_sources.{self.source_name}.{key}."
                            f"{sub_key} is still TBD."
                        )


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def _validate_canonical_columns(df: pd.DataFrame, expected: Iterable[str]) -> None:
    missing = sorted(set(expected) - set(df.columns))
    if missing:
        raise FileLoaderError(
            f"Parsed dataframe is missing canonical columns: {missing}. "
            "Check sources.yaml column_mapping against the actual file header."
        )


def _event_date_for_dataframe(df: pd.DataFrame) -> date | None:
    """Pick a representative date for raw-response partitioning."""
    for col in (
        "period_month",          # new long-format monthly schemas
        "trade_date",            # new REC weekly long-format
        "year_month",            # legacy
        "week_start_date",       # legacy
        "year",
        "observed_at",
    ):
        if col in df.columns and len(df):
            try:
                val = pd.to_datetime(df[col].iloc[-1])
                return val.date() if not pd.isna(val) else None
            except Exception:
                continue
    return None


def _stamp(ts: datetime | None = None) -> str:
    ts = ts or collected_now_utc()
    return ts.strftime("%Y%m%dT%H%M%SZ")
