"""I/O helpers: raw payload persistence, snapshot CSV writes, DuckDB writes."""

from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from datetime import datetime, date
from pathlib import Path
from typing import Any

import pandas as pd

from src.config.settings import get_settings
from src.utils.logging import get_logger
from src.utils.time import collected_now_utc

logger = get_logger(__name__)


@dataclass
class RawSavePaths:
    raw_response: Path
    parsed_dataframe: Path
    metadata: Path


def _stamp(ts: datetime | None = None) -> str:
    ts = ts or collected_now_utc()
    return ts.strftime("%Y%m%dT%H%M%SZ")


def raw_response_dir(source: str, event_date: date | None = None) -> Path:
    """Standard layout: data/raw/<source-namespace>/YYYY/MM/DD/."""
    settings = get_settings()
    base = settings.data_dir / "raw"
    if "_" in source:
        ns, _, rest = source.partition("_")
        sub = Path(ns) / rest
    else:
        sub = Path(source)
    if event_date is None:
        event_date = date.today()
    return base / sub / f"{event_date:%Y}" / f"{event_date:%m}" / f"{event_date:%d}"


def write_raw_payload(
    source: str,
    payload_text: str,
    *,
    event_date: date | None = None,
    suffix: str = ".json",
    collected_at: datetime | None = None,
) -> Path:
    directory = raw_response_dir(source, event_date)
    directory.mkdir(parents=True, exist_ok=True)
    stamp = _stamp(collected_at)
    path = directory / f"response_{stamp}{suffix}"
    path.write_text(payload_text, encoding="utf-8")
    logger.info("Wrote raw payload: %s (%d bytes)", path, len(payload_text))
    return path


def write_parsed_dataframe(
    source: str,
    df: pd.DataFrame,
    *,
    event_date: date | None = None,
    collected_at: datetime | None = None,
) -> Path:
    directory = raw_response_dir(source, event_date)
    directory.mkdir(parents=True, exist_ok=True)
    stamp = _stamp(collected_at)
    path = directory / f"parsed_{stamp}.parquet"
    df.to_parquet(path, index=False)
    logger.info("Wrote parsed dataframe: %s (rows=%d)", path, len(df))
    return path


def write_metadata(
    source: str,
    metadata: dict[str, Any],
    *,
    event_date: date | None = None,
    collected_at: datetime | None = None,
) -> Path:
    directory = raw_response_dir(source, event_date)
    directory.mkdir(parents=True, exist_ok=True)
    stamp = _stamp(collected_at)
    path = directory / f"metadata_{stamp}.json"
    path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    logger.info("Wrote metadata: %s", path)
    return path


def write_snapshot_csv(source: str, df: pd.DataFrame, name: str | None = None) -> Path:
    settings = get_settings()
    directory = settings.data_dir / "snapshots" / source
    directory.mkdir(parents=True, exist_ok=True)
    stamp = _stamp()
    fname = f"{name or 'snapshot'}_{stamp}.csv"
    path = directory / fname
    df.to_csv(path, index=False, encoding="utf-8-sig")
    logger.info("Wrote snapshot CSV: %s (rows=%d)", path, len(df))
    return path


def persist_collector_output(
    source: str,
    *,
    raw_text: str,
    raw_suffix: str,
    parsed_df: pd.DataFrame,
    metadata: dict[str, Any],
    event_date: date | None = None,
) -> RawSavePaths:
    """One-shot helper: write raw text + parsed dataframe + metadata together."""
    collected_at = metadata.get("collected_at") or collected_now_utc()
    raw_path = write_raw_payload(
        source, raw_text, event_date=event_date, suffix=raw_suffix, collected_at=collected_at
    )
    parsed_path = write_parsed_dataframe(
        source, parsed_df, event_date=event_date, collected_at=collected_at
    )
    metadata = {**metadata, "raw_response_path": str(raw_path), "parsed_path": str(parsed_path)}
    meta_path = write_metadata(source, metadata, event_date=event_date, collected_at=collected_at)
    return RawSavePaths(raw_response=raw_path, parsed_dataframe=parsed_path, metadata=meta_path)
