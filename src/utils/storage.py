"""DuckDB storage helpers (Postgres is wired but not yet implemented)."""

from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path

import duckdb
import pandas as pd

from src.config.settings import get_settings
from src.utils.logging import get_logger

logger = get_logger(__name__)


@contextmanager
def duckdb_connection():
    settings = get_settings()
    if settings.storage_backend != "duckdb":
        raise NotImplementedError(
            f"storage_backend={settings.storage_backend} not yet implemented."
        )
    path: Path = settings.duckdb_full_path
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = duckdb.connect(str(path))
    try:
        yield conn
    finally:
        conn.close()


SCHEMA_SQL = {
    "raw_kpx_smp_hourly": """
        CREATE TABLE IF NOT EXISTS raw_kpx_smp_hourly (
            area TEXT NOT NULL,
            trade_date DATE NOT NULL,
            trade_hour INTEGER NOT NULL,
            interval_start TIMESTAMP,
            interval_end TIMESTAMP,
            smp_krw_per_kwh DOUBLE,
            demand_forecast DOUBLE,
            source_name TEXT,
            collected_at TIMESTAMP NOT NULL,
            schema_version INTEGER,
            raw_payload_path TEXT,
            PRIMARY KEY (area, trade_date, trade_hour, collected_at)
        );
    """,
    "collection_runs": """
        CREATE TABLE IF NOT EXISTS collection_runs (
            run_id TEXT PRIMARY KEY,
            source TEXT NOT NULL,
            started_at TIMESTAMP NOT NULL,
            finished_at TIMESTAMP,
            status TEXT NOT NULL,
            row_count INTEGER,
            params_json TEXT,
            error TEXT
        );
    """,
}


def ensure_schema(table_names: list[str] | None = None) -> None:
    targets = table_names or list(SCHEMA_SQL.keys())
    with duckdb_connection() as conn:
        for name in targets:
            if name not in SCHEMA_SQL:
                raise KeyError(f"Unknown table {name}; known: {sorted(SCHEMA_SQL)}")
            conn.execute(SCHEMA_SQL[name])
    logger.info("Ensured DuckDB schema: %s", targets)


def upsert_dataframe(table: str, df: pd.DataFrame, key_columns: list[str]) -> int:
    """Insert dataframe rows, replacing on key_columns conflict.

    DuckDB doesn't support ON CONFLICT for arbitrary tables yet, so we delete
    matching keys and then insert.
    """
    if df.empty:
        return 0
    with duckdb_connection() as conn:
        conn.register("incoming", df)
        key_pred = " AND ".join(
            f"{table}.{col} = incoming.{col}" for col in key_columns
        )
        conn.execute(
            f"DELETE FROM {table} USING incoming WHERE {key_pred}"
        )
        conn.execute(f"INSERT INTO {table} SELECT * FROM incoming")
        conn.unregister("incoming")
    return len(df)
