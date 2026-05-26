"""Project-wide settings loaded from environment variables and YAML.

The two main entry points are:

* `get_settings()` — typed env settings (Pydantic).
* `load_sources_config()` — the source assumption registry from sources.yaml.

Both are cached so importing them is cheap.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = PROJECT_ROOT / "src" / "config"
DATA_DIR = PROJECT_ROOT / "data"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=PROJECT_ROOT / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    kpx_public_api_key: str | None = Field(default=None)
    kpx_public_api_key_encoded: str | None = Field(default=None)
    kma_public_api_key: str | None = Field(default=None)
    kma_public_api_key_encoded: str | None = Field(default=None)
    ecos_api_key: str | None = Field(default=None)
    eia_api_key: str | None = Field(default=None)

    storage_backend: Literal["duckdb", "postgres"] = Field(default="duckdb")
    duckdb_path: Path = Field(default=Path("data/kpx_forecast.duckdb"))
    postgres_dsn: str | None = Field(default=None)

    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = Field(default="INFO")
    project_timezone: str = Field(default="Asia/Seoul")

    # Optional notification webhook for the MLOps smoke test. None / empty
    # → the integration silently no-ops. Treat as a secret (anyone with
    # the URL can post to the Discord channel).
    mlops_discord_webhook_url: str | None = Field(default=None)

    @property
    def project_root(self) -> Path:
        return PROJECT_ROOT

    @property
    def data_dir(self) -> Path:
        return DATA_DIR

    @property
    def outputs_dir(self) -> Path:
        return OUTPUTS_DIR

    @property
    def duckdb_full_path(self) -> Path:
        path = self.duckdb_path
        if not path.is_absolute():
            path = PROJECT_ROOT / path
        return path


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()


@lru_cache(maxsize=1)
def load_sources_config() -> dict[str, Any]:
    """Load src/config/sources.yaml as a plain dict."""
    with (CONFIG_DIR / "sources.yaml").open("r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def get_source_config(source_name: str) -> dict[str, Any]:
    """Return the per-source config block, raising if the source is unknown.

    Searches `sources:` (API collectors) first, then `file_sources:`
    (manual-download file loaders) so callers can use one helper for both.
    """
    config = load_sources_config()
    api_sources = config.get("sources", {}) or {}
    file_sources = config.get("file_sources", {}) or {}
    if source_name in api_sources:
        return api_sources[source_name]
    if source_name in file_sources:
        return file_sources[source_name]
    raise KeyError(
        f"Unknown source '{source_name}'. Known API sources: {sorted(api_sources)}; "
        f"known file sources: {sorted(file_sources)}"
    )


def get_source_kind(source_name: str) -> str:
    """Return 'api' or 'file' for a source."""
    config = load_sources_config()
    if source_name in (config.get("sources") or {}):
        return "api"
    if source_name in (config.get("file_sources") or {}):
        return "file"
    raise KeyError(f"Unknown source '{source_name}'.")


def list_file_sources() -> list[str]:
    return sorted((load_sources_config().get("file_sources") or {}).keys())


def get_source_defaults() -> dict[str, Any]:
    return load_sources_config().get("defaults", {})
