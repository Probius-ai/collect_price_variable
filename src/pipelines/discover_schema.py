"""Schema-discovery CLI.

Use this before populating column_mapping in sources.yaml. It does ONE API
call, persists the raw response under data/raw/<source>/..., and prints a
structural summary of the JSON keys so you can update sources.yaml.

Example:
    python -m src.pipelines.discover_schema --source kpx_smp_day_ahead \
        --param base_date=20240301
"""

from __future__ import annotations

import json
from datetime import date
from typing import Any

import typer

from src.collectors.kpx_smp import KpxSmpCollector
from src.utils.io import write_raw_payload, write_metadata
from src.utils.logging import get_logger
from src.utils.time import collected_now_utc

app = typer.Typer(help="Discover the actual fields of a remote source.")

COLLECTORS: dict[str, type] = {
    "kpx_smp_day_ahead": KpxSmpCollector,
}


def _parse_extra(params: list[str]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for raw in params:
        if "=" not in raw:
            raise typer.BadParameter(f"--param expects key=value, got {raw!r}")
        k, _, v = raw.partition("=")
        result[k] = v
    return result


@app.command()
def main(
    source: str = typer.Option(..., help="Source key in sources.yaml"),
    param: list[str] = typer.Option(
        [], "--param", help="Extra request params, e.g. --param base_date=20240301"
    ),
):
    log = get_logger("discover_schema")
    if source not in COLLECTORS:
        raise typer.BadParameter(
            f"No collector registered for {source!r}. Known: {sorted(COLLECTORS)}"
        )
    collector = COLLECTORS[source]()
    extra = _parse_extra(param)
    log.info("Calling %s with %s", source, extra)

    fetch = collector.fetch_with_retry(**extra)
    collected_at = collected_now_utc()

    raw_path = write_raw_payload(
        source,
        fetch.raw_text,
        event_date=date.today(),
        suffix=fetch.raw_suffix,
        collected_at=collected_at,
    )
    summary = collector.discover_schema.__wrapped__(collector, **extra) if False else {  # noqa: E501
        "request": fetch.request,
        "raw_excerpt": fetch.raw_text[:2000],
        "discovered_keys": _summarize(fetch.parsed),
    }
    summary["raw_response_path"] = str(raw_path)
    summary["collected_at"] = collected_at
    write_metadata(source, summary, event_date=date.today(), collected_at=collected_at)

    print(json.dumps(summary["discovered_keys"], indent=2, ensure_ascii=False))
    print(f"\nRaw response saved to: {raw_path}")
    print(
        "Next: open the raw response, decide which vendor keys map to the "
        "internal column names, then update sources.yaml::sources."
        f"{source}.column_mapping and bump schema_version."
    )


def _summarize(obj: Any, depth: int = 4) -> Any:
    if depth < 0:
        return type(obj).__name__
    if isinstance(obj, dict):
        return {k: _summarize(v, depth - 1) for k, v in obj.items()}
    if isinstance(obj, list):
        if not obj:
            return []
        return [_summarize(obj[0], depth - 1), f"(+{len(obj) - 1} more)"]
    return type(obj).__name__


if __name__ == "__main__":
    app()
