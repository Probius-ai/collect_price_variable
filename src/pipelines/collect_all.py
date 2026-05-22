"""High-level collector CLI.

Currently only the KPX SMP collector is wired. Add others here as their
sources.yaml entries get verified.
"""

from __future__ import annotations

from datetime import date, datetime, timedelta
from typing import Optional

import typer

from src.collectors.kpx_smp import KpxSmpCollector
from src.utils.logging import get_logger

app = typer.Typer(help="Run data collectors.")

COLLECTORS = {
    "kpx_smp": KpxSmpCollector,
}


def _daterange(start: date, end: date):
    cur = start
    while cur <= end:
        yield cur
        cur += timedelta(days=1)


@app.command()
def main(
    source: str = typer.Option(..., help=f"One of: {sorted(COLLECTORS)}"),
    start: str = typer.Option(..., help="Inclusive YYYY-MM-DD"),
    end: str = typer.Option(..., help="Inclusive YYYY-MM-DD"),
):
    log = get_logger("collect_all")
    if source not in COLLECTORS:
        raise typer.BadParameter(f"Unknown source {source!r}. Known: {sorted(COLLECTORS)}")
    collector_cls = COLLECTORS[source]
    collector = collector_cls()
    start_d = datetime.strptime(start, "%Y-%m-%d").date()
    end_d = datetime.strptime(end, "%Y-%m-%d").date()
    if end_d < start_d:
        raise typer.BadParameter("--end must be on or after --start")

    total_rows = 0
    failures: list[tuple[date, str]] = []
    for day in _daterange(start_d, end_d):
        try:
            run = collector.collect(base_date=day.strftime("%Y%m%d"))
            log.info("OK  %s rows=%d run=%s", day, run.row_count, run.run_id)
            total_rows += run.row_count
        except Exception as exc:  # noqa: BLE001
            log.exception("FAIL %s: %s", day, exc)
            failures.append((day, repr(exc)))

    log.info("Done. rows=%d failures=%d", total_rows, len(failures))
    if failures:
        for day, msg in failures[:10]:
            log.error("  %s -> %s", day, msg)


if __name__ == "__main__":
    app()
