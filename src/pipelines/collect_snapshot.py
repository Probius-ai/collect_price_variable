"""Snapshot collectors (single fetch per run).

Used for sources that don't follow the day-by-day pattern of
`collect_all.py` (EIA STEO publishes monthly, KMA mid-term forecast is
twice-daily — but each call returns one self-contained snapshot).

Usage:
  python -m src.pipelines.collect_snapshot --source eia_steo
  python -m src.pipelines.collect_snapshot --source kma_mid_temperature
"""

from __future__ import annotations

import typer

from src.collectors.eia_steo import EiaSteoCollector
from src.collectors.kma_mid_temperature import KmaMidTemperatureCollector
from src.collectors.kma_village_fcst import KmaVilageFcstCollector
from src.utils.logging import get_logger

app = typer.Typer(help="Run snapshot collectors (EIA, KMA).")

COLLECTORS = {
    "eia_steo": EiaSteoCollector,
    "kma_mid_temperature": KmaMidTemperatureCollector,
    "kma_village_fcst": KmaVilageFcstCollector,
}


@app.command()
def main(
    source: str = typer.Option(..., help=f"One of: {sorted(COLLECTORS)}"),
):
    log = get_logger("collect_snapshot")
    if source not in COLLECTORS:
        raise typer.BadParameter(f"Unknown source {source!r}. Known: {sorted(COLLECTORS)}")
    collector = COLLECTORS[source]()
    run = collector.collect()
    log.info("OK %s rows=%d run=%s", source, run.row_count, run.run_id)
    for p in run.raw_paths:
        log.info("  parsed=%s", p.parsed_dataframe)


if __name__ == "__main__":
    app()
