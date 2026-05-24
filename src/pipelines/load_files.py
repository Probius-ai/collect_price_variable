"""Run a file loader against the user's drop directory.

Loaders read from ``data/raw/manual_or_filedata/<source>/`` and write:

  - a date-partitioned copy of the raw file
  - parsed_<stamp>.parquet  (canonical column names from sources.yaml)
  - metadata_<stamp>.json   (source_url, frequency, unit, limitations, etc.)

Example:
    python -m src.pipelines.load_files --source kpx_smp_monthly_mainland
"""

from __future__ import annotations

from pathlib import Path

import typer

from src.collectors.kpx_files import LOADERS
from src.utils.io import manual_drop_dir
from src.utils.logging import get_logger

app = typer.Typer(help="Load downloaded files for a file-based source.")


@app.command()
def main(
    source: str = typer.Option(..., help=f"One of: {sorted(LOADERS)}"),
    path: Path | None = typer.Option(
        None,
        help="Optional single file path; default loads every file in the drop directory.",
    ),
):
    log = get_logger("load_files")
    if source not in LOADERS:
        raise typer.BadParameter(f"Unknown source {source!r}. Known: {sorted(LOADERS)}")
    loader = LOADERS[source]()
    log.info("Drop dir: %s", manual_drop_dir(source))

    if path is not None:
        if not path.exists():
            raise typer.BadParameter(f"file not found: {path}")
        run = loader.load_one(path)
        log.info("OK rows=%d run=%s", run.row_count, run.run_id)
        return

    runs = loader.load_all()
    total = sum(r.row_count for r in runs)
    log.info("Loaded %d files, %d total rows.", len(runs), total)


if __name__ == "__main__":
    app()
