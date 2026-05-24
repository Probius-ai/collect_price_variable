"""Inspect a downloaded file and print its raw columns + sample rows.

Use this BEFORE filling in `sources.yaml::file_sources.<source>.column_mapping`.
It only reads — it does not persist a parsed dataframe — so it is safe to run
on a file with unknown structure.

Example:
    python -m src.pipelines.discover_file \
        --source kpx_smp_monthly_mainland \
        --path data/raw/manual_or_filedata/kpx_smp_monthly_mainland/smp_2024.csv
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
import typer

from src.config.settings import get_source_config, list_file_sources
from src.utils.logging import get_logger

app = typer.Typer(help="Discover raw columns/headers of a downloaded file.")


def _read_any(path: Path, config: dict[str, Any]) -> pd.DataFrame:
    fmt = (config.get("file_format") or "").lower()
    if fmt in {"", "tbd_after_first_download"}:
        fmt = path.suffix.lower().lstrip(".")
    encoding = config.get("encoding") or "utf-8"
    if encoding == "TBD_AFTER_FIRST_DOWNLOAD":
        encoding = "utf-8"
    skip = int(config.get("drop_rows_before", 0) or 0)
    header_row = int(config.get("header_row", 0) or 0)
    if fmt in {"csv", "tsv"}:
        return pd.read_csv(
            path,
            sep="\t" if fmt == "tsv" else ",",
            encoding=encoding,
            skiprows=skip,
            header=header_row,
        )
    if fmt in {"xlsx", "xls"}:
        sheet = config.get("sheet_name") or 0
        return pd.read_excel(path, sheet_name=sheet, skiprows=skip, header=header_row)
    raise typer.BadParameter(f"Unsupported file_format={fmt!r}")


@app.command()
def main(
    source: str = typer.Option(..., help=f"Known file sources: {list_file_sources()}"),
    path: Path = typer.Option(..., help="Path to the downloaded file"),
    rows: int = typer.Option(5, help="Number of sample rows to print"),
):
    log = get_logger("discover_file")
    if source not in list_file_sources():
        raise typer.BadParameter(
            f"{source!r} is not in sources.yaml::file_sources. "
            f"Known: {list_file_sources()}"
        )
    if not path.exists():
        raise typer.BadParameter(f"file does not exist: {path}")
    config = get_source_config(source)
    df = _read_any(path, config)
    log.info("File rows=%d cols=%d", len(df), len(df.columns))
    print("Detected columns:")
    for i, c in enumerate(df.columns):
        print(f"  [{i:2d}] {c!r}")
    print("\nFirst rows:")
    print(df.head(rows).to_string(index=False))
    print(
        "\nNext: open src/config/sources.yaml and fill "
        f"file_sources.{source}.column_mapping using the column headers above. "
        "Then re-run `python -m src.pipelines.load_files --source <name>`."
    )


if __name__ == "__main__":
    app()
