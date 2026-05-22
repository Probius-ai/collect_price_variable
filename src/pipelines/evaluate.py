"""Aggregate metrics across already-trained models."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import typer

from src.config.settings import get_settings
from src.utils.logging import get_logger

app = typer.Typer(help="Compare metrics across saved models.")


@app.command()
def main(
    models_dir: Path | None = typer.Option(None, help="Defaults to outputs/models"),
    output_csv: Path | None = typer.Option(None, help="Where to write the comparison CSV"),
):
    log = get_logger("evaluate")
    settings = get_settings()
    models_dir = models_dir or settings.outputs_dir / "models"
    rows: list[dict] = []
    for metrics_path in sorted(models_dir.glob("*/metrics.json")):
        model_name = metrics_path.parent.name
        metrics = json.loads(metrics_path.read_text())
        for split, vals in metrics.items():
            if not vals:
                continue
            rows.append({"model": model_name, "split": split, **vals})
    if not rows:
        log.warning("No metrics.json found under %s", models_dir)
        return
    df = pd.DataFrame(rows)
    df = df.sort_values(["split", "mae"])
    output_csv = output_csv or settings.outputs_dir / "metrics" / "comparison.csv"
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_csv, index=False)
    log.info("Wrote comparison -> %s", output_csv)
    print(df.to_string(index=False))


if __name__ == "__main__":
    app()
