"""Build the monthly SMP feature table from long-format file-loader outputs.

Reads parsed_*.parquet under data/raw/kpx/smp_monthly_{kepco_file,home_avg_file}/
(and optionally data/raw/kpx/settlement_monthly_file/) and writes
data/processed/smp_monthly_<area>_h<horizon>m.{parquet,csv}.

Also writes a side-info JSON next to the parquet capturing source_priority
dedup decisions and settlement fuel coverage for the DQ report.
"""

from __future__ import annotations

import json
from pathlib import Path

import typer

from src.config.settings import get_settings
from src.features.build_monthly import build_smp_monthly_features
from src.utils.logging import get_logger

app = typer.Typer(help="Build monthly SMP feature tables.")


@app.command()
def main(
    area: str = typer.Option("mainland", help="mainland | jeju | integrated"),
    horizon_months: int = typer.Option(1, help="Forecast horizon in months"),
    include_settlement: bool = typer.Option(True, help="Join settlement monthly lag features"),
    output_path: Path | None = typer.Option(None, help="Defaults to data/processed/..."),
    write_csv: bool = typer.Option(True, help="Also write a CSV next to the Parquet"),
):
    log = get_logger("build_monthly_features")
    if area not in {"mainland", "jeju", "integrated"}:
        raise typer.BadParameter("area must be mainland | jeju | integrated")

    feats, side_info = build_smp_monthly_features(
        area=area,
        horizon_months=horizon_months,
        include_settlement=include_settlement,
    )
    settings = get_settings()
    output_path = output_path or (
        settings.data_dir
        / "processed"
        / f"smp_monthly_{area}_h{horizon_months}m.parquet"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    feats.to_parquet(output_path, index=False)
    log.info("Wrote %d rows × %d cols -> %s", len(feats), len(feats.columns), output_path)

    if write_csv:
        csv_path = output_path.with_suffix(".csv")
        feats.to_csv(csv_path, index=False, encoding="utf-8-sig")
        log.info("Wrote CSV -> %s", csv_path)

    # Side-info: dedup decisions + fuel coverage for the DQ report.
    si_path = output_path.with_suffix(".sideinfo.json")
    dedup_log = side_info.pop("smp_priority_dedup_log", None)
    payload = dict(side_info)
    if dedup_log is not None:
        payload["smp_priority_dedup_log"] = dedup_log.assign(
            period_month=dedup_log["period_month"].astype(str)
        ).to_dict(orient="records")
    si_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, default=str),
                       encoding="utf-8")
    log.info("Wrote side-info -> %s", si_path)


if __name__ == "__main__":
    app()
