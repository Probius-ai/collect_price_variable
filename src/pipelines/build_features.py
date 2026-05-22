"""Read collected SMP Parquet files and emit a feature dataframe."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import typer

from src.collectors.kpx_smp import KpxSmpCollector
from src.config.settings import get_settings
from src.features.build import build_smp_hourly_features
from src.utils.io import source_root_dir
from src.utils.logging import get_logger
from src.validation.leakage_checks import (
    assert_hourly_contiguous,
    assert_no_duplicate_keys,
    daily_missing_hour_report,
)

app = typer.Typer(help="Build SMP feature tables.")


def _load_collected_smp(source_name: str = KpxSmpCollector.source_name) -> pd.DataFrame:
    """Read every parsed_*.parquet written by the SMP collector.

    The path is resolved via `source_root_dir(source_name)` so collector and
    feature builder always agree on the on-disk layout.
    """
    root = source_root_dir(source_name)
    files = sorted(root.rglob("parsed_*.parquet"))
    if not files:
        raise FileNotFoundError(
            f"No parsed SMP Parquet files under {root}. "
            f"Run `python -m src.pipelines.collect_all --source kpx_smp ...` first."
        )
    frames = [pd.read_parquet(p) for p in files]
    df = pd.concat(frames, ignore_index=True)
    df["interval_end"] = pd.to_datetime(df["interval_end"])
    # Drop exact duplicates across collection runs (same area/trade_hour/trade_date)
    df = df.sort_values(["area", "interval_end"]).drop_duplicates(
        subset=["area", "interval_end"], keep="last"
    )
    return df.reset_index(drop=True)


@app.command()
def main(
    target: str = typer.Option("smp_hourly", help="Currently only smp_hourly is supported"),
    area: str = typer.Option("mainland", help="Filter to a single area before featurising"),
    horizon: int = typer.Option(24, help="Forecast horizon in hours"),
    output_path: Path | None = typer.Option(None, help="Defaults to data/processed/<target>_<area>.parquet"),
    skip_contiguity_check: bool = typer.Option(False, help="Allow gaps in hourly history (NOT recommended)"),
):
    log = get_logger("build_features")
    if target != "smp_hourly":
        raise typer.BadParameter("Only smp_hourly is implemented in this MVP")
    smp_df = _load_collected_smp()
    log.info("Loaded %d SMP rows across %d areas", len(smp_df), smp_df["area"].nunique())

    # The vendor area label may be Korean ('육지'/'제주'). Allow callers to
    # pass either the canonical English code or the original token.
    aliases = {"mainland": {"mainland", "육지"}, "jeju": {"jeju", "제주"}}
    accepted = aliases.get(area, {area})
    filtered = smp_df[smp_df["area"].isin(accepted)]
    if filtered.empty:
        raise typer.BadParameter(
            f"No SMP rows for area={area!r}. Found areas: {sorted(smp_df['area'].unique())}"
        )

    assert_no_duplicate_keys(filtered, ["area", "interval_end"], label="SMP hourly rows")
    if not skip_contiguity_check:
        try:
            assert_hourly_contiguous(filtered)
        except AssertionError as exc:
            report = daily_missing_hour_report(filtered)
            log.warning("Hourly gap detected: %s", exc)
            log.warning("Missing-hour report:\n%s", report[report["missing_hours"] > 0].to_string())
            raise

    features = build_smp_hourly_features(filtered, forecast_horizon_hours=horizon)
    log.info("Feature rows=%d cols=%d", len(features), len(features.columns))

    output_path = (
        output_path
        or get_settings().data_dir / "processed" / f"{target}_{area}_h{horizon}.parquet"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    features.to_parquet(output_path, index=False)
    log.info("Wrote features -> %s", output_path)


if __name__ == "__main__":
    app()
