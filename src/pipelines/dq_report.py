"""Data Quality report — round 2 pre-approval MVP.

Aggregates per-source metadata sidecars + quarantine manifests + the
feature-builder's side_info into a single human-readable JSON report under
``outputs/data_quality/``.

Sections:
  * `sources`: per-source row counts, coverage range, dropped columns,
    missing canonical categories, latest collected_at.
  * `monthly_smp_priority_dedup`: rows where the SMP loaders' two sources
    disagreed on the same (period_month, area).
  * `quarantine`: misnamed files moved aside this round.
  * `notes`: human-readable summary lines.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd
import typer

from src.collectors.kpx_files import LOADERS
from src.collectors.quarantine import quarantine_dir
from src.config.settings import get_settings
from src.utils.io import source_root_dir
from src.utils.logging import get_logger

app = typer.Typer(help="Build a data-quality report from loader outputs.")

logger = get_logger("dq_report")


def _collect_metadata(source_id: str) -> dict[str, Any]:
    root = source_root_dir(source_id)
    runs: list[dict] = []
    for meta in sorted(root.rglob("metadata_*.json")):
        try:
            runs.append(json.loads(meta.read_text(encoding="utf-8")))
        except Exception as exc:  # noqa: BLE001
            logger.warning("Could not read %s: %s", meta, exc)
    if not runs:
        return {"runs": 0}
    latest = max(runs, key=lambda r: r.get("collected_at") or "")
    return {
        "runs": len(runs),
        "latest_collected_at": latest.get("collected_at"),
        "latest_source_file": latest.get("file_path"),
        "source_url": latest.get("source_url"),
        "frequency": latest.get("frequency"),
        "unit": latest.get("unit"),
        "schema_version": latest.get("schema_version"),
        "dropped_columns": latest.get("dropped_columns", []),
        "missing_canonical_categories": latest.get("missing_canonical_categories", []),
        "zero_coerced_rows": latest.get("zero_coerced_rows", {}),
        "limitations": latest.get("limitations", []),
    }


def _coverage_for_source(source_id: str) -> dict[str, Any]:
    root = source_root_dir(source_id)
    files = sorted(root.rglob("parsed_*.parquet"))
    if not files:
        return {"rows": 0}
    df = pd.concat([pd.read_parquet(p) for p in files], ignore_index=True)
    summary: dict[str, Any] = {"rows": int(len(df))}
    if "period_month" in df.columns:
        ts = pd.to_datetime(df["period_month"])
        summary["period_min"] = str(ts.min().date()) if not ts.empty else None
        summary["period_max"] = str(ts.max().date()) if not ts.empty else None
        if "area" in df.columns:
            summary["rows_by_area"] = df.groupby("area").size().to_dict()
        if "fuel_type" in df.columns:
            summary["rows_by_fuel_type"] = df.groupby("fuel_type").size().to_dict()
    elif "trade_date" in df.columns:
        ts = pd.to_datetime(df["trade_date"])
        summary["trade_date_min"] = str(ts.min().date()) if not ts.empty else None
        summary["trade_date_max"] = str(ts.max().date()) if not ts.empty else None
    return summary


def _collect_quarantine() -> list[dict[str, Any]]:
    qroot = quarantine_dir()
    if not qroot.exists():
        return []
    all_entries: list[dict[str, Any]] = []
    for manifest in sorted(qroot.rglob("manifest_*.json")):
        try:
            entries = json.loads(manifest.read_text(encoding="utf-8"))
            for e in entries:
                e["_manifest"] = str(manifest)
                all_entries.append(e)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Could not read %s: %s", manifest, exc)
    return all_entries


def build_report(extra: dict[str, Any] | None = None) -> dict[str, Any]:
    sources_report: dict[str, dict[str, Any]] = {}
    for sid in sorted(LOADERS.keys()):
        meta = _collect_metadata(sid)
        coverage = _coverage_for_source(sid)
        sources_report[sid] = {**meta, **coverage}

    quarantine = _collect_quarantine()
    report: dict[str, Any] = {
        "generated_at": pd.Timestamp.now("UTC").isoformat(),
        "sources": sources_report,
        "quarantine": quarantine,
    }
    if extra:
        report.update(extra)

    notes: list[str] = []
    for sid, info in sources_report.items():
        if info.get("rows", 0) == 0:
            notes.append(f"{sid}: no parsed rows on disk yet.")
        elif info.get("dropped_columns"):
            notes.append(
                f"{sid}: dropped vendor columns (renewable subcategories etc): "
                + ", ".join(info["dropped_columns"])
            )
        if info.get("missing_canonical_categories"):
            notes.append(
                f"{sid}: canonical categories absent from the latest file: "
                + ", ".join(info["missing_canonical_categories"])
            )
        if info.get("zero_coerced_rows"):
            counts = info["zero_coerced_rows"]
            total = sum(counts.values()) if isinstance(counts, dict) else 0
            if total:
                notes.append(
                    f"{sid}: dropped {total} rows where vendor wrote 0 as a "
                    f"'no published value' marker — per-area counts: {counts}"
                )
    if quarantine:
        for q in quarantine:
            notes.append(
                f"quarantine: {q.get('original_path')} → {q.get('quarantined_path')} "
                f"(reason={q.get('reason')})"
            )
    report["notes"] = notes
    return report


@app.command()
def main(
    output_path: Path | None = typer.Option(
        None, help="Defaults to outputs/data_quality/report_<stamp>.json"
    ),
):
    report = build_report()
    settings = get_settings()
    out_dir = settings.outputs_dir / "data_quality"
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = pd.Timestamp.now("UTC").strftime("%Y%m%dT%H%M%SZ")
    out_path = output_path or out_dir / f"report_{stamp}.json"
    out_path.write_text(json.dumps(report, indent=2, ensure_ascii=False, default=str),
                        encoding="utf-8")
    print(f"Wrote DQ report -> {out_path}")
    if report["notes"]:
        print("\nNotes:")
        for n in report["notes"]:
            print(f"  - {n}")


if __name__ == "__main__":
    app()
