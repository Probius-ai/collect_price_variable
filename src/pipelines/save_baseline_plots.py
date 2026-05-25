"""Snapshot the current monthly SMP baseline plots for later comparison.

We need an immutable visual + tabular record of where the models stand
TODAY, so that when the next iteration (LNG-price forecast integration,
exogenous variables, hourly API data) lands we can plot the same view and
say "look — the one-month lag shrank by X".

Outputs land under ``outputs/figures/<tag>/`` (default tag derives from
today's date + the round/feature snapshot). Each run writes:

  * 6 PNG plots
  * the raw test-split predictions per model (`predictions_test_<m>.csv`)
  * a summary.md that records what feature pipeline + model registry was
    in effect when the snapshot was taken
  * a metrics_snapshot.csv (copy of outputs/metrics/comparison.csv)
  * a walk_forward_snapshot.csv if available
  * a feature_group_ablation_snapshot.csv if available

The script does NOT retrain models — it reads the already-produced
artefacts under outputs/. Run train + walk_forward + ablation first if
you want a fresh snapshot.

Usage:
    python -m src.pipelines.save_baseline_plots
    python -m src.pipelines.save_baseline_plots --tag baseline_pre_lng_v1
"""

from __future__ import annotations

import shutil
from datetime import date
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # headless backend — server-safe
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import typer  # noqa: E402

from src.config.settings import get_settings
from src.models.registry import (
    BASELINE_MODELS,
    STRONG_MONTHLY_BASELINE,
    TRAINABLE_MODELS,
    classify,
)
from src.utils.logging import get_logger

app = typer.Typer(help="Snapshot current baseline plots for later comparison.")

# Top trainable models to spotlight on overlay plots (alphabetical for
# deterministic output).
SPOTLIGHT_MODELS = ["delta_ar_ridge", "delta_ridge", "monthly_ar_ridge", "ridge"]


def _load_test_predictions(model: str, models_dir: Path) -> pd.DataFrame | None:
    path = models_dir / model / "predictions_test.csv"
    if not path.exists():
        return None
    df = pd.read_csv(path)
    if "period_month" in df.columns:
        df["period_month"] = pd.to_datetime(df["period_month"])
    if "target_month" in df.columns:
        df["target_month"] = pd.to_datetime(df["target_month"])
    return df.sort_values(df.columns[0]).reset_index(drop=True)


def _load_walk_forward(model: str, wf_dir: Path) -> pd.DataFrame | None:
    path = wf_dir / model / "predictions.csv"
    if not path.exists():
        return None
    df = pd.read_csv(path)
    if "period_month" in df.columns:
        df["period_month"] = pd.to_datetime(df["period_month"])
    return df.sort_values(df.columns[0]).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Individual plot helpers
# ---------------------------------------------------------------------------


def plot_actual_vs_predicted_overlay(
    preds_by_model: dict[str, pd.DataFrame],
    out_path: Path,
) -> None:
    """Plot actual SMP + each spotlight model's prediction on the test split.

    X-axis is **forecast_origin_month** (= period_month). The persistence
    baseline line will visibly trail the actual line by ~1 month — that is
    the "1-month lag" the user wants to capture for before/after comparison.
    """
    fig, ax = plt.subplots(figsize=(11, 5.5))
    actual_drawn = False
    palette = {
        "persistence_monthly": "#9aa0a6",   # grey — baseline
        "monthly_ar_ridge":    "#1f77b4",
        "delta_ar_ridge":      "#2ca02c",
        "delta_ridge":         "#d62728",
        "ridge":               "#ff7f0e",
        "lightgbm":            "#9467bd",
    }
    for model, df in preds_by_model.items():
        if df is None or df.empty:
            continue
        if not actual_drawn:
            ax.plot(df["period_month"], df["y_true"], "-o",
                    color="black", linewidth=2.0, markersize=4,
                    label="Actual SMP", zorder=10)
            actual_drawn = True
        style = "--" if classify(model) == "baseline" else "-"
        lw = 1.3 if classify(model) == "baseline" else 1.6
        ax.plot(df["period_month"], df["y_pred"], style, marker="o",
                markersize=3, linewidth=lw,
                color=palette.get(model, None),
                label=f"{model} ({classify(model)})")
    ax.set_xlabel("forecast_origin_month")
    ax.set_ylabel("SMP [KRW/kWh]")
    ax.set_title("Test split: actual vs predicted SMP (1-step monthly forecast)")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", fontsize=9, framealpha=0.9)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def plot_persistence_lag_zoom(
    persistence_df: pd.DataFrame, out_path: Path
) -> None:
    """Zoom in on persistence baseline to make the 1-month lag visible.

    The point: `persistence_monthly` predicts SMP(M+1) = SMP(M). On a plot
    indexed by period_month (the forecast origin), the prediction line
    naturally trails the actual line — and we want to capture this look
    so a future LNG-forecast-augmented run can show the lag shrinking.
    """
    if persistence_df is None or persistence_df.empty:
        return
    fig, ax = plt.subplots(figsize=(11, 5.0))
    ax.plot(persistence_df["period_month"], persistence_df["y_true"],
            "-o", color="black", linewidth=2.2, markersize=5, label="Actual SMP")
    ax.plot(persistence_df["period_month"], persistence_df["y_pred"],
            "--s", color="#9aa0a6", linewidth=1.8, markersize=4,
            label="persistence_monthly prediction")
    # Annotate the 1-month-lag interpretation
    ax.fill_between(
        persistence_df["period_month"],
        persistence_df["y_true"], persistence_df["y_pred"],
        alpha=0.12, color="red", label="prediction error (one-month-lag residual)",
    )
    ax.set_xlabel("forecast_origin_month")
    ax.set_ylabel("SMP [KRW/kWh]")
    ax.set_title(
        "Persistence baseline visualised — prediction trails actual by ~1 month\n"
        "(this is the floor any leading-indicator model must beat)"
    )
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", fontsize=10)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def plot_residuals_per_model(
    preds_by_model: dict[str, pd.DataFrame], out_path: Path
) -> None:
    """One panel per spotlight model: residual = pred - actual over time."""
    models = [m for m, df in preds_by_model.items() if df is not None and not df.empty]
    if not models:
        return
    n = len(models)
    cols = 2
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(11, 3 * rows), sharex=True)
    axes = np.atleast_1d(axes).flatten()
    for ax, m in zip(axes, models):
        df = preds_by_model[m]
        resid = df["y_pred"].to_numpy() - df["y_true"].to_numpy()
        ax.axhline(0, color="black", linewidth=0.8)
        ax.plot(df["period_month"], resid, "-o", markersize=3, linewidth=1.4)
        mae = float(np.mean(np.abs(resid)))
        bias = float(np.mean(resid))
        ax.set_title(f"{m}  |  MAE={mae:.2f}  mean_bias={bias:+.2f}",
                     fontsize=10)
        ax.set_ylabel("residual [KRW/kWh]")
        ax.grid(True, alpha=0.3)
    for ax in axes[n:]:
        ax.set_visible(False)
    for ax in axes[(rows - 1) * cols:]:
        ax.set_xlabel("forecast_origin_month")
    fig.suptitle("Residual (predicted − actual) per model — test split",
                 fontsize=12, y=1.005)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def plot_test_metrics_bar(metrics_df: pd.DataFrame, out_path: Path) -> None:
    """Bar chart: MAE and R² per model, baseline vs trainable colour-coded."""
    sub = metrics_df[metrics_df["split"] == "test"].copy()
    sub["kind"] = sub["model"].apply(classify)
    sub = sub.sort_values("mae")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))
    colors = ["#9aa0a6" if k == "baseline" else "#1f77b4" for k in sub["kind"]]
    ax1.barh(sub["model"], sub["mae"], color=colors)
    ax1.invert_yaxis()
    ax1.set_xlabel("MAE [KRW/kWh] (lower is better)")
    ax1.set_title("Test MAE by model")
    for i, v in enumerate(sub["mae"]):
        ax1.text(v + 0.1, i, f"{v:.2f}", va="center", fontsize=9)
    sub_r2 = sub.sort_values("r2", ascending=False)
    colors2 = ["#9aa0a6" if classify(m) == "baseline" else "#2ca02c"
               for m in sub_r2["model"]]
    ax2.barh(sub_r2["model"], sub_r2["r2"], color=colors2)
    ax2.invert_yaxis()
    ax2.axvline(0, color="black", linewidth=0.8)
    ax2.set_xlabel("R² (higher is better; <0 = worse than mean)")
    ax2.set_title("Test R² by model")
    for i, v in enumerate(sub_r2["r2"]):
        ax2.text(v + 0.02, i, f"{v:.2f}", va="center", fontsize=9)
    fig.suptitle("Test-split metrics — grey=baseline, blue/green=trainable",
                 fontsize=12, y=1.02)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def plot_walk_forward_long_range(
    wf_preds_by_model: dict[str, pd.DataFrame], out_path: Path
) -> None:
    """Walk-forward (refit per step) actual vs predicted over the full
    150-month evaluation window. Shows whether the lag persists across
    regimes (pre-shock, LNG shock, post-shock)."""
    models = [m for m, df in wf_preds_by_model.items()
              if df is not None and not df.empty]
    if not models:
        return
    fig, ax = plt.subplots(figsize=(12, 5.5))
    actual_drawn = False
    for m in models:
        df = wf_preds_by_model[m]
        if not actual_drawn:
            ax.plot(df["period_month"], df["y_true"], "-",
                    color="black", linewidth=1.6, label="Actual SMP")
            actual_drawn = True
        style = "--" if classify(m) == "baseline" else "-"
        ax.plot(df["period_month"], df["y_pred"], style, linewidth=1.1,
                alpha=0.8, label=f"{m} ({classify(m)})")
    ax.set_xlabel("period_month")
    ax.set_ylabel("SMP [KRW/kWh]")
    ax.set_title("Walk-forward CV — actual vs predicted across the full eval window")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", fontsize=8, framealpha=0.9)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def plot_feature_group_ablation_heatmap(
    ablation_df: pd.DataFrame, out_path: Path
) -> None:
    """Heatmap: feature groups × models, cells coloured by MAE."""
    if ablation_df.empty:
        return
    sub = ablation_df.dropna(subset=["mae_level"]).copy()
    if sub.empty:
        return
    pv = sub.pivot_table(index="feature_group", columns="model",
                         values="mae_level", aggfunc="mean")
    fig, ax = plt.subplots(figsize=(9, max(4, 0.5 * len(pv))))
    im = ax.imshow(pv.values, aspect="auto", cmap="RdYlGn_r",
                   vmin=float(np.nanmin(pv.values)),
                   vmax=float(np.nanmax(pv.values)))
    ax.set_xticks(range(len(pv.columns)))
    ax.set_xticklabels(pv.columns, rotation=30, ha="right")
    ax.set_yticks(range(len(pv.index)))
    ax.set_yticklabels(pv.index)
    for i in range(len(pv.index)):
        for j in range(len(pv.columns)):
            v = pv.values[i, j]
            if np.isnan(v):
                ax.text(j, i, "—", ha="center", va="center",
                        color="black", fontsize=9)
            else:
                ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                        color="black", fontsize=9)
    ax.set_title("Feature-group × model ablation — MAE on test split (lower=better)")
    fig.colorbar(im, ax=ax, label="MAE [KRW/kWh]")
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Top-level: orchestrate snapshot
# ---------------------------------------------------------------------------


@app.command()
def main(
    tag: str = typer.Option(
        None,
        help="Output subdir name. Default: baseline_pre_lng_forecast_<YYYYMMDD>.",
    ),
    outputs_dir: Path = typer.Option(
        None, help="Defaults to project outputs/."
    ),
    docs_copy: bool = typer.Option(
        False,
        help=(
            "Also copy the 6 PNGs + summary.md into docs/figures/<tag>/ so "
            "any markdown report that references them survives a fresh clone "
            "(outputs/figures/** is gitignored)."
        ),
    ),
):
    log = get_logger("save_baseline_plots")
    settings = get_settings()
    outputs_dir = outputs_dir or settings.outputs_dir
    models_dir = outputs_dir / "models"
    wf_dir = outputs_dir / "walk_forward"
    metrics_csv = outputs_dir / "metrics" / "comparison.csv"
    ablation_csv = outputs_dir / "metrics" / "feature_group_ablation_monthly.csv"
    wf_compare_csv = wf_dir / "comparison.csv"

    tag = tag or f"baseline_pre_lng_forecast_{date.today():%Y%m%d}"
    snap_dir = outputs_dir / "figures" / tag
    snap_dir.mkdir(parents=True, exist_ok=True)
    log.info("Snapshot dir: %s", snap_dir)

    spotlight = [STRONG_MONTHLY_BASELINE, *SPOTLIGHT_MODELS]
    # De-dupe while preserving order
    seen = set()
    spotlight = [m for m in spotlight if not (m in seen or seen.add(m))]
    preds_by_model = {m: _load_test_predictions(m, models_dir) for m in spotlight}
    wf_preds = {m: _load_walk_forward(m, wf_dir) for m in spotlight}

    # ---- Save raw predictions snapshots ----------------------------------
    for m, df in preds_by_model.items():
        if df is not None and not df.empty:
            df.to_csv(snap_dir / f"predictions_test_{m}.csv", index=False)
    for m, df in wf_preds.items():
        if df is not None and not df.empty:
            df.to_csv(snap_dir / f"walk_forward_predictions_{m}.csv", index=False)

    # ---- Copy metric snapshots -------------------------------------------
    if metrics_csv.exists():
        shutil.copy2(metrics_csv, snap_dir / "metrics_comparison_snapshot.csv")
    if wf_compare_csv.exists():
        shutil.copy2(wf_compare_csv, snap_dir / "walk_forward_comparison_snapshot.csv")
    if ablation_csv.exists():
        shutil.copy2(ablation_csv, snap_dir / "feature_group_ablation_snapshot.csv")

    # ---- Generate plots --------------------------------------------------
    plot_actual_vs_predicted_overlay(
        preds_by_model, snap_dir / "plot_01_predictions_test_all_models.png"
    )
    plot_persistence_lag_zoom(
        preds_by_model.get(STRONG_MONTHLY_BASELINE),
        snap_dir / "plot_02_persistence_lag_zoom.png",
    )
    plot_residuals_per_model(
        preds_by_model, snap_dir / "plot_03_residuals_per_model.png"
    )
    if metrics_csv.exists():
        plot_test_metrics_bar(
            pd.read_csv(metrics_csv),
            snap_dir / "plot_04_test_mae_r2_comparison.png",
        )
    plot_walk_forward_long_range(
        wf_preds, snap_dir / "plot_05_walk_forward_long_range.png"
    )
    if ablation_csv.exists():
        plot_feature_group_ablation_heatmap(
            pd.read_csv(ablation_csv),
            snap_dir / "plot_06_feature_group_ablation_heatmap.png",
        )

    # ---- Summary markdown ------------------------------------------------
    summary_rows = []
    if metrics_csv.exists():
        m = pd.read_csv(metrics_csv)
        for _, r in m[m["split"] == "test"].sort_values("mae").iterrows():
            summary_rows.append(
                f"| {r['model']} | {classify(r['model'])} | {r['mae']:.3f} | "
                f"{r.get('mape', float('nan')):.3f} | {r.get('r2', float('nan')):.3f} | "
                f"{r.get('delta_mae', float('nan')):.3f} | "
                f"{r.get('delta_direction_accuracy', float('nan')):.3f} |"
            )
    summary_md = f"""# Baseline snapshot — `{tag}`

Captured {date.today():%Y-%m-%d}.

This snapshot freezes the **current monthly SMP forecast results** before
any LNG-price-forecast integration. The intended use is before/after
comparison: re-run `save_baseline_plots --tag baseline_post_lng_v1` (or
similar) after the LNG model lands, then diff the two folders.

## Forecast contract in effect

* forecast_origin_month = period_month
* information_cutoff    = end-of-day of forecast_origin_month
* target_month          = forecast_origin_month + 1 month
* horizon               = 1M

The persistence baseline (`persistence_monthly`) predicts
`SMP(M+1) = SMP(M)`. On any plot indexed by forecast_origin_month, its
line **trails the actual by one month** — that is the lag this snapshot
documents and which a leading-indicator model is expected to compress.

## Model classification

* Baselines: {', '.join(sorted(BASELINE_MODELS))}
* Trainable: {', '.join(sorted(TRAINABLE_MODELS))}

## Test-split metrics (sorted by MAE)

| model | kind | MAE | MAPE [%] | R² | delta_MAE | delta_dir_acc |
|---|---|---|---|---|---|---|
{chr(10).join(summary_rows) if summary_rows else '| (no metrics on disk) | | | | | | |'}

## Files in this snapshot

| file | what it shows |
|---|---|
| plot_01_predictions_test_all_models.png | Actual + top trainable + persistence overlay (test split) |
| plot_02_persistence_lag_zoom.png        | The one-month lag of persistence visualised explicitly |
| plot_03_residuals_per_model.png         | Per-model residual (pred − actual) over time |
| plot_04_test_mae_r2_comparison.png      | Bar chart of MAE / R² by model |
| plot_05_walk_forward_long_range.png     | Walk-forward CV over the full 150-month eval window |
| plot_06_feature_group_ablation_heatmap.png | Feature-group × model MAE heatmap |
| predictions_test_<m>.csv                | Raw test-split predictions per spotlight model |
| walk_forward_predictions_<m>.csv        | Raw walk-forward predictions per spotlight model |
| metrics_comparison_snapshot.csv         | Copy of `outputs/metrics/comparison.csv` |
| walk_forward_comparison_snapshot.csv    | Copy of `outputs/walk_forward/comparison.csv` |
| feature_group_ablation_snapshot.csv     | Copy of the ablation result table |

## Reproduce a comparable snapshot

```bash
# 1. rebuild monthly features (or run the file pipeline)
python -m src.pipelines.build_monthly_features --area mainland --horizon-months 1

# 2. retrain all models you want spotlighted
F=data/processed/smp_monthly_mainland_h1m.parquet
for M in persistence_monthly monthly_ar_ridge delta_ar_ridge delta_ridge ridge; do
  python -m src.pipelines.train --features-path $F --model $M \\
      --target target_smp_t_plus_h_months --timestamp-col period_month \\
      --min-train-rows 24
done

# 3. (optional) walk-forward + ablation for plots 5/6
for M in persistence_monthly monthly_ar_ridge delta_ar_ridge delta_ridge ridge; do
  python -m src.pipelines.walk_forward main --features-path $F --model $M --min-train-rows 24
done
python -m src.pipelines.walk_forward compare
python -m src.pipelines.feature_group_ablation --features-path $F

# 4. snapshot under a new tag
python -m src.pipelines.evaluate
python -m src.pipelines.save_baseline_plots --tag baseline_post_lng_v1
```
"""
    (snap_dir / "summary.md").write_text(summary_md, encoding="utf-8")
    log.info("Wrote %d plots + summary.md under %s",
             len(list(snap_dir.glob("plot_*.png"))), snap_dir)
    print(f"Snapshot ready: {snap_dir}")
    for p in sorted(snap_dir.iterdir()):
        print(f"  {p.name}  ({p.stat().st_size:,} bytes)")

    if docs_copy:
        # Mirror PNGs + summary.md into docs/figures/<tag>/ so any committed
        # markdown report referencing these images stays self-contained on
        # a fresh clone (outputs/figures/** is gitignored).
        docs_target = settings.project_root / "docs" / "figures" / tag
        docs_target.mkdir(parents=True, exist_ok=True)
        copied = 0
        for src_path in sorted(snap_dir.iterdir()):
            if src_path.suffix == ".png" or src_path.name == "summary.md":
                shutil.copy2(src_path, docs_target / src_path.name)
                copied += 1
        log.info("Mirrored %d files into %s (tracked, for committed reports)",
                 copied, docs_target)
        print(f"docs/ mirror: {docs_target}")


if __name__ == "__main__":
    app()
