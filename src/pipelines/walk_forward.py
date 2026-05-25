"""Walk-forward cross-validation for monthly SMP models.

Instead of a single chronological train/valid/test split, walk-forward re-fits
the model at each step of an expanding window:

    for t in eval_period:
        train_set = [rows where period_month < t]
        if len(train_set) < min_train_rows: skip
        refit model on train_set
        predict the single row at t
    aggregate (y_true, y_pred) over the eval_period
    compute MAE / RMSE / MAPE / R^2 / directional_accuracy

This is the right evaluation for forecasting in a non-stationary regime
(post-LNG shock SMP) because it mirrors how the model would be used in
production: every month, refit on everything-so-far and predict next month.

Output:
    outputs/walk_forward/<model>/predictions.csv  (period_month, y_true, y_pred, area)
    outputs/walk_forward/<model>/metrics.json
    outputs/walk_forward/comparison.csv           (all models side-by-side)
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import typer

from src.config.settings import get_settings
from src.models.ar_monthly import MonthlyARRidge
from src.models.delta_models import DeltaARRidge, DeltaLightGBM, DeltaRidge
from src.models.lightgbm_model import LightGBMModel
from src.models.metrics import compute_metrics
from src.models.naive import NaiveLag1m, PersistenceMonthly, SeasonalNaiveLag12m
from src.models.ridge_model import RidgeModel
from src.utils.logging import get_logger

app = typer.Typer(help="Walk-forward cross-validation for monthly models.")

MODELS = {
    "naive_lag_1m": NaiveLag1m,
    "persistence_monthly": PersistenceMonthly,
    "seasonal_naive_lag_12m": SeasonalNaiveLag12m,
    "ridge": RidgeModel,
    "lightgbm": LightGBMModel,
    "monthly_ar_ridge": MonthlyARRidge,
    "delta_ridge": DeltaRidge,
    "delta_lightgbm": DeltaLightGBM,
    "delta_ar_ridge": DeltaARRidge,
}


def _walk_forward(
    df: pd.DataFrame,
    *,
    model_name: str,
    target: str,
    timestamp_col: str,
    feature_cols: list[str],
    eval_start_idx: int,
) -> pd.DataFrame:
    """Return a 1-row-per-step prediction frame."""
    rows = []
    for idx in range(eval_start_idx, len(df)):
        train_slice = df.iloc[:idx]
        pred_row = df.iloc[idx : idx + 1]
        model_cls = MODELS[model_name]
        model = model_cls()
        model.fit(train_slice[feature_cols], train_slice[target])
        y_pred = float(model.predict(pred_row[feature_cols]).iloc[0])
        row_out = {
            timestamp_col: pred_row[timestamp_col].iloc[0],
            "area": pred_row.get("area", pd.Series(["mainland"])).iloc[0],
            "y_true": float(pred_row[target].iloc[0]),
            "y_pred": y_pred,
        }
        # Forecast-origin metadata carry-through (round 5).
        for meta_col in ("forecast_origin_month", "target_month",
                         "information_cutoff", "horizon"):
            if meta_col in pred_row.columns:
                row_out[meta_col] = pred_row[meta_col].iloc[0]
        if "smp_t_observed" in pred_row.columns:
            obs = float(pred_row["smp_t_observed"].iloc[0])
            row_out["smp_t_observed"] = obs
            row_out["true_delta_1m"] = row_out["y_true"] - obs
            row_out["predicted_delta_1m"] = y_pred - obs
        rows.append(row_out)
    return pd.DataFrame(rows)


@app.command()
def main(
    features_path: Path = typer.Option(...),
    model: str = typer.Option(..., help=f"One of: {sorted(MODELS)}"),
    target: str = typer.Option("target_smp_t_plus_h_months"),
    timestamp_col: str = typer.Option("period_month"),
    min_train_rows: int = typer.Option(24, help="Minimum train window before evaluation starts"),
    eval_start_period: str = typer.Option(
        None,
        help="Optional ISO date to start evaluation (default: row after min_train_rows).",
    ),
):
    log = get_logger("walk_forward")
    if model not in MODELS:
        raise typer.BadParameter(f"Unknown model {model!r}. Known: {sorted(MODELS)}")
    settings = get_settings()
    out_dir = settings.outputs_dir / "walk_forward" / model
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_parquet(features_path).sort_values(timestamp_col).reset_index(drop=True)
    if target not in df.columns:
        raise typer.BadParameter(f"target {target!r} missing")
    drop_cols = {target, timestamp_col, "area"}
    feature_cols = [c for c in df.columns if c not in drop_cols]

    if eval_start_period:
        start_ts = pd.Timestamp(eval_start_period)
        match = df.index[df[timestamp_col] >= start_ts]
        if not len(match):
            raise typer.BadParameter(
                f"eval_start_period {eval_start_period} is past last row "
                f"{df[timestamp_col].max()}"
            )
        eval_start_idx = max(int(match[0]), min_train_rows)
    else:
        eval_start_idx = min_train_rows

    if eval_start_idx >= len(df):
        raise typer.BadParameter(
            f"eval_start_idx={eval_start_idx} but only {len(df)} rows"
        )

    log.info(
        "Walk-forward %s: train_min=%d, eval_rows=%d, target=%s",
        model, eval_start_idx, len(df) - eval_start_idx, target,
    )

    preds = _walk_forward(
        df,
        model_name=model,
        target=target,
        timestamp_col=timestamp_col,
        feature_cols=feature_cols,
        eval_start_idx=eval_start_idx,
    )
    preds.to_csv(out_dir / "predictions.csv", index=False)

    metrics = compute_metrics(
        pd.Series(preds["y_true"].values),
        pd.Series(preds["y_pred"].values),
    ).to_dict()
    metrics["n_observations"] = len(preds)
    metrics["eval_start_period"] = str(preds[timestamp_col].iloc[0])
    metrics["eval_end_period"] = str(preds[timestamp_col].iloc[-1])
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    log.info("Walk-forward metrics: %s", json.dumps(metrics, indent=2))


@app.command("compare")
def compare(
    models_dir: Path | None = typer.Option(None, help="Defaults to outputs/walk_forward"),
):
    """Roll up metrics.json from each model's walk-forward dir into a CSV."""
    log = get_logger("walk_forward.compare")
    settings = get_settings()
    models_dir = models_dir or settings.outputs_dir / "walk_forward"
    rows = []
    for mp in sorted(models_dir.glob("*/metrics.json")):
        m = json.loads(mp.read_text())
        rows.append({"model": mp.parent.name, **m})
    if not rows:
        log.warning("No walk-forward metrics found under %s", models_dir)
        return
    df = pd.DataFrame(rows).sort_values("mae")
    out_csv = models_dir / "comparison.csv"
    df.to_csv(out_csv, index=False)
    log.info("Wrote walk-forward comparison -> %s", out_csv)
    print(df.to_string(index=False))


if __name__ == "__main__":
    app()
