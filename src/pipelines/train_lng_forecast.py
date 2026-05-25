"""Train and compare LNG forecast models per the PDF guide.

This is the round-7 LNG forecasting pipeline. It mirrors the design from
`docs/에너지가격예측_파이프라인_가이드.pdf` (TimeSeriesSplit + embargo,
per-fold scaler fit, RMSE/MAPE fold-averaged with std) and reports
LightGBM vs MLP vs Naive-lag1 on the same fold structure.

Run:
  python -m src.pipelines.train_lng_forecast
  python -m src.pipelines.train_lng_forecast --n-splits 8 --embargo 2
  python -m src.pipelines.train_lng_forecast --horizon 1 --no-mlp
"""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import pandas as pd
import typer

from src.config.settings import get_settings
from src.models.lng_forecast.evaluate import run_cv
from src.models.lng_forecast.features import build_lng_forecast_features
from src.models.lng_forecast.models import (
    LightGBMForecaster,
    MLPForecaster,
    NaiveLagForecaster,
)
from src.utils.logging import get_logger

app = typer.Typer(help="Train + compare LNG monthly forecast models.")


@app.command()
def main(
    horizon: int = typer.Option(1, help="Forecast horizon in months."),
    n_splits: int = typer.Option(5, help="TimeSeriesSplit n_splits."),
    embargo: int = typer.Option(1, help="Rows dropped from train tail per fold."),
    enable_mlp: bool = typer.Option(True, "--mlp/--no-mlp"),
    output_dir: Path | None = typer.Option(None,
        help="Defaults to outputs/lng_forecast/<YYYYMMDD>/"),
):
    log = get_logger("train_lng_forecast")

    log.info("Building features (horizon=%dM)...", horizon)
    X, y, period_months, info = build_lng_forecast_features(horizon_months=horizon)
    log.info(
        "Panel: rows=%d, period=%s..%s, features=%d, dropped_nan=%d",
        info.rows, info.period_min.date(), info.period_max.date(),
        info.n_features, info.n_dropped_for_nan,
    )

    settings = get_settings()
    out_dir = output_dir or (
        settings.outputs_dir / "lng_forecast"
        / f"{pd.Timestamp.utcnow():%Y%m%d}_h{horizon}m_cv{n_splits}_emb{embargo}"
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    log.info("Output dir: %s", out_dir)

    # ---- Models per the PDF guide -------------------------------------
    factories = {
        "naive_lag1": lambda: NaiveLagForecaster(),
        "lightgbm": lambda: LightGBMForecaster(),
    }
    if enable_mlp:
        factories["mlp"] = lambda: MLPForecaster()

    aggregates = []
    for name, factory in factories.items():
        log.info("CV %s ...", name)
        rep = run_cv(
            X, y, period_months,
            model_factory=factory,
            model_name=name,
            n_splits=n_splits,
            embargo=embargo,
        )
        agg = rep.aggregate()
        aggregates.append(agg)
        log.info(
            "%s: RMSE %.4f±%.4f | MAPE %.2f±%.2f%% | MAE %.4f±%.4f",
            name, agg["rmse_mean"], agg["rmse_std"],
            agg["mape_mean"], agg["mape_std"],
            agg["mae_mean"], agg["mae_std"],
        )

        # Persist per-model outputs.
        model_dir = out_dir / name
        model_dir.mkdir(parents=True, exist_ok=True)
        pd.DataFrame([asdict(f) for f in rep.folds]).to_csv(
            model_dir / "fold_metrics.csv", index=False
        )
        if rep.predictions is not None:
            rep.predictions.to_csv(model_dir / "predictions.csv", index=False)

    # ---- Aggregate comparison ----------------------------------------
    comp = pd.DataFrame(aggregates).sort_values("mae_mean")
    comp.to_csv(out_dir / "comparison.csv", index=False)
    (out_dir / "panel_info.json").write_text(json.dumps({
        "rows": info.rows,
        "period_min": str(info.period_min.date()),
        "period_max": str(info.period_max.date()),
        "n_features": info.n_features,
        "horizon_months": info.horizon_months,
        "n_dropped_for_nan": info.n_dropped_for_nan,
        "feature_columns": list(X.columns),
        "n_splits": n_splits,
        "embargo": embargo,
    }, indent=2), encoding="utf-8")

    log.info("Wrote comparison + panel_info to %s", out_dir)
    print(f"\n=== LNG monthly forecast — CV comparison (horizon={horizon}M) ===")
    print(comp.round(3).to_string(index=False))


if __name__ == "__main__":
    app()
