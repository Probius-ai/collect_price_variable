"""Train one of {naive, seasonal_naive, ridge, lightgbm} on the SMP features.

This pipeline:

1. Loads the feature dataframe (Parquet) produced by `build_features.py`.
2. Splits chronologically into train / valid / test.
3. Fits the requested model.
4. Writes predictions, metrics JSON, and feature importance (LightGBM only).
"""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Optional

import pandas as pd
import typer

from src.config.settings import get_settings
from src.models.lightgbm_model import LightGBMModel
from src.models.metrics import compute_metrics
from src.models.naive import (
    NaiveLag1m,
    NaiveLag24h,
    PersistenceMonthly,
    SeasonalNaiveLag12m,
    SeasonalNaiveLag168h,
)
from src.models.ridge_model import RidgeModel
from src.models.ar_monthly import MonthlyARRidge
from src.models.delta_models import DeltaARRidge, DeltaLightGBM, DeltaRidge
from src.utils.logging import get_logger

app = typer.Typer(help="Train SMP forecasting models.")

MODELS = {
    "naive": NaiveLag24h,                  # hourly
    "seasonal_naive": SeasonalNaiveLag168h,
    "naive_lag_1m": NaiveLag1m,            # monthly: SMP(M+1) ≈ SMP(M-1) (2-step lag)
    "persistence_monthly": PersistenceMonthly,   # monthly: SMP(M+1) ≈ SMP(M) (true naive)
    "seasonal_naive_lag_12m": SeasonalNaiveLag12m,
    "ridge": RidgeModel,
    "lightgbm": LightGBMModel,
    "monthly_ar_ridge": MonthlyARRidge,    # monthly: lag_1m + rolling_3m + lag_12m
    # Delta-target wrappers: learn y_delta = target − smp_t_observed, then
    # reconstruct as smp_t_observed + predicted_delta. Lets the model focus
    # on the residual on top of the persistence baseline.
    "delta_ridge": DeltaRidge,
    "delta_lightgbm": DeltaLightGBM,
    "delta_ar_ridge": DeltaARRidge,
}


def _split_chronological(
    df: pd.DataFrame,
    valid_frac: float,
    test_frac: float,
    ts_col: str,
    min_train_rows: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    df = df.sort_values(ts_col).reset_index(drop=True)
    n = len(df)
    n_test = int(n * test_frac)
    n_valid = int(n * valid_frac)
    n_train = n - n_test - n_valid
    if n_train < min_train_rows:
        raise ValueError(
            f"Train split too small ({n_train} rows < min_train_rows={min_train_rows}). "
            "Lower --valid-frac / --test-frac or pass --min-train-rows."
        )
    return df.iloc[:n_train], df.iloc[n_train : n_train + n_valid], df.iloc[n_train + n_valid :]


@app.command()
def main(
    features_path: Path = typer.Option(..., help="Parquet from build_features.py"),
    model: str = typer.Option(..., help=f"One of: {sorted(MODELS)}"),
    target: str = typer.Option("target_smp_t_plus_h"),
    timestamp_col: str = typer.Option("interval_end"),
    valid_frac: float = typer.Option(0.15),
    test_frac: float = typer.Option(0.15),
    min_train_rows: int = typer.Option(
        24,
        help="Minimum train rows required (hourly: ≥168 recommended; monthly: ≥24).",
    ),
    output_dir: Optional[Path] = typer.Option(None, help="Defaults to outputs/<model>/"),
):
    log = get_logger("train")
    if model not in MODELS:
        raise typer.BadParameter(f"Unknown model {model!r}. Known: {sorted(MODELS)}")
    settings = get_settings()
    out_dir = output_dir or (settings.outputs_dir / "models" / model)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_parquet(features_path)
    if target not in df.columns:
        raise typer.BadParameter(f"target column {target!r} not in features")
    drop_cols = {target, timestamp_col, "area"}
    feature_cols = [c for c in df.columns if c not in drop_cols]

    train_df, valid_df, test_df = _split_chronological(
        df, valid_frac, test_frac, timestamp_col, min_train_rows
    )
    log.info(
        "Split sizes: train=%d valid=%d test=%d", len(train_df), len(valid_df), len(test_df)
    )

    model_obj = MODELS[model]()
    fit_kwargs = {}
    if model == "lightgbm" and len(valid_df):
        fit_kwargs = {
            "X_valid": valid_df[feature_cols],
            "y_valid": valid_df[target],
        }
    model_obj.fit(train_df[feature_cols], train_df[target], **fit_kwargs)

    def _eval(name: str, split: pd.DataFrame) -> dict:
        if split.empty:
            return {}
        preds = model_obj.predict(split[feature_cols])
        metrics = compute_metrics(split[target], preds).to_dict()

        # Pull forecast-origin metadata if the feature pipeline added it.
        # These were added in round 5 to make the forecast contract explicit
        # (what month is being predicted, what was knowable when).
        meta_cols = ["forecast_origin_month", "target_month",
                     "information_cutoff", "horizon"]
        out_dict = {
            timestamp_col: split[timestamp_col].values,
            "area": split.get("area", pd.Series(["mainland"] * len(split))).values,
            "y_true": split[target].values,
            "y_pred": preds.values,
        }
        for c in meta_cols:
            if c in split.columns:
                out_dict[c] = split[c].values

        # Delta-target diagnostics: when smp_t_observed is in features AND
        # the model exposes predict_delta, compute the delta-MAE and the
        # sign-agreement of the predicted delta. We compute the same delta
        # info even for non-delta models so the columns are comparable
        # across the whole MODELS table.
        if "smp_t_observed" in split.columns:
            obs = split["smp_t_observed"].to_numpy(dtype=float)
            true_delta = split[target].to_numpy(dtype=float) - obs
            pred_delta = preds.to_numpy(dtype=float) - obs
            out_dict["smp_t_observed"] = obs
            out_dict["true_delta_1m"] = true_delta
            out_dict["predicted_delta_1m"] = pred_delta
            import numpy as np
            metrics["delta_mae"] = float(np.mean(np.abs(true_delta - pred_delta)))
            sign_mask = true_delta != 0
            metrics["delta_direction_accuracy"] = (
                float((np.sign(true_delta[sign_mask])
                       == np.sign(pred_delta[sign_mask])).mean())
                if sign_mask.any() else float("nan")
            )

        out = pd.DataFrame(out_dict)
        out.to_csv(out_dir / f"predictions_{name}.csv", index=False)
        return metrics

    all_metrics = {
        "train": _eval("train", train_df),
        "valid": _eval("valid", valid_df),
        "test": _eval("test", test_df),
    }
    (out_dir / "metrics.json").write_text(json.dumps(all_metrics, indent=2))
    log.info("Metrics: %s", json.dumps(all_metrics, indent=2))

    if model == "lightgbm":
        imp = model_obj.feature_importance()
        imp.to_csv(out_dir / "feature_importance.csv", index=False)
        log.info("Top features:\n%s", imp.head(15).to_string(index=False))


if __name__ == "__main__":
    app()
