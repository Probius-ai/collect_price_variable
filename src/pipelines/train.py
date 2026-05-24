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
    SeasonalNaiveLag12m,
    SeasonalNaiveLag168h,
)
from src.models.ridge_model import RidgeModel
from src.utils.logging import get_logger

app = typer.Typer(help="Train SMP forecasting models.")

MODELS = {
    "naive": NaiveLag24h,                  # hourly
    "seasonal_naive": SeasonalNaiveLag168h,
    "naive_lag_1m": NaiveLag1m,            # monthly
    "seasonal_naive_lag_12m": SeasonalNaiveLag12m,
    "ridge": RidgeModel,
    "lightgbm": LightGBMModel,
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
        out = pd.DataFrame(
            {
                timestamp_col: split[timestamp_col].values,
                "area": split.get("area", pd.Series(["mainland"] * len(split))).values,
                "y_true": split[target].values,
                "y_pred": preds.values,
            }
        )
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
