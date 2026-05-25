"""Feature-group ablation for monthly SMP models.

Runs each (model × feature_group) combination on the chronologically-split
test set and reports:
  - level MAE / RMSE / MAPE / R^2
  - delta MAE  (target_delta vs predicted_delta)
  - delta direction accuracy
  - n_features_used

Groups (per round-5 spec):

    calendar_only
    smp_lags_without_smp_t_observed
    smp_t_observed_only
    smp_t_observed_plus_calendar
    smp_t_observed_plus_settlement
    smp_t_observed_plus_capacity
    smp_t_observed_plus_transaction
    all_features

We deliberately run a small set of models so the table fits in one screen:

    persistence_monthly  (baseline; smp_t_observed only — equivalent to
                          delta=0)
    monthly_ar_ridge
    ridge
    delta_ridge

Outputs:
    outputs/metrics/feature_group_ablation_monthly.csv
    outputs/metrics/feature_group_ablation_monthly.json

Run:
    python -m src.pipelines.feature_group_ablation \
        --features-path data/processed/smp_monthly_mainland_h1m.parquet
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import typer

from src.config.settings import get_settings
from src.models.ar_monthly import MonthlyARRidge
from src.models.delta_models import DeltaRidge, compute_delta_metrics
from src.models.lightgbm_model import LightGBMModel  # noqa: F401  (future use)
from src.models.metrics import compute_metrics
from src.models.naive import PersistenceMonthly
from src.models.ridge_model import RidgeModel
from src.utils.logging import get_logger

app = typer.Typer(help="Feature-group ablation for monthly SMP models.")


CALENDAR_COLS = [
    "year", "month", "quarter", "month_sin", "month_cos",
    "is_summer", "is_winter", "is_peak_season",
]

SMP_LAG_COLS = [
    "smp_lag_1m", "smp_lag_2m", "smp_lag_3m", "smp_lag_6m", "smp_lag_12m",
    "smp_rolling_3m_mean", "smp_rolling_6m_mean",
    "smp_rolling_12m_mean", "smp_rolling_12m_std",
]

OBSERVED_COL = "smp_t_observed"


def _resolve_group(group_name: str, all_cols: list[str]) -> list[str]:
    """Return the subset of `all_cols` belonging to the named feature group."""

    def keep(seq):
        return [c for c in seq if c in all_cols]

    if group_name == "calendar_only":
        return keep(CALENDAR_COLS)
    if group_name == "smp_lags_without_smp_t_observed":
        return keep(SMP_LAG_COLS)
    if group_name == "smp_t_observed_only":
        return [OBSERVED_COL] if OBSERVED_COL in all_cols else []
    if group_name == "smp_t_observed_plus_calendar":
        return ([OBSERVED_COL] if OBSERVED_COL in all_cols else []) + keep(CALENDAR_COLS)
    if group_name == "smp_t_observed_plus_settlement":
        return ([OBSERVED_COL] if OBSERVED_COL in all_cols else []) + [
            c for c in all_cols if c.startswith("settlement_unit_price_")
        ]
    if group_name == "smp_t_observed_plus_capacity":
        return ([OBSERVED_COL] if OBSERVED_COL in all_cols else []) + [
            c for c in all_cols if c.startswith("capacity_")
        ]
    if group_name == "smp_t_observed_plus_transaction":
        return ([OBSERVED_COL] if OBSERVED_COL in all_cols else []) + [
            c for c in all_cols if c.startswith(("transaction_", "market_trade_price_"))
        ]
    if group_name == "all_features":
        return list(all_cols)
    raise typer.BadParameter(f"Unknown feature group: {group_name!r}")


GROUPS = [
    "calendar_only",
    "smp_lags_without_smp_t_observed",
    "smp_t_observed_only",
    "smp_t_observed_plus_calendar",
    "smp_t_observed_plus_settlement",
    "smp_t_observed_plus_capacity",
    "smp_t_observed_plus_transaction",
    "all_features",
]

MODELS = {
    "persistence_monthly": PersistenceMonthly,   # baseline reference
    "monthly_ar_ridge": MonthlyARRidge,
    "ridge": RidgeModel,
    "delta_ridge": DeltaRidge,
}


def _split_chronological(df: pd.DataFrame, ts_col: str,
                         valid_frac: float, test_frac: float):
    df = df.sort_values(ts_col).reset_index(drop=True)
    n = len(df)
    n_test = int(n * test_frac)
    n_valid = int(n * valid_frac)
    n_train = n - n_test - n_valid
    return df.iloc[:n_train], df.iloc[n_train:n_train + n_valid], df.iloc[n_train + n_valid:]


@app.command()
def main(
    features_path: Path = typer.Option(...),
    target: str = typer.Option("target_smp_t_plus_h_months"),
    timestamp_col: str = typer.Option("period_month"),
    valid_frac: float = typer.Option(0.15),
    test_frac: float = typer.Option(0.15),
):
    log = get_logger("feature_group_ablation")
    df = pd.read_parquet(features_path)
    if target not in df.columns:
        raise typer.BadParameter(f"target {target!r} not in features")
    meta_cols = {target, timestamp_col, "area", "smp_krw_per_kwh",
                 "forecast_origin_month", "target_month",
                 "information_cutoff", "horizon"}
    all_feature_cols = [c for c in df.columns if c not in meta_cols
                        and df[c].dtype != object]
    train_df, _valid, test_df = _split_chronological(
        df, timestamp_col, valid_frac, test_frac
    )

    rows: list[dict] = []
    for group in GROUPS:
        cols = _resolve_group(group, all_feature_cols)
        if not cols:
            log.warning("Group %s is empty on this dataset — skipping.", group)
            continue
        for model_name, cls in MODELS.items():
            # Persistence-only is informative only against smp_t_observed-
            # based groups; skip when smp_t_observed isn't there.
            if model_name == "persistence_monthly" and OBSERVED_COL not in cols:
                continue
            # Delta models require smp_t_observed in the column set, AND
            # at least one OTHER column to learn the residual from. The
            # `smp_t_observed_only` group has nothing left after the wrapper
            # strips the observed column, so the delta variant degenerates
            # to "predict 0 residual" — which is exactly persistence and is
            # already covered by the persistence_monthly row.
            if model_name == "delta_ridge":
                if OBSERVED_COL not in cols:
                    continue
                if len([c for c in cols if c != OBSERVED_COL]) == 0:
                    continue
            try:
                # CRITICAL: pass feature_cols=cols explicitly so the model
                # uses EXACTLY the group's columns instead of silently
                # filtering down to its hardcoded default list. Without
                # this, monthly_ar_ridge/ridge produce identical MAE
                # across every group containing smp_t_observed because
                # they ignore the extra capacity/transaction/settlement
                # columns the group meant to test.
                if model_name == "persistence_monthly":
                    # PersistenceMonthly hardcodes lag_column='smp_t_observed';
                    # it doesn't accept feature_cols. Skip the override.
                    model = cls()
                else:
                    model = cls(feature_cols=cols)
                # Slice features. Some optional columns carry NaN before
                # their respective source starts (e.g. capacity_* only
                # exists post-2012-12). Median-impute by training-column
                # median so linear models don't crash. The imputation is
                # neutral (zero contribution under StandardScaler).
                X_train_raw = train_df[cols]
                X_test_raw = test_df[cols]
                train_medians = X_train_raw.median(numeric_only=True)
                # Two-stage NaN handling: (1) per-column median from train
                # (handles columns with mixed NaN/non-NaN values), then
                # (2) zero-fill anything still NaN (these are columns whose
                # train-window is ALL NaN — e.g. transaction_*_lag_1m
                # starts in 2023-12 and is entirely missing from the
                # 2011-01..2021-11 training window). Filling with 0 is
                # neutral under StandardScaler and lets Ridge fit; the
                # corresponding zero column contributes nothing to the
                # prediction, which is the honest answer.
                X_train = X_train_raw.fillna(train_medians).fillna(0)
                X_test = X_test_raw.fillna(train_medians).fillna(0)
                model.fit(X_train, train_df[target])
                y_pred = model.predict(X_test)
            except Exception as exc:  # noqa: BLE001
                rows.append({
                    "model": model_name, "feature_group": group,
                    "n_features": len(cols), "error": repr(exc),
                })
                continue
            obs = test_df[OBSERVED_COL] if OBSERVED_COL in test_df.columns else (
                pd.Series(np.zeros(len(test_df)), index=test_df.index)
            )
            delta = compute_delta_metrics(test_df[target], y_pred, obs)
            level = compute_metrics(test_df[target], y_pred)
            rows.append({
                "model": model_name,
                "feature_group": group,
                "n_features": len(cols),
                "mae_level": level.mae,
                "rmse_level": level.rmse,
                "mape_level": level.mape,
                "r2_level": level.r2,
                "delta_mae": delta["delta_mae"],
                "delta_direction_accuracy": delta["delta_direction_accuracy"],
                "n_observations": len(test_df),
            })

    out_df = pd.DataFrame(rows).sort_values(["feature_group", "mae_level"])
    settings = get_settings()
    out_dir = settings.outputs_dir / "metrics"
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "feature_group_ablation_monthly.csv"
    json_path = out_dir / "feature_group_ablation_monthly.json"
    out_df.to_csv(csv_path, index=False)
    json_path.write_text(
        json.dumps(out_df.to_dict(orient="records"), indent=2, default=str),
        encoding="utf-8",
    )
    log.info("Wrote feature-group ablation -> %s + .json", csv_path)
    print(out_df.round(3).to_string(index=False))


if __name__ == "__main__":
    app()
