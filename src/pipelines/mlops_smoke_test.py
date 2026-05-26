"""MLOps staged-retraining smoke test (v1..v5).

Simulates the workflow of retraining as new data becomes available, by
truncating the feature panel to successively-later cutoff months. For
each (version, model) pair we:

  1. Filter the feature table to ``period_month <= data_cutoff_month``
     (no future leakage — this is what makes the simulation honest).
  2. Train the requested model on training rows (where the target is
     non-null and falls within the train window).
  3. Evaluate per the version's ``evaluation_mode``:
       * fixed_holdout — score on rows where
         ``test_start <= period_month <= test_end``.
       * latest_rolling_validation — predict each of the last N months
         ending at the cutoff using only data strictly before that
         month (a backward walk-forward).
  4. Log one MLflow run per (version, model) with params + metrics +
     artifacts (predictions.csv, metrics.json, model.pkl, run_summary).
  5. Append a JSON registry record so the version history survives
     even if MLflow's own backend is down.

After all versions finish, write a comparison report (md + json + csv)
under ``outputs/reports/`` and ``outputs/metrics/``.

Promotion: NEVER automatic. We only tag rows
  * historical_backtest    (v1..v4)
  * latest_candidate       (v5)
  * recommended_historical (best v1..v4 by MAE, regardless of model)

CLI:
    python -m src.pipelines.mlops_smoke_test
        --config config/mlops_smoke_test.yaml
        [--log-to-mlflow]
        [--only-versions v1,v2]
        [--only-models persistence_monthly,ridge]
"""

from __future__ import annotations

import json
import pickle
from dataclasses import dataclass, asdict, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd
import typer
import yaml

from src.models.ar_monthly import MonthlyARRidge
from src.models.delta_models import DeltaARRidge, DeltaLightGBM, DeltaRidge
from src.models.lightgbm_model import LightGBMModel
from src.models.metrics import compute_metrics
from src.models.naive import NaiveLag1m, PersistenceMonthly, SeasonalNaiveLag12m
from src.models.ridge_model import RidgeModel
from src.tracking.mlflow_utils import (
    RegistryRecord,
    append_registry_record,
    load_registry,
    maybe_mlflow_run,
    write_run_summary,
)
from src.utils.logging import get_logger

# Iterative-training models for learning-curve visualisation in MLflow.
# Each exposes an `eval_history` (or `loss_curve_`) attribute that the
# smoke test extracts and logs as a stepped metric series.
#
# Soft imports — torch / xgboost are heavy optional deps. The factory
# raises at call time when missing, which the per-(version, model) loop
# converts to a controlled skip with a logged reason.
try:
    from src.models.iterative_models import (
        MLPMonthlyModel,
        XGBoostMonthlyModel,
        TorchMLPModel,
        TorchLSTMModel,
    )
    _HAS_ITERATIVE = True
except Exception:
    _HAS_ITERATIVE = False

# Legacy LNG forecaster — kept around so older registry rows referencing
# `mlp_lng` still resolve.
try:
    from src.models.lng_forecast.models import MLPForecaster  # noqa: F401
    _HAS_LNG_MLP = True
except Exception:
    _HAS_LNG_MLP = False


log = get_logger("mlops_smoke_test")
app = typer.Typer(add_completion=False, help="MLOps v1..v5 retraining smoke test.")


# ---------------------------------------------------------------------------
# Model registry — name → factory. Mirrors src/pipelines/train.py but
# scoped to monthly-applicable models so the smoke test can't accidentally
# load an hourly baseline.
# ---------------------------------------------------------------------------


def _make_persistence_monthly():
    return PersistenceMonthly()


def _make_naive_lag_1m():
    return NaiveLag1m(lag_column="smp_lag_1m")


def _make_seasonal_naive():
    return SeasonalNaiveLag12m()


def _make_ridge():
    return RidgeModel()


def _make_monthly_ar_ridge():
    return MonthlyARRidge()


def _make_delta_ridge():
    return DeltaRidge()


def _make_delta_ar_ridge():
    return DeltaARRidge()


def _make_lightgbm():
    return LightGBMModel()


def _make_delta_lightgbm():
    return DeltaLightGBM()


def _make_mlp():
    """sklearn MLPRegressor with NaN imputation + scaling pipeline.

    Replaces the prior MLPForecaster-on-monthly-panel path that crashed
    with `Input X contains NaN` because the panel has sparse columns
    (JKM lags from 2013+, settlement/capacity publication gaps).
    """
    if not _HAS_ITERATIVE:
        raise RuntimeError("MLPMonthlyModel unavailable — check sklearn import")
    return MLPMonthlyModel()


def _make_xgboost():
    if not _HAS_ITERATIVE:
        raise RuntimeError("XGBoostMonthlyModel unavailable — `pip install xgboost`")
    return XGBoostMonthlyModel()


def _make_torch_mlp():
    if not _HAS_ITERATIVE:
        raise RuntimeError("TorchMLPModel unavailable — `pip install torch`")
    return TorchMLPModel()


def _make_torch_lstm():
    if not _HAS_ITERATIVE:
        raise RuntimeError("TorchLSTMModel unavailable — `pip install torch`")
    return TorchLSTMModel()


MODEL_FACTORIES: dict[str, Callable[[], Any]] = {
    "persistence_monthly":   _make_persistence_monthly,
    "naive_lag_1m":          _make_naive_lag_1m,
    "seasonal_naive_lag_12m": _make_seasonal_naive,
    "ridge":                 _make_ridge,
    "monthly_ar_ridge":      _make_monthly_ar_ridge,
    "delta_ridge":           _make_delta_ridge,
    "delta_ar_ridge":        _make_delta_ar_ridge,
    "lightgbm":              _make_lightgbm,
    "delta_lightgbm":        _make_delta_lightgbm,
    "mlp":                   _make_mlp,
    "xgboost":               _make_xgboost,
    "torch_mlp":             _make_torch_mlp,
    "torch_lstm":            _make_torch_lstm,
}


# ---------------------------------------------------------------------------
# Result dataclasses
# ---------------------------------------------------------------------------


@dataclass
class VersionResult:
    version: str
    model_name: str
    data_cutoff_month: str
    train_end: str | None
    test_start: str | None
    test_end: str | None
    evaluation_mode: str
    n_train: int
    n_test: int
    metrics: dict[str, float]
    mlflow_run_id: str | None
    artifact_dir: str | None
    skipped: bool = False
    skip_reason: str | None = None


# ---------------------------------------------------------------------------
# Core data filtering — the leak-safety contract for v1..v5
# ---------------------------------------------------------------------------


def filter_to_cutoff(
    df: pd.DataFrame,
    *,
    data_cutoff_month: pd.Timestamp,
    horizon_months: int = 0,
) -> pd.DataFrame:
    """Restrict the feature table to rows observable at the cutoff.

    Two modes:

    * ``horizon_months=0`` (default) — keep rows where
      ``period_month <= data_cutoff_month``. This is the "features
      observable" filter — every input column for that row was computed
      from data available by end-of-cutoff-month.

    * ``horizon_months>0`` — also enforce that the row's TARGET would
      have been observable by the cutoff, i.e. ``period_month + horizon
      <= cutoff``. This is the **strict-correct filter for training
      data**: a row whose target month is in the future relative to the
      simulated retraining moment carries a label that didn't exist at
      retraining time, and including it leaks future information into
      the fit. For h=1 monthly, this means dropping the row at
      ``period_month == cutoff`` (target = cutoff + 1 = future).

    The looser ``horizon_months=0`` form is kept for the test-set path
    (where we DO want to score predictions against retrospectively-
    observable labels) and for the structural leak-safety canaries that
    just check "no rows after cutoff".
    """
    if "period_month" not in df.columns:
        raise KeyError("Feature table missing required column `period_month`")
    pm = pd.to_datetime(df["period_month"])
    if horizon_months > 0:
        # Target month = period_month + horizon. Require it to be
        # observable at the cutoff (i.e., <= cutoff).
        target_month = pm + pd.DateOffset(months=horizon_months)
        mask = target_month <= data_cutoff_month
    else:
        mask = pm <= data_cutoff_month
    return df.loc[mask].copy()


def _split_train_test(
    df: pd.DataFrame,
    *,
    target_col: str,
    train_end: pd.Timestamp,
    test_start: pd.Timestamp | None,
    test_end: pd.Timestamp | None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    pm = pd.to_datetime(df["period_month"])
    train_mask = (pm <= train_end) & df[target_col].notna()
    train = df.loc[train_mask].copy()
    if test_start is None or test_end is None:
        return train, df.iloc[0:0].copy()
    test_mask = (pm >= test_start) & (pm <= test_end) & df[target_col].notna()
    test = df.loc[test_mask].copy()
    return train, test


def _drop_leakage_columns(df: pd.DataFrame, target_col: str) -> pd.DataFrame:
    """Drop the column the model is predicting + every non-numeric index
    column.

    Models receive the remaining columns as a candidate feature pool;
    their own `_select_features` then narrows further (e.g. Ridge picks
    from `DEFAULT_RIDGE_FEATURES_MONTHLY`). We drop ALL datetime + object
    columns here rather than enumerating names — the feature builder
    emits several bookkeeping timestamps (forecast_origin_month,
    target_month, information_cutoff) that sklearn rejects with a
    "float() argument must be ... not 'Timestamp'" error if they slip
    into `.to_numpy(dtype=float)`.
    """
    # Positive selection: keep numeric (+ boolean) columns. Negative
    # enumeration of "non-numeric" types is fragile because pandas exposes
    # several string-like dtypes (`object`, `str`, `string[pyarrow]`)
    # depending on version, and the feature builder may emit any of them.
    drop_cols: set[str] = {target_col}
    for col in df.columns:
        s = df[col]
        if not (pd.api.types.is_numeric_dtype(s) or pd.api.types.is_bool_dtype(s)):
            drop_cols.add(col)
    return df.drop(columns=[c for c in drop_cols if c in df.columns])


# ---------------------------------------------------------------------------
# Evaluation modes
# ---------------------------------------------------------------------------


def _evaluate_fixed_holdout(
    model: Any,
    train: pd.DataFrame,
    test: pd.DataFrame,
    *,
    target_col: str,
) -> tuple[pd.DataFrame, dict[str, float]]:
    X_train = _drop_leakage_columns(train, target_col)
    y_train = train[target_col].astype(float)
    # Hand the validation set into fit() for models that support it.
    # LightGBM uses it for (a) early stopping and (b) populating its
    # per-iteration `eval_history`, which the smoke test then logs as
    # a learning curve. Non-iterative models silently ignore the
    # extra kwargs (they don't accept them) — guard with introspection.
    import inspect
    fit_params = inspect.signature(model.fit).parameters
    X_test_for_fit = (
        _drop_leakage_columns(test, target_col)
        if (not test.empty and "X_valid" in fit_params)
        else None
    )
    if X_test_for_fit is not None:
        model.fit(
            X_train, y_train,
            X_valid=X_test_for_fit,
            y_valid=test[target_col].astype(float),
        )
    else:
        model.fit(X_train, y_train)

    if test.empty:
        return pd.DataFrame(columns=["period_month", "y_true", "y_pred"]), {}

    X_test = _drop_leakage_columns(test, target_col)
    y_pred = pd.Series(model.predict(X_test)).reset_index(drop=True)
    y_true = test[target_col].astype(float).reset_index(drop=True)
    preds = pd.DataFrame({
        "period_month": pd.to_datetime(test["period_month"]).reset_index(drop=True),
        "y_true": y_true,
        "y_pred": y_pred,
    })
    metrics = compute_metrics(y_true, y_pred)
    return preds, _metrics_to_dict(metrics)


def _extract_learning_curve(model: Any) -> dict[str, list[float]] | None:
    """Pull per-iteration metric histories out of a fitted model.

    Returns a FLAT dict ``{metric_key: [values_per_iter]}`` so the
    smoke test can log each series to MLflow under its own metric
    name and the UI can overlay them across models.

    Key naming convention (so cross-model overlay works in MLflow's
    Compare-runs view): each metric from the inner `eval_history`
    dict is renamed `{metric_name}_{dataset_name}` — e.g.
    `rmse_valid`, `rmse_train`, `r2_valid`. So when the user picks
    `metric=rmse_valid` in Compare → all iterative models that had a
    validation set are overlaid with one line per model.

    Source mapping:
      * LightGBM / XGBoost — `eval_history` dict from the boost callback
      * Torch models (MLP / LSTM) — `eval_history` dict from the manual
        training loop
      * sklearn MLPRegressor — `loss_curve_` (training MSE → RMSE) +
        optional `validation_scores_` (R² on internal early-stopping slice)
      * Wrappers (`_DeltaWrapped`) — recurse into `.base` / `.base_model`
        / `.model` until we find a model with one of the above
    Returns ``None`` for closed-form models (Ridge etc.) so the smoke
    test knows there's no curve to log.
    """
    history = getattr(model, "eval_history", None)
    if isinstance(history, dict) and history:
        out: dict[str, list[float]] = {}
        for dataset_name, metrics in history.items():
            if not isinstance(metrics, dict):
                continue
            for metric_name, values in metrics.items():
                try:
                    out[f"{metric_name}_{dataset_name}"] = [float(v) for v in values]
                except (TypeError, ValueError):
                    continue
        if out:
            return out

    # sklearn MLPRegressor exposes loss_curve_ — fitted training loss
    # per iteration. Reported as `rmse_train` (sqrt of MSE) for unit-
    # consistency with the other iterative models.
    if hasattr(model, "loss_curve_"):
        try:
            import math
            out: dict[str, list[float]] = {
                "rmse_train": [math.sqrt(max(0.0, float(v))) for v in model.loss_curve_],
            }
            if (hasattr(model, "validation_scores_")
                    and getattr(model, "validation_scores_")):
                out["r2_valid"] = [float(v) for v in model.validation_scores_]
            return out
        except Exception:
            pass

    # Wrappers (DeltaLightGBM etc.) — recurse into the inner model
    for inner_attr in ("base", "base_model", "model"):
        inner = getattr(model, inner_attr, None)
        if inner is not None and inner is not model:
            recurse = _extract_learning_curve(inner)
            if recurse is not None:
                return recurse

    return None


def _evaluate_latest_rolling_validation(
    model_factory: Callable[[], Any],
    panel: pd.DataFrame,
    *,
    target_col: str,
    data_cutoff_month: pd.Timestamp,
    window_months: int,
    horizon_months: int = 1,
) -> tuple[pd.DataFrame, dict[str, float]]:
    """Backward walk-forward, observable-label-only.

    Two leak-safety constraints, BOTH required to honour v5's spec
    "do not fabricate future labels":

      1. **Eval months are bounded by target observability.** The latest
         eval month is ``cutoff - horizon``, not ``cutoff``. For h=1,
         scoring against ``m = cutoff`` would mean comparing the
         prediction against ``SMP(cutoff + 1)`` — a value that doesn't
         exist at the simulated retraining moment. The rolling window
         is the last ``window_months`` months whose target is observable.

      2. **Per-fold training mask is target-bounded too.** For each fold
         month m, training rows are those where ``period_month + horizon
         < m`` (the row's TARGET landed strictly before m). The looser
         ``period_month < m`` would let in rows whose target IS m or
         later — leaking the fold's eval label into its training fit.

    Together: every (train_target, eval_target) pair satisfies
    train_target < eval_target ≤ cutoff.
    """
    pm = pd.to_datetime(panel["period_month"])
    target_month = pm + pd.DateOffset(months=horizon_months)

    # Latest eval month whose TARGET is observable at the simulated cutoff
    latest_eval_month = data_cutoff_month - pd.DateOffset(months=horizon_months)
    months = pd.date_range(
        end=latest_eval_month, periods=window_months, freq="MS"
    )

    rows: list[dict[str, Any]] = []
    # Stash the last successful fold's model so the caller can pull a
    # learning curve out of it (LightGBM eval_history is per-model, so
    # we can only show ONE curve for the whole rolling sequence — pick
    # the latest fold since it sees the most data).
    last_fold_model: Any | None = None
    import inspect

    for m in months:
        # Train: row's target must land strictly before this fold's eval
        train_mask = (target_month < m) & panel[target_col].notna()
        train = panel.loc[train_mask]
        # Eval: single row at this fold's month, whose target IS by
        # construction observable at the cutoff (we bounded `months` above).
        eval_row = panel.loc[(pm == m) & panel[target_col].notna()]
        if eval_row.empty or train.empty:
            continue
        m_model = model_factory()
        X_train = _drop_leakage_columns(train, target_col)
        y_train = train[target_col].astype(float)

        # Carve a small per-fold validation slice from the END of the
        # training window so iterative models (LightGBM family) can:
        #   * early-stop on it
        #   * populate `eval_history` → per-iteration learning curve
        # Strict leak-safety: this slice is BEFORE the eval row's
        # target month (it's drawn from `train`, which already passed
        # `target_month < m`). So the curve doesn't see the fold's
        # actual eval label.
        fit_kwargs: dict[str, Any] = {}
        try:
            if (
                "X_valid" in inspect.signature(m_model.fit).parameters
                and len(X_train) >= 10
            ):
                slice_n = max(3, len(X_train) // 7)  # ~15 %
                fit_kwargs["X_valid"] = X_train.iloc[-slice_n:]
                fit_kwargs["y_valid"] = y_train.iloc[-slice_n:]
        except (TypeError, ValueError):
            fit_kwargs = {}

        m_model.fit(X_train, y_train, **fit_kwargs)
        X_eval = _drop_leakage_columns(eval_row, target_col)
        y_pred = pd.Series(m_model.predict(X_eval)).iloc[0]
        y_true = float(eval_row[target_col].iloc[0])
        rows.append({"period_month": m, "y_true": y_true, "y_pred": float(y_pred)})
        last_fold_model = m_model

    preds = pd.DataFrame(rows)
    if preds.empty:
        return preds, {}, None
    metrics = compute_metrics(preds["y_true"], preds["y_pred"])
    return preds, _metrics_to_dict(metrics), last_fold_model


def _metrics_to_dict(m: Any) -> dict[str, float]:
    """Extract a dict of finite metric floats from the project's
    ``compute_metrics`` result, which is an ``EvaluationMetrics`` dataclass.

    We deliberately drop NaN/inf values rather than logging them as
    MLflow metrics — MLflow's strict-mode rejects non-finite values.
    """
    if hasattr(m, "__dict__"):
        raw = m.__dict__
    elif isinstance(m, dict):
        raw = m
    else:
        raw = asdict(m)
    out: dict[str, float] = {}
    for k, v in raw.items():
        try:
            f = float(v)
            if np.isfinite(f):
                out[k] = f
        except (TypeError, ValueError):
            continue
    return out


# ---------------------------------------------------------------------------
# Per-(version, model) execution
# ---------------------------------------------------------------------------


def _train_one(
    *,
    version_cfg: dict[str, Any],
    model_cfg: dict[str, Any],
    panel: pd.DataFrame,
    target_col: str,
    log_to_mlflow: bool,
    artifact_root: Path,
    feature_table_path: str,
    horizon_months: int = 1,
) -> VersionResult:
    model_name = model_cfg["name"]
    version = version_cfg["name"]
    cutoff = pd.Timestamp(version_cfg["data_cutoff_month"])
    train_end = pd.Timestamp(version_cfg["train_end"])
    eval_mode = version_cfg["evaluation_mode"]
    test_start = pd.Timestamp(version_cfg["test_start"]) if version_cfg.get("test_start") else None
    test_end = pd.Timestamp(version_cfg["test_end"]) if version_cfg.get("test_end") else None

    artifact_dir = artifact_root / version / model_name
    artifact_dir.mkdir(parents=True, exist_ok=True)

    # Leak-safe filter ONLY for training data. Two layers of "future":
    #   1) row's features (period_month) must be observable at cutoff
    #   2) row's TARGET (period_month + horizon) must also be observable
    #      at cutoff — otherwise the label itself is a future leak
    #      (e.g. for horizon=1 monthly, the row at period_month=cutoff
    #      has target = SMP(cutoff+1) which doesn't exist yet at the
    #      simulated retraining moment).
    # Passing horizon_months>0 enforces both.
    panel_train_pool = filter_to_cutoff(
        panel, data_cutoff_month=cutoff, horizon_months=horizon_months,
    )

    n_train_pool = int((pd.to_datetime(panel_train_pool["period_month"]) <= train_end).sum())
    min_train = int(model_cfg.get("min_train_rows", 0))

    # Controlled-skip — too little data after the cutoff filter
    if n_train_pool < min_train:
        reason = (
            model_cfg.get("skip_reason_if_underfit")
            or f"n_train={n_train_pool} < min_train_rows={min_train} at cutoff {cutoff:%Y-%m}"
        ).strip()
        log.warning("[%s/%s] skipping: %s", version, model_name, reason)
        return _record_skip(
            version=version, model_name=model_name, cutoff=cutoff,
            train_end=train_end, test_start=test_start, test_end=test_end,
            eval_mode=eval_mode, reason=reason, artifact_dir=artifact_dir,
            log_to_mlflow=log_to_mlflow, feature_table_path=feature_table_path,
            n_train=n_train_pool,
        )

    # Special-case: MLP factory raises if the optional dep isn't there.
    try:
        model = MODEL_FACTORIES[model_name]()
    except Exception as exc:
        reason = f"factory error: {exc}"
        log.warning("[%s/%s] skipping: %s", version, model_name, reason)
        return _record_skip(
            version=version, model_name=model_name, cutoff=cutoff,
            train_end=train_end, test_start=test_start, test_end=test_end,
            eval_mode=eval_mode, reason=reason, artifact_dir=artifact_dir,
            log_to_mlflow=log_to_mlflow, feature_table_path=feature_table_path,
            n_train=n_train_pool,
        )

    # Train + evaluate. Any exception from a single (version, model)
    # converts to a controlled skip rather than crashing the whole batch
    # — the whole point of a smoke test is to surface multiple failures
    # in one run, not abort on the first one. The skip reason includes
    # the exception type so an operator can grep the report.
    try:
        if eval_mode == "fixed_holdout":
            # Train rows live within the cutoff; test rows live outside
            # it (forward labels, observable only retrospectively).
            train_df, _ = _split_train_test(
                panel_train_pool, target_col=target_col,
                train_end=train_end, test_start=None, test_end=None,
            )
            pm = pd.to_datetime(panel["period_month"])
            test_mask = (
                (pm >= test_start) & (pm <= test_end)
                & panel[target_col].notna()
            )
            test_df = panel.loc[test_mask].copy()
            preds, metrics = _evaluate_fixed_holdout(
                model, train_df, test_df, target_col=target_col,
            )
            n_train, n_test = len(train_df), len(test_df)
        elif eval_mode == "latest_rolling_validation":
            # Only past data needed — pass the FULL panel so the rolling
            # window can see eval-month rows (period_month==m); the
            # rolling function itself filters training by
            # `period_month + horizon < m` per fold for strict leak-safety.
            window = int(version_cfg.get("rolling_window_months", 12))
            preds, metrics, last_fold_model = _evaluate_latest_rolling_validation(
                MODEL_FACTORIES[model_name], panel,
                target_col=target_col,
                data_cutoff_month=cutoff,
                window_months=window,
                horizon_months=horizon_months,
            )
            # Replace the locally-fit `model` with the last fold's model
            # so the downstream learning-curve extraction picks up the
            # most-data-trained convergence series.
            if last_fold_model is not None:
                model = last_fold_model
            n_train, n_test = n_train_pool, len(preds)
        else:
            raise ValueError(f"Unknown evaluation_mode: {eval_mode!r}")
    except Exception as exc:
        reason = f"{type(exc).__name__}: {exc}"
        log.warning("[%s/%s] fit/predict failed → controlled-skip: %s",
                    version, model_name, reason)
        return _record_skip(
            version=version, model_name=model_name, cutoff=cutoff,
            train_end=train_end, test_start=test_start, test_end=test_end,
            eval_mode=eval_mode, reason=reason, artifact_dir=artifact_dir,
            log_to_mlflow=log_to_mlflow, feature_table_path=feature_table_path,
            n_train=n_train_pool,
        )

    # Always persist predictions + model + run summary locally — these
    # exist even if MLflow is off. The MLflow run additionally tracks
    # them as artifacts.
    preds_path = artifact_dir / "predictions.csv"
    if not preds.empty:
        preds.to_csv(preds_path, index=False)
    metrics_path = artifact_dir / "metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    model_path = artifact_dir / "model.pkl"
    try:
        with model_path.open("wb") as f:
            pickle.dump(model, f)
    except Exception as e:
        log.warning("Could not pickle model %s/%s: %s", version, model_name, e)
        model_path = None

    params = {
        "version": version,
        "retrain_trigger": "manual_cli_smoke_test",
        "data_cutoff_month": str(cutoff.date()),
        "train_end": str(train_end.date()),
        "test_start": str(test_start.date()) if test_start is not None else None,
        "test_end": str(test_end.date()) if test_end is not None else None,
        "evaluation_mode": eval_mode,
        "target": target_col,
        "area": version_cfg.get("area", "mainland"),
        "model_name": model_name,
        "feature_group": model_cfg.get("kind", "unknown"),
        "horizon": 1,
        "forecast_origin_policy": "end_of_month_M",
        "feature_table_path": feature_table_path,
    }

    # Capture a per-iteration learning curve if the model exposes one
    # (LightGBM family via `eval_history`, MLP via `loss_curve_`). Models
    # without epochs return None — those runs just have a single final
    # metric point, which is the honest "this model converges in one shot"
    # representation.
    learning_curve = _extract_learning_curve(model)

    mlflow_run_id: str | None = None
    with maybe_mlflow_run(
        enable=log_to_mlflow,
        run_name=f"{version}_{model_name}",
        tags={
            "version": version,
            "kind": model_cfg.get("kind"),
            "smoke_test": "v1_v5",
            "evaluation_mode": eval_mode,
            "has_learning_curve": "true" if learning_curve else "false",
        },
    ) as run:
        run.log_params(params)

        # Log per-iteration learning curve as stepped metrics so MLflow's
        # Metrics tab renders the convergence line. Curve keys keep
        # their natural name (e.g. `rmse`); FINAL test-set metrics get
        # a `_test` suffix to avoid collision at step=0 with the curve's
        # first iteration.
        if learning_curve:
            for metric_name, values in learning_curve.items():
                for step, value in enumerate(values):
                    try:
                        v = float(value)
                    except (TypeError, ValueError):
                        continue
                    if v != v:  # NaN check
                        continue
                    run.log_metric(metric_name, v, step=step)
            run.log_params({
                "learning_curve_metrics": ",".join(sorted(learning_curve.keys())),
                "learning_curve_n_iters": max(len(v) for v in learning_curve.values()),
            })
            # Final holdout metrics under `_test` suffix — coexists
            # cleanly with the curve, so the user can see both
            # "convergence during training" and "final score" without
            # MLflow's last-write-wins overwriting either.
            run.log_metrics({f"{k}_test": v for k, v in metrics.items()})
        else:
            # Non-iterative model — single final point at default step.
            run.log_metrics(metrics)

        if preds_path.exists():
            run.log_artifact(preds_path)
        run.log_artifact(metrics_path)
        if model_path and model_path.exists():
            run.log_artifact(model_path)
        mlflow_run_id = run.run_id

    write_run_summary(
        artifact_dir,
        params=params,
        metrics=metrics,
        tags={"version": version, "kind": model_cfg.get("kind"), "evaluation_mode": eval_mode},
        mlflow_run_id=mlflow_run_id,
    )

    # Registry record — written even when MLflow is off
    promotion_status = (
        "latest_candidate" if version == "v5" else "historical_backtest"
    )
    record = RegistryRecord(
        model_name=f"smp_monthly_mainland_{model_name}",
        version=version,
        mlflow_run_id=mlflow_run_id,
        data_cutoff_month=str(cutoff.date()),
        evaluation_mode=eval_mode,
        metrics=metrics,
        artifact_uri=str(artifact_dir),
        created_at=datetime.now(timezone.utc).isoformat(),
        promotion_status=promotion_status,
        extra={
            "test_start": str(test_start.date()) if test_start is not None else None,
            "test_end": str(test_end.date()) if test_end is not None else None,
            "n_train": n_train,
            "n_test": n_test,
        },
    )
    append_registry_record(record)

    return VersionResult(
        version=version,
        model_name=model_name,
        data_cutoff_month=str(cutoff.date()),
        train_end=str(train_end.date()),
        test_start=str(test_start.date()) if test_start is not None else None,
        test_end=str(test_end.date()) if test_end is not None else None,
        evaluation_mode=eval_mode,
        n_train=n_train,
        n_test=n_test,
        metrics=metrics,
        mlflow_run_id=mlflow_run_id,
        artifact_dir=str(artifact_dir),
    )


def _record_skip(
    *, version: str, model_name: str, cutoff: pd.Timestamp,
    train_end: pd.Timestamp, test_start, test_end, eval_mode: str,
    reason: str, artifact_dir: Path, log_to_mlflow: bool,
    feature_table_path: str, n_train: int,
) -> VersionResult:
    """Skip path that still logs an MLflow run (with skipped=true tag) so
    the version appears in the comparison report."""
    skip_payload = {
        "version": version, "model_name": model_name,
        "data_cutoff_month": str(cutoff.date()),
        "skip_reason": reason, "n_train_pool": n_train,
    }
    (artifact_dir / "skip_reason.json").write_text(
        json.dumps(skip_payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    mlflow_run_id: str | None = None
    with maybe_mlflow_run(
        enable=log_to_mlflow,
        run_name=f"{version}_{model_name}_SKIPPED",
        tags={
            "version": version, "smoke_test": "v1_v5",
            "skipped": "true", "skip_reason": reason,
            "evaluation_mode": eval_mode,
        },
    ) as run:
        run.log_params({
            "version": version,
            "model_name": model_name,
            "data_cutoff_month": str(cutoff.date()),
            "evaluation_mode": eval_mode,
            "feature_table_path": feature_table_path,
            "retrain_trigger": "manual_cli_smoke_test",
            "skipped": True,
        })
        mlflow_run_id = run.run_id

    record = RegistryRecord(
        model_name=f"smp_monthly_mainland_{model_name}",
        version=version,
        mlflow_run_id=mlflow_run_id,
        data_cutoff_month=str(cutoff.date()),
        evaluation_mode=eval_mode,
        metrics={},
        artifact_uri=str(artifact_dir),
        created_at=datetime.now(timezone.utc).isoformat(),
        promotion_status="skipped",
        extra={"skip_reason": reason, "n_train_pool": n_train},
    )
    append_registry_record(record)

    return VersionResult(
        version=version, model_name=model_name,
        data_cutoff_month=str(cutoff.date()),
        train_end=str(train_end.date()),
        test_start=str(test_start.date()) if test_start is not None else None,
        test_end=str(test_end.date()) if test_end is not None else None,
        evaluation_mode=eval_mode,
        n_train=n_train, n_test=0,
        metrics={},
        mlflow_run_id=mlflow_run_id,
        artifact_dir=str(artifact_dir),
        skipped=True, skip_reason=reason,
    )


# ---------------------------------------------------------------------------
# Report writers
# ---------------------------------------------------------------------------


def _build_overlay_chart_artifacts(
    *, results: list[VersionResult], out_dir: Path,
) -> list[Path]:
    """Generate per-metric overlay line charts (one PNG + one HTML each).

    Each chart plots all models as separate lines on a shared x-axis
    (version step 1..N), so the user can compare every model's
    trajectory across v1..v5 on a single visual. Generated as both:

      * **PNG** (matplotlib) — stable, screenshot-friendly, renders in
        MLflow's artifact preview, good for presentations.
      * **HTML** (plotly) — interactive (hover for values, click legend
        to toggle a series on/off), good for the dashboard / exploration.

    Skipped (version, model) pairs are excluded so the line for that
    model just has a gap at that step rather than a NaN spike.

    Returns the list of generated file paths.
    """
    rows: list[dict[str, Any]] = []
    for r in results:
        if r.skipped or not r.metrics:
            continue
        step = int(r.version.lstrip("v"))
        for metric_name, value in r.metrics.items():
            rows.append({
                "version": r.version, "step": step,
                "model": r.model_name, "metric": metric_name,
                "value": float(value),
            })
    if not rows:
        return []

    df = pd.DataFrame(rows)
    out_dir.mkdir(parents=True, exist_ok=True)
    saved: list[Path] = []

    # Defer matplotlib + plotly imports until we actually have rows to
    # plot — keeps the smoke-test import cost flat when MLflow logging
    # is off (the common test-runner case).
    import matplotlib
    matplotlib.use("Agg")  # headless — no GUI backend even in interactive shells
    import matplotlib.pyplot as plt
    try:
        import plotly.graph_objects as go
        _HAS_PLOTLY = True
    except ImportError:
        _HAS_PLOTLY = False

    for metric_name in sorted(df["metric"].unique()):
        sub = df[df["metric"] == metric_name]
        models_sorted = sorted(sub["model"].unique())

        # ---- matplotlib PNG ----
        fig, ax = plt.subplots(figsize=(9, 5))
        for model_name in models_sorted:
            m_data = sub[sub["model"] == model_name].sort_values("step")
            ax.plot(
                m_data["step"], m_data["value"],
                marker="o", label=model_name,
            )
        ax.set_title(f"{metric_name.upper()} by version — all models overlaid")
        ax.set_xlabel("Version step (1=v1 … 5=v5)")
        ax.set_ylabel(metric_name)
        ax.set_xticks(sorted(sub["step"].unique()))
        ax.legend(loc="best", fontsize=8, ncol=2)
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        png_path = out_dir / f"overlay_{metric_name}.png"
        fig.savefig(png_path, dpi=120)
        plt.close(fig)
        saved.append(png_path)

        # ---- plotly HTML (interactive) ----
        if _HAS_PLOTLY:
            fig = go.Figure()
            for model_name in models_sorted:
                m_data = sub[sub["model"] == model_name].sort_values("step")
                fig.add_trace(go.Scatter(
                    x=m_data["version"], y=m_data["value"],
                    mode="lines+markers", name=model_name,
                ))
            fig.update_layout(
                title=f"{metric_name.upper()} by version — all models overlaid",
                xaxis_title="version (data_cutoff retraining stage)",
                yaxis_title=metric_name,
                hovermode="x unified",
                legend_title="model",
            )
            html_path = out_dir / f"overlay_{metric_name}.html"
            fig.write_html(str(html_path))
            saved.append(html_path)

    return saved


def _log_per_model_summary_runs(
    *, results: list[VersionResult], log_to_mlflow: bool,
) -> dict[str, str]:
    """N runs (one per model). Each logs a clean metric series
    (``mae``, ``rmse``, ``mape``, ``r2``, ``directional_accuracy``) at
    step=1..5 corresponding to v1..v5.

    Why this shape (over the prior single-overlay-run with ``mae__<model>``
    keys): MLflow's Compare-runs view auto-colour-codes by RUN, not by
    metric-key family. Multiple keys in one run share the default
    palette colour and become visually indistinguishable. Per-model
    runs let MLflow assign a distinct colour per model line out of the
    box — the canonical "one run per training configuration" pattern.

    Run naming: ``summary_<model_name>`` so the Runs table can filter
    by tag ``summary=true`` AND by `model_name=…`.

    Returns ``{model_name: run_id}``.
    """
    summary_ids: dict[str, str] = {}
    if not log_to_mlflow:
        return summary_ids

    by_model: dict[str, list[VersionResult]] = {}
    for r in results:
        by_model.setdefault(r.model_name, []).append(r)

    for model_name, runs in by_model.items():
        runs_sorted = sorted(runs, key=lambda x: int(x.version.lstrip("v")))
        with maybe_mlflow_run(
            enable=True,
            run_name=f"summary_{model_name}",
            tags={
                "summary": "true",
                "kind": "per_model",
                "model_name": model_name,
                "smoke_test": "v1_v5",
                "view_hint": (
                    "To overlay every model on one chart: filter Runs by "
                    "`tags.summary = 'true' AND tags.kind = 'per_model'`, "
                    "select all, click Compare → Metric history. MLflow "
                    "auto-colour-codes by run."
                ),
            },
        ) as run:
            run.log_params({
                "model_name": model_name,
                "summary_kind": "per_model",
                "n_versions": len(runs_sorted),
                "versions": ",".join(r.version for r in runs_sorted),
            })
            for r in runs_sorted:
                if r.skipped or not r.metrics:
                    continue
                step = int(r.version.lstrip("v"))
                # Clean metric names — `mae`, `rmse`, etc. NOT `mae__ridge`.
                # MLflow then plots one series per metric per run, with
                # the RUN providing the colour distinction across models.
                run.log_metrics(r.metrics, step=step)
            summary_ids[model_name] = run.run_id

    return summary_ids


def _log_overview_summary_run(
    *, results: list[VersionResult], log_to_mlflow: bool,
) -> str | None:
    """ONE summary run with pre-rendered overlay chart artifacts attached.

    No metrics logged (no Compare gymnastics needed) — just a clean
    bucket for the matplotlib PNGs + plotly HTMLs the user can open
    directly in the Artifacts tab. The per-model summary runs (above)
    cover the interactive-comparison path; this run covers the
    one-click "give me the screenshot" path.
    """
    if not log_to_mlflow:
        return None

    with maybe_mlflow_run(
        enable=True,
        run_name="summary_overview_v1_v5",
        tags={
            "summary": "true",
            "kind": "overview",
            "smoke_test": "v1_v5",
            "view_hint": (
                "Open Artifacts tab → overlay_charts/ → click any "
                "`overlay_<metric>.png` for a one-shot screenshot, or the "
                "matching `.html` for an interactive Plotly chart."
            ),
        },
    ) as run:
        run.log_params({
            "summary_kind": "overview_artifacts_only",
            "n_models": len({r.model_name for r in results}),
            "n_versions": len({r.version for r in results}),
        })

        import tempfile
        chart_dir = Path(tempfile.mkdtemp(prefix="mlops_overlay_charts_"))
        try:
            chart_paths = _build_overlay_chart_artifacts(
                results=results, out_dir=chart_dir,
            )
            for p in chart_paths:
                run.log_artifact(p, artifact_path="overlay_charts")
        except Exception as exc:
            # Don't let a charting failure take down the smoke test —
            # the per-model runs above are the load-bearing path.
            log.warning("Could not generate overlay charts: %s", exc)

        return run.run_id


def build_comparison_dataframe(results: list[VersionResult]) -> pd.DataFrame:
    rows = []
    for r in results:
        row = {
            "version": r.version,
            "model": r.model_name,
            "data_cutoff_month": r.data_cutoff_month,
            "train_end": r.train_end,
            "test_start": r.test_start,
            "test_end": r.test_end,
            "evaluation_mode": r.evaluation_mode,
            "n_train": r.n_train,
            "n_test": r.n_test,
            "mlflow_run_id": r.mlflow_run_id,
            "skipped": r.skipped,
            "skip_reason": r.skip_reason,
        }
        for mk in ("mae", "rmse", "mape", "r2", "directional_accuracy"):
            row[mk] = r.metrics.get(mk)
        rows.append(row)
    return pd.DataFrame(rows)


def _annotate_recommendations(
    df: pd.DataFrame, *, primary_metric: str, lower_is_better: bool,
    historical_versions: list[str], latest_version: str,
) -> pd.DataFrame:
    df = df.copy()
    df["registry_status"] = "historical_backtest"
    df.loc[df["version"] == latest_version, "registry_status"] = "latest_candidate"
    df.loc[df["skipped"] == True, "registry_status"] = "skipped"  # noqa: E712

    # Persistence baseline MAE per version, for "improvement_vs_persistence"
    baseline = (
        df[df["model"] == "persistence_monthly"]
        .set_index("version")[primary_metric]
        .to_dict()
    )
    df["persistence_baseline_mae"] = df["version"].map(baseline)
    df["improvement_vs_persistence"] = (
        df["persistence_baseline_mae"] - df[primary_metric]
    )

    # "recommended_historical" = best among non-skipped historical rows
    hist = df[(df["version"].isin(historical_versions)) & (df["skipped"] != True)]  # noqa: E712
    if not hist.empty and primary_metric in hist.columns:
        idx = (
            hist[primary_metric].idxmin() if lower_is_better
            else hist[primary_metric].idxmax()
        )
        df.loc[idx, "registry_status"] = "recommended_historical"
    return df


def write_reports(
    df: pd.DataFrame,
    *,
    report_md: Path,
    report_json: Path,
    comparison_csv: Path,
) -> None:
    report_md.parent.mkdir(parents=True, exist_ok=True)
    report_json.parent.mkdir(parents=True, exist_ok=True)
    comparison_csv.parent.mkdir(parents=True, exist_ok=True)

    df.to_csv(comparison_csv, index=False)

    # JSON: list of records keeps it dashboard-friendly
    report_json.write_text(
        json.dumps(
            {"generated_at": datetime.now(timezone.utc).isoformat(),
             "rows": df.to_dict(orient="records")},
            indent=2, ensure_ascii=False, default=str,
        ),
        encoding="utf-8",
    )

    # Markdown: a single sortable table + a short text summary
    lines = [
        "# MLOps smoke-test report (v1..v5)",
        "",
        f"_generated_at: {datetime.now(timezone.utc).isoformat()}_",
        "",
        "## Per-(version, model) results",
        "",
    ]
    md_cols = [
        "version", "model", "data_cutoff_month", "evaluation_mode",
        "n_train", "n_test", "mae", "rmse", "mape", "r2",
        "improvement_vs_persistence", "registry_status",
    ]
    lines.append("| " + " | ".join(md_cols) + " |")
    lines.append("| " + " | ".join(["---"] * len(md_cols)) + " |")
    for _, row in df.iterrows():
        cells = []
        for c in md_cols:
            v = row.get(c)
            if isinstance(v, float):
                cells.append(f"{v:.3f}" if np.isfinite(v) else "—")
            elif v is None or (isinstance(v, float) and np.isnan(v)):
                cells.append("—")
            else:
                cells.append(str(v))
        lines.append("| " + " | ".join(cells) + " |")

    # Short summary
    lines.extend(["", "## Summary", ""])
    rec = df[df["registry_status"] == "recommended_historical"]
    if not rec.empty:
        r = rec.iloc[0]
        lines.append(
            f"- **Best historical model** (recommended_historical): "
            f"`{r['model']}` from {r['version']} "
            f"(cutoff {r['data_cutoff_month']}) — MAE {r['mae']:.3f}"
        )
    latest = df[df["registry_status"] == "latest_candidate"]
    if not latest.empty:
        lines.append(
            f"- **Latest candidate**: {len(latest)} models trained at cutoff "
            f"{latest.iloc[0]['data_cutoff_month']} "
            "(no future labels → rolling-validation only). Promotion to "
            "production is intentionally manual."
        )
    skipped = df[df["skipped"] == True]  # noqa: E712
    if not skipped.empty:
        lines.append(
            f"- **{len(skipped)} controlled skips** "
            f"(insufficient data after cutoff filter — see `skip_reason` column)."
        )

    report_md.write_text("\n".join(lines), encoding="utf-8")


# ---------------------------------------------------------------------------
# Top-level runner
# ---------------------------------------------------------------------------


def run_smoke_test(
    *,
    config_path: Path,
    log_to_mlflow: bool = False,
    only_versions: list[str] | None = None,
    only_models: list[str] | None = None,
) -> pd.DataFrame:
    import time
    from src.utils.discord import discord_enabled, send_discord_message

    cfg = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    target_cfg = cfg["target"]
    feature_table_path = target_cfg["feature_table"]
    target_col = target_cfg["target_column"]
    horizon_months = int(target_cfg.get("horizon_months", 1))
    panel = pd.read_parquet(feature_table_path)

    versions = cfg["versions"]
    models = cfg["models"]
    if only_versions:
        versions = [v for v in versions if v["name"] in set(only_versions)]
    if only_models:
        models = [m for m in models if m["name"] in set(only_models)]

    # Discord notify: start. Wrapped in try so a webhook failure can't
    # block training. discord_enabled() short-circuits when no URL set.
    started_at = time.time()
    n_runs_expected = len(versions) * len(models)
    if discord_enabled():
        send_discord_message(
            embeds=[{
                "title": "🚀 MLOps smoke test started",
                "description": (
                    f"**{len(versions)} versions** × **{len(models)} models** "
                    f"= {n_runs_expected} runs\n"
                    f"`{config_path.name}` → MLflow: "
                    f"{'✅ enabled' if log_to_mlflow else '⚪ disabled'}"
                ),
                "color": 0x3b82f6,
                "fields": [
                    {"name": "versions", "value": ", ".join(v["name"] for v in versions), "inline": True},
                    {"name": "models", "value": ", ".join(m["name"] for m in models)[:1024], "inline": False},
                ],
            }]
        )

    artifact_root = Path(cfg["outputs"]["artifact_root"])
    results: list[VersionResult] = []
    try:
        for v in versions:
            # Inherit area from target if not version-specific
            v.setdefault("area", target_cfg.get("area", "mainland"))
            for m in models:
                log.info("=== version=%s model=%s ===", v["name"], m["name"])
                result = _train_one(
                    version_cfg=v, model_cfg=m, panel=panel,
                    target_col=target_col, log_to_mlflow=log_to_mlflow,
                    artifact_root=artifact_root,
                    feature_table_path=feature_table_path,
                    horizon_months=horizon_months,
                )
                results.append(result)
            # End-of-version Discord ping with quick top-3 by primary metric
            if discord_enabled():
                primary = cfg["promotion"]["primary_metric"]
                version_results = [r for r in results if r.version == v["name"]]
                ranked = sorted(
                    [r for r in version_results if r.metrics and primary in r.metrics],
                    key=lambda x: x.metrics[primary],
                )
                top3 = "\n".join(
                    f"  {i + 1}. `{r.model_name}` — {primary}={r.metrics[primary]:.3f}"
                    for i, r in enumerate(ranked[:3])
                )
                send_discord_message(
                    embeds=[{
                        "title": f"✅ {v['name']} complete",
                        "description": (
                            f"{len(version_results)} runs · "
                            f"{sum(1 for r in version_results if r.skipped)} skipped\n"
                            f"**Top by {primary}:**\n{top3 or '(no scored runs)'}"
                        ),
                        "color": 0x10b981,
                    }]
                )
    except Exception as exc:
        if discord_enabled():
            send_discord_message(
                embeds=[{
                    "title": "❌ MLOps smoke test FAILED",
                    "description": (
                        f"After **{len(results)}/{n_runs_expected}** runs.\n"
                        f"`{type(exc).__name__}`: {str(exc)[:500]}"
                    ),
                    "color": 0xef4444,
                }]
            )
        raise

    # Two summary layers on top of the 35 individual (version, model) runs:
    #   1. PER-MODEL — one run per model with clean metric names (`mae`,
    #      `rmse`, …) at step=1..5. MLflow's Compare-runs view auto-
    #      colour-codes by run, so selecting all per-model summary runs
    #      and clicking Compare → Metric history gives a clean overlay
    #      with one colour per model. This is the canonical MLflow shape.
    #   2. OVERVIEW — one run with pre-rendered overlay PNG + interactive
    #      HTML chart artifacts attached. One-click viewing in the
    #      Artifacts tab; no Compare gymnastics required.
    _log_per_model_summary_runs(results=results, log_to_mlflow=log_to_mlflow)
    _log_overview_summary_run(results=results, log_to_mlflow=log_to_mlflow)

    df = build_comparison_dataframe(results)
    df = _annotate_recommendations(
        df,
        primary_metric=cfg["promotion"]["primary_metric"],
        lower_is_better=cfg["promotion"]["lower_is_better"],
        historical_versions=cfg["promotion"]["historical_versions"],
        latest_version=cfg["promotion"]["latest_version"],
    )
    write_reports(
        df,
        report_md=Path(cfg["outputs"]["report_md"]),
        report_json=Path(cfg["outputs"]["report_json"]),
        comparison_csv=Path(cfg["outputs"]["comparison_csv"]),
    )

    # Final Discord ping — overall summary + top model across all versions
    if discord_enabled():
        elapsed = time.time() - started_at
        primary = cfg["promotion"]["primary_metric"]
        latest_ver = cfg["promotion"]["latest_version"]
        latest_rows = df[df["version"] == latest_ver].sort_values(primary)
        winner_block = "(no scored rows)"
        if not latest_rows.empty:
            top = latest_rows.iloc[0]
            winner_block = (
                f"**{latest_ver} winner:** `{top['model']}` — "
                f"{primary}={top[primary]:.3f}"
            )
        n_skipped = int((df.get("skipped", False) == True).sum()) if "skipped" in df.columns else 0
        send_discord_message(
            embeds=[{
                "title": "🎉 MLOps smoke test complete",
                "description": (
                    f"{len(df)} (version, model) rows in {elapsed / 60:.1f} min\n"
                    f"Skipped: {n_skipped}\n\n{winner_block}"
                ),
                "color": 0x8b5cf6,
            }]
        )
    return df


@app.command()
def main(
    config: Path = typer.Option(
        Path("config/mlops_smoke_test.yaml"),
        "--config", help="Smoke-test YAML config.",
    ),
    log_to_mlflow: bool = typer.Option(
        False, "--log-to-mlflow",
        help="Log every (version, model) run to MLflow. Requires the MLflow "
             "server to be reachable (e.g. `docker compose up -d`).",
    ),
    only_versions: str | None = typer.Option(
        None, "--only-versions",
        help="Comma-separated subset, e.g. 'v1,v2'.",
    ),
    only_models: str | None = typer.Option(
        None, "--only-models",
        help="Comma-separated subset of model names.",
    ),
) -> None:
    """Run the v1..v5 retraining smoke test end-to-end."""
    df = run_smoke_test(
        config_path=config,
        log_to_mlflow=log_to_mlflow,
        only_versions=only_versions.split(",") if only_versions else None,
        only_models=only_models.split(",") if only_models else None,
    )
    typer.echo(f"\nDone. {len(df)} (version, model) rows in the comparison table.")
    typer.echo(f"Versions covered: {sorted(df['version'].unique().tolist())}")
    typer.echo(f"Skipped: {int(df['skipped'].sum())} run(s).")


if __name__ == "__main__":
    app()
