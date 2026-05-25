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

# Optional models — sklearn MLPRegressor isn't a project model but we
# allow it via lng_forecast.models.MLPForecaster for completeness; if the
# user has that dependency missing or the panel is too small we
# controlled-skip and log the reason as an MLflow tag.
try:
    from src.models.lng_forecast.models import MLPForecaster  # noqa: F401
    _HAS_MLP = True
except Exception:
    _HAS_MLP = False


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
    # The neural baseline. Wired as a controlled-skip in practice — see
    # MLPSkip handling in `_train_one`. The factory is here for symmetry.
    if not _HAS_MLP:
        raise RuntimeError("MLPForecaster not available in this environment")
    return MLPForecaster()


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


def _evaluate_latest_rolling_validation(
    model_factory: Callable[[], Any],
    panel: pd.DataFrame,
    *,
    target_col: str,
    data_cutoff_month: pd.Timestamp,
    window_months: int,
    horizon_months: int = 1,
) -> tuple[pd.DataFrame, dict[str, float]]:
    """Backward walk-forward over the last `window_months` months ending
    at `data_cutoff_month`.

    For each month m in [cutoff - window+1 ... cutoff], fit a fresh model
    on rows whose TARGET is observable strictly before m (i.e.
    period_month + horizon < m), and predict the target at m. This is the
    "production-like" evaluation used when no future holdout labels exist.

    Key leak-safety detail: we filter by ``period_month + horizon < m``,
    NOT ``period_month < m`` — the latter would let a fold's training
    pool include a row whose target IS m or beyond, leaking the eval
    label into training.
    """
    pm = pd.to_datetime(panel["period_month"])
    target_month = pm + pd.DateOffset(months=horizon_months)
    months = pd.date_range(
        end=data_cutoff_month, periods=window_months, freq="MS"
    )
    rows: list[dict[str, Any]] = []
    for m in months:
        train_mask = (target_month < m) & panel[target_col].notna()
        train = panel.loc[train_mask]
        eval_row = panel.loc[(pm == m) & panel[target_col].notna()]
        if eval_row.empty or train.empty:
            continue
        m_model = model_factory()
        X_train = _drop_leakage_columns(train, target_col)
        y_train = train[target_col].astype(float)
        m_model.fit(X_train, y_train)
        X_eval = _drop_leakage_columns(eval_row, target_col)
        y_pred = pd.Series(m_model.predict(X_eval)).iloc[0]
        y_true = float(eval_row[target_col].iloc[0])
        rows.append({"period_month": m, "y_true": y_true, "y_pred": float(y_pred)})

    preds = pd.DataFrame(rows)
    if preds.empty:
        return preds, {}
    metrics = compute_metrics(preds["y_true"], preds["y_pred"])
    return preds, _metrics_to_dict(metrics)


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
            preds, metrics = _evaluate_latest_rolling_validation(
                MODEL_FACTORIES[model_name], panel,
                target_col=target_col,
                data_cutoff_month=cutoff,
                window_months=window,
                horizon_months=horizon_months,
            )
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

    mlflow_run_id: str | None = None
    with maybe_mlflow_run(
        enable=log_to_mlflow,
        run_name=f"{version}_{model_name}",
        tags={
            "version": version,
            "kind": model_cfg.get("kind"),
            "smoke_test": "v1_v5",
            "evaluation_mode": eval_mode,
        },
    ) as run:
        run.log_params(params)
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

    artifact_root = Path(cfg["outputs"]["artifact_root"])
    results: list[VersionResult] = []
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
