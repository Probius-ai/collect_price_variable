"""Tests for the v1..v5 MLOps staged-retraining smoke test.

The 11 canaries here are the user's MLOps spec verbatim — each test
name maps 1:1 to the requirements list. They exercise:

  * the leak-safety contract (vN does not use data after its cutoff)
  * structural invariants of the report (five versions, distinct
    cutoffs, no auto-production-promotion)
  * the rolling-validation fallback for v5 (no fabricated future labels)
  * the dual MLflow/registry-JSON logging surface

Heavy models (LightGBM, MLP) are excluded from the test runs via
``only_models`` to keep CI fast — the smoke-test code path is identical
regardless of which models we include.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest
import yaml

from src.pipelines.mlops_smoke_test import (
    filter_to_cutoff,
    run_smoke_test,
)


# Fast, leak-safe model subset for the test runs.
_FAST_MODELS = ["persistence_monthly", "ridge"]


# ---------------------------------------------------------------------------
# Shared fixtures — synthetic monthly panel + minimal config
# ---------------------------------------------------------------------------


@pytest.fixture
def synthetic_panel(tmp_path: Path) -> Path:
    """A synthetic monthly SMP feature table spanning 2018-01..2026-03 with
    the columns Ridge/PersistenceMonthly need.

    Real data is too large to bundle into tests and would couple the
    smoke-test contract to feature-builder changes. Synthetic panel has
    the same column shape (period_month, target, smp_t_observed, lags,
    rolling means, calendar) and a predictable signal so we can sanity-
    check the model output too.
    """
    import numpy as np
    rng = np.random.default_rng(7)
    months = pd.date_range("2018-01-01", "2026-03-01", freq="MS")
    n = len(months)
    smp = 80 + 30 * np.sin(2 * np.pi * np.arange(n) / 12) + rng.normal(0, 5, n)
    df = pd.DataFrame({
        "period_month": months,
        "area": "mainland",
        "smp_t_observed": smp,
        "smp_lag_1m": np.r_[np.nan, smp[:-1]],
        "smp_lag_2m": np.r_[np.nan, np.nan, smp[:-2]],
        "smp_lag_3m": np.r_[[np.nan]*3, smp[:-3]],
        "smp_lag_6m": np.r_[[np.nan]*6, smp[:-6]],
        "smp_lag_12m": np.r_[[np.nan]*12, smp[:-12]],
        "smp_rolling_3m_mean":  pd.Series(smp).rolling(3).mean().values,
        "smp_rolling_6m_mean":  pd.Series(smp).rolling(6).mean().values,
        "smp_rolling_12m_mean": pd.Series(smp).rolling(12).mean().values,
        "smp_rolling_12m_std":  pd.Series(smp).rolling(12).std().values,
        "month_sin": np.sin(2*np.pi*months.month/12),
        "month_cos": np.cos(2*np.pi*months.month/12),
        "quarter":   ((months.month - 1)//3 + 1).astype(float),
        "target_smp_t_plus_h_months": np.r_[smp[1:], np.nan],
        "forecast_origin_month": months,
        "target_month": months + pd.DateOffset(months=1),
        "information_cutoff": months,
    })
    path = tmp_path / "synthetic_smp_monthly.parquet"
    df.to_parquet(path, index=False)
    return path


@pytest.fixture
def smoke_config(tmp_path: Path, synthetic_panel: Path) -> Path:
    """Smoke-test config pointing at the synthetic panel and writing
    every output under tmp_path."""
    cfg = {
        "target": {
            "feature_table": str(synthetic_panel),
            "target_column": "target_smp_t_plus_h_months",
            "area": "mainland",
            "horizon_months": 1,
        },
        "versions": [
            {"name": "v1", "data_cutoff_month": "2021-12-01",
             "train_end": "2021-12-01", "test_start": "2022-01-01",
             "test_end": "2022-12-01", "evaluation_mode": "fixed_holdout"},
            {"name": "v2", "data_cutoff_month": "2022-12-01",
             "train_end": "2022-12-01", "test_start": "2023-01-01",
             "test_end": "2023-12-01", "evaluation_mode": "fixed_holdout"},
            {"name": "v3", "data_cutoff_month": "2023-12-01",
             "train_end": "2023-12-01", "test_start": "2024-01-01",
             "test_end": "2024-12-01", "evaluation_mode": "fixed_holdout"},
            {"name": "v4", "data_cutoff_month": "2024-12-01",
             "train_end": "2024-12-01", "test_start": "2025-01-01",
             "test_end": "2025-08-01", "evaluation_mode": "fixed_holdout"},
            {"name": "v5", "data_cutoff_month": "2025-08-01",
             "train_end": "2025-08-01", "evaluation_mode":
                 "latest_rolling_validation", "rolling_window_months": 12},
        ],
        "models": [
            {"name": "persistence_monthly", "kind": "baseline", "min_train_rows": 24},
            {"name": "ridge",               "kind": "linear",   "min_train_rows": 24},
        ],
        "promotion": {
            "primary_metric": "mae",
            "lower_is_better": True,
            "historical_versions": ["v1", "v2", "v3", "v4"],
            "latest_version": "v5",
        },
        "outputs": {
            "report_md":      str(tmp_path / "report.md"),
            "report_json":    str(tmp_path / "report.json"),
            "comparison_csv": str(tmp_path / "comparison.csv"),
            "artifact_root":  str(tmp_path / "artifacts"),
        },
    }
    cfg_path = tmp_path / "smoke.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    return cfg_path


@pytest.fixture
def smoke_results(smoke_config: Path, tmp_path: Path, monkeypatch) -> pd.DataFrame:
    """Run the smoke test once per test session and reuse the DataFrame.

    Registry writes go under a tmp directory so canaries can read the
    JSON fallback without colliding with other tests.
    """
    monkeypatch.setattr(
        "src.tracking.mlflow_utils.REGISTRY_DIR",
        tmp_path / "registry",
    )
    monkeypatch.setattr(
        "src.pipelines.mlops_smoke_test.REGISTRY_DIR",
        tmp_path / "registry",
        raising=False,
    )
    return run_smoke_test(config_path=smoke_config, log_to_mlflow=False)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_mlops_smoke_test_creates_five_versions(smoke_results: pd.DataFrame):
    versions = sorted(smoke_results["version"].unique().tolist())
    assert versions == ["v1", "v2", "v3", "v4", "v5"], versions


def test_each_version_has_distinct_data_cutoff_month(smoke_results: pd.DataFrame):
    cutoffs = (
        smoke_results.groupby("version")["data_cutoff_month"]
        .first()
        .to_dict()
    )
    distinct = set(cutoffs.values())
    assert len(distinct) == 5, f"Expected 5 distinct cutoffs, got {cutoffs}"


# --- Leak-safety canaries: vN does not see data after its cutoff -----------


def _max_train_month_for_version(panel_path: Path, cutoff: str) -> pd.Timestamp:
    df = pd.read_parquet(panel_path)
    filtered = filter_to_cutoff(df, data_cutoff_month=pd.Timestamp(cutoff))
    return pd.to_datetime(filtered["period_month"]).max()


def test_v1_does_not_use_data_after_2021_12(synthetic_panel: Path):
    assert _max_train_month_for_version(
        synthetic_panel, "2021-12-01"
    ) == pd.Timestamp("2021-12-01")


def test_v2_does_not_use_data_after_2022_12(synthetic_panel: Path):
    assert _max_train_month_for_version(
        synthetic_panel, "2022-12-01"
    ) == pd.Timestamp("2022-12-01")


def test_v3_does_not_use_data_after_2023_12(synthetic_panel: Path):
    assert _max_train_month_for_version(
        synthetic_panel, "2023-12-01"
    ) == pd.Timestamp("2023-12-01")


def test_v4_does_not_use_data_after_2024_12(synthetic_panel: Path):
    assert _max_train_month_for_version(
        synthetic_panel, "2024-12-01"
    ) == pd.Timestamp("2024-12-01")


def test_v5_uses_latest_rolling_validation_when_no_future_holdout(
    smoke_results: pd.DataFrame,
):
    v5 = smoke_results[smoke_results["version"] == "v5"]
    assert not v5.empty
    # Every v5 row uses the rolling-validation mode, never fixed_holdout
    assert set(v5["evaluation_mode"].unique()) == {"latest_rolling_validation"}, (
        v5["evaluation_mode"].unique()
    )
    # No test_start/test_end on v5 (we filter to '' or None — JSON nulls)
    for col in ("test_start", "test_end"):
        assert v5[col].isna().all() or (v5[col] == "None").all() or (v5[col] == "").all()


def _make_results_fixture():
    """Two models × three versions. v3 of ridge is skipped. Shared by
    the per-model and overview canaries."""
    from src.pipelines.mlops_smoke_test import VersionResult
    return [
        VersionResult(
            version="v1", model_name="ridge", data_cutoff_month="2021-12-01",
            train_end="2021-12-01", test_start="2022-01-01", test_end="2022-12-01",
            evaluation_mode="fixed_holdout", n_train=10, n_test=12,
            metrics={"mae": 20.0, "rmse": 25.0}, mlflow_run_id=None,
            artifact_dir=None, skipped=False,
        ),
        VersionResult(
            version="v2", model_name="ridge", data_cutoff_month="2022-12-01",
            train_end="2022-12-01", test_start="2023-01-01", test_end="2023-12-01",
            evaluation_mode="fixed_holdout", n_train=22, n_test=12,
            metrics={"mae": 15.0, "rmse": 19.0}, mlflow_run_id=None,
            artifact_dir=None, skipped=False,
        ),
        VersionResult(
            version="v3", model_name="ridge", data_cutoff_month="2023-12-01",
            train_end="2023-12-01", test_start="2024-01-01", test_end="2024-12-01",
            evaluation_mode="fixed_holdout", n_train=34, n_test=12,
            metrics={}, mlflow_run_id=None, artifact_dir=None,
            skipped=True, skip_reason="synthetic skip",
        ),
        VersionResult(
            version="v1", model_name="persistence_monthly",
            data_cutoff_month="2021-12-01", train_end="2021-12-01",
            test_start="2022-01-01", test_end="2022-12-01",
            evaluation_mode="fixed_holdout", n_train=10, n_test=12,
            metrics={"mae": 26.0, "rmse": 31.0}, mlflow_run_id=None,
            artifact_dir=None, skipped=False,
        ),
        VersionResult(
            version="v2", model_name="persistence_monthly",
            data_cutoff_month="2022-12-01", train_end="2022-12-01",
            test_start="2023-01-01", test_end="2023-12-01",
            evaluation_mode="fixed_holdout", n_train=22, n_test=12,
            metrics={"mae": 18.0, "rmse": 22.0}, mlflow_run_id=None,
            artifact_dir=None, skipped=False,
        ),
    ]


def _patch_mlflow_run():
    """Return (recorders_list, context_manager_replacement) so a test
    can capture every log_* call without a live MLflow server."""
    from contextlib import contextmanager

    class _RecorderRun:
        def __init__(self):
            self.run_id = f"fake-id-{id(self)}"
            self.params: dict = {}
            self.metrics_calls: list[tuple[dict, int | None]] = []
            self.artifacts: list[tuple[str, str | None]] = []
            self.tags: dict = {}
        def log_params(self, p): self.params.update(p)
        def log_metrics(self, m, step=None): self.metrics_calls.append((dict(m), step))
        def log_metric(self, k, v, step=None): self.metrics_calls.append(({k: v}, step))
        def set_tag(self, k, v): self.tags[k] = v
        def set_tags(self, t): self.tags.update(t)
        def log_artifact(self, path, artifact_path=None):
            self.artifacts.append((str(path), artifact_path))

    recorders: list[_RecorderRun] = []

    @contextmanager
    def _fake_run(*, enable=None, run_name=None, tags=None, nested=False):
        r = _RecorderRun()
        if tags:
            r.tags.update(tags)
        r.run_name = run_name
        recorders.append(r)
        yield r

    return recorders, _fake_run


def test_per_model_summary_runs_emit_clean_metric_keys():
    """The canonical MLflow shape: one run per model, with metric keys
    `mae`/`rmse`/… (NOT prefixed by the model name).

    Why this matters: MLflow's Compare-runs view auto-colour-codes by
    RUN. Multiple metric keys in a single run share the default palette
    colour and become visually indistinguishable. Per-model runs let
    MLflow assign one distinct colour per model line out of the box.
    """
    from unittest.mock import patch
    from src.pipelines.mlops_smoke_test import _log_per_model_summary_runs

    results = _make_results_fixture()
    recorders, fake_run = _patch_mlflow_run()

    with patch("src.pipelines.mlops_smoke_test.maybe_mlflow_run", fake_run):
        ids = _log_per_model_summary_runs(results=results, log_to_mlflow=True)

    # One summary run per distinct model name
    assert len(recorders) == 2, f"expected 2 per-model runs, got {len(recorders)}"
    assert set(ids.keys()) == {"ridge", "persistence_monthly"}

    # Each run's run_name = `summary_<model_name>`
    run_names = {r.run_name for r in recorders}
    assert run_names == {"summary_ridge", "summary_persistence_monthly"}

    # Tags include the model_name so the Compare-runs filter can scope
    for r in recorders:
        assert r.tags.get("summary") == "true"
        assert r.tags.get("kind") == "per_model"
        assert r.tags.get("model_name") in {"ridge", "persistence_monthly"}

    # Metric keys are CLEAN (mae / rmse) — NOT prefixed
    for r in recorders:
        for payload, _step in r.metrics_calls:
            for key in payload:
                assert "__" not in key, (
                    f"metric key {key!r} carries a `__model` suffix; should "
                    "be clean (`mae`, `rmse`, …) so MLflow Compare can "
                    "auto-colour-code by run instead."
                )

    # Skipped v3 of ridge produces no log_metrics call at step=3
    ridge_rec = next(r for r in recorders
                     if r.tags.get("model_name") == "ridge")
    ridge_steps = [step for _, step in ridge_rec.metrics_calls]
    assert 3 not in ridge_steps, (
        f"skipped v3/ridge leaked into per-model summary; steps={ridge_steps}"
    )


def test_per_model_summary_runs_noop_when_logging_disabled():
    """Pure UI helper — no work when MLflow logging is off."""
    from src.pipelines.mlops_smoke_test import _log_per_model_summary_runs
    assert _log_per_model_summary_runs(results=[], log_to_mlflow=False) == {}


def test_overview_summary_run_attaches_overlay_chart_artifacts(tmp_path):
    """The overview run carries NO metrics — just pre-rendered overlay
    chart artifacts (PNG + HTML) under `overlay_charts/`. Gives the
    user a one-click viewing path without needing the Compare gesture.
    """
    from unittest.mock import patch
    from src.pipelines.mlops_smoke_test import _log_overview_summary_run

    results = _make_results_fixture()
    recorders, fake_run = _patch_mlflow_run()

    with patch("src.pipelines.mlops_smoke_test.maybe_mlflow_run", fake_run):
        _log_overview_summary_run(results=results, log_to_mlflow=True)

    assert len(recorders) == 1, "overview must emit exactly one run"
    rec = recorders[0]
    assert rec.run_name == "summary_overview_v1_v5"
    assert rec.tags.get("summary") == "true"
    assert rec.tags.get("kind") == "overview"

    # No metric logging — artifacts only
    assert rec.metrics_calls == [], rec.metrics_calls

    # PNG artifact per metric (we have 2 metrics: mae + rmse) under
    # the `overlay_charts/` subdir. HTML may or may not exist depending
    # on whether plotly is importable — at minimum the PNGs must.
    png_artifacts = [
        a for (a, sub) in rec.artifacts
        if a.endswith(".png") and sub == "overlay_charts"
    ]
    assert len(png_artifacts) >= 2, (
        f"expected >= 2 PNG overlay charts under overlay_charts/, "
        f"got {png_artifacts}"
    )


def test_overview_summary_run_noop_when_logging_disabled():
    from src.pipelines.mlops_smoke_test import _log_overview_summary_run
    assert _log_overview_summary_run(results=[], log_to_mlflow=False) is None


def test_extract_learning_curve_handles_lightgbm_and_non_iterative_models():
    """Pin the contract of `_extract_learning_curve`:

    - LightGBM (fit with X_valid set) → returns `{metric_name: [values]}`
      pulled from `eval_history['valid']`. This is the per-iteration
      learning curve MLflow needs to render as a line chart instead of
      a single point.
    - Models without an epoch concept (Ridge, PersistenceMonthly) →
      returns None. We don't fabricate a curve for them.
    """
    import numpy as np
    import pandas as pd
    from src.pipelines.mlops_smoke_test import _extract_learning_curve
    from src.models.lightgbm_model import LightGBMModel
    from src.models.ridge_model import RidgeModel
    from src.models.naive import PersistenceMonthly

    # Synthetic monthly panel — enough rows for LightGBM to actually fit
    rng = np.random.default_rng(0)
    n = 60
    X_train = pd.DataFrame({
        "smp_t_observed": rng.normal(100, 10, n),
        "smp_lag_1m":     rng.normal(100, 10, n),
        "smp_lag_2m":     rng.normal(100, 10, n),
    })
    y_train = pd.Series(rng.normal(100, 10, n))
    X_valid = X_train.iloc[:12].copy()
    y_valid = y_train.iloc[:12].copy()

    # LightGBM with validation set → eval_history populated → curve returned
    lgb_model = LightGBMModel(num_boost_round=20, early_stopping_rounds=50)
    lgb_model.fit(X_train, y_train, X_valid=X_valid, y_valid=y_valid)
    curve = _extract_learning_curve(lgb_model)
    assert curve is not None, "LightGBM with X_valid must return a curve"
    assert isinstance(curve, dict)
    # rmse is the default `metric` param — should be a list ≥ 1 long
    assert any(len(v) >= 1 for v in curve.values()), curve

    # Ridge — no epochs, no curve
    ridge = RidgeModel().fit(X_train, y_train)
    assert _extract_learning_curve(ridge) is None, (
        "Ridge converges in one shot — must not fabricate a fake curve."
    )

    # Persistence — pure lookup, no fit work
    persistence = PersistenceMonthly().fit(X_train, y_train)
    assert _extract_learning_curve(persistence) is None


def test_v5_rolling_validation_never_scores_target_beyond_cutoff(
    synthetic_panel: Path, tmp_path: Path,
):
    """v5's spec is explicit: do not fabricate future labels. The rolling
    eval window's LATEST fold must have target month <= cutoff. Earlier
    versions of the rolling code stopped the window AT the cutoff itself,
    which scored against `SMP(cutoff + horizon)` — a label that doesn't
    exist at the simulated retraining moment.

    This canary calls the rolling helper directly with the synthetic panel
    and asserts every eval fold's target month is <= cutoff.
    """
    from src.pipelines.mlops_smoke_test import _evaluate_latest_rolling_validation
    from src.models.naive import PersistenceMonthly

    df = pd.read_parquet(synthetic_panel)
    cutoff = pd.Timestamp("2025-08-01")
    horizon = 1
    preds, _metrics, _last_fold_model = _evaluate_latest_rolling_validation(
        PersistenceMonthly, df,
        target_col="target_smp_t_plus_h_months",
        data_cutoff_month=cutoff,
        window_months=12,
        horizon_months=horizon,
    )
    assert not preds.empty, "rolling validation returned no folds"
    eval_months = pd.to_datetime(preds["period_month"])
    target_months = eval_months + pd.DateOffset(months=horizon)
    # Critical invariant: every eval fold's TARGET is observable at cutoff
    assert (target_months <= cutoff).all(), (
        f"v5 rolling validation scored against future target(s): "
        f"max target_month = {target_months.max()}, cutoff = {cutoff}. "
        f"This fabricates a future label (SMP({target_months.max():%Y-%m})) "
        f"that doesn't exist at the simulated retraining moment."
    )
    # And the window should be exactly 12 (for synthetic panel that has
    # enough history before 2025-07).
    assert len(preds) == 12, len(preds)
    # Latest eval month is cutoff - horizon = 2025-07
    assert eval_months.max() == pd.Timestamp("2025-07-01"), eval_months.max()


def test_each_version_logs_mlflow_run_or_registry_record(
    smoke_results: pd.DataFrame, tmp_path: Path,
):
    """With log_to_mlflow=False the MLflow run_id is None, but the JSON
    registry still gets a record per (version, model). At least one
    record per version must exist."""
    reg_dir = tmp_path / "registry"
    files = list(reg_dir.glob("smp_monthly_mainland_*_registry.json"))
    assert files, f"No registry files written under {reg_dir}"
    seen_versions: set[str] = set()
    for f in files:
        records = json.loads(f.read_text())["records"]
        for r in records:
            seen_versions.add(r["version"])
    assert seen_versions == {"v1", "v2", "v3", "v4", "v5"}, seen_versions


def test_version_comparison_report_has_five_rows(
    smoke_results: pd.DataFrame, smoke_config: Path,
):
    """The CSV report writer emits one row per (version, model). With
    2 models × 5 versions we expect 10 rows; the per-version count must
    be exactly 5 distinct values."""
    cfg = yaml.safe_load(Path(smoke_config).read_text())
    comparison_csv = Path(cfg["outputs"]["comparison_csv"])
    assert comparison_csv.exists(), f"missing {comparison_csv}"
    df = pd.read_csv(comparison_csv)
    versions = sorted(df["version"].unique().tolist())
    assert len(versions) == 5, versions
    assert set(versions) == {"v1", "v2", "v3", "v4", "v5"}


def test_no_production_auto_promotion(smoke_results: pd.DataFrame, tmp_path: Path):
    """The promotion logic must NEVER auto-tag a model as production /
    staging. The only allowed registry_status values are:
        historical_backtest | latest_candidate | recommended_historical | skipped
    """
    allowed = {
        "historical_backtest",
        "latest_candidate",
        "recommended_historical",
        "skipped",
    }
    statuses = set(smoke_results["registry_status"].unique())
    assert statuses <= allowed, (
        f"unexpected promotion statuses: {statuses - allowed}"
    )
    # And no record in the JSON registry should claim production
    for f in (tmp_path / "registry").glob("*_registry.json"):
        records = json.loads(f.read_text())["records"]
        for r in records:
            assert "production" not in (r.get("promotion_status") or "").lower(), r
            assert "prod" not in (r.get("promotion_status") or "").lower(), r


def test_training_pool_excludes_rows_whose_target_is_after_cutoff(
    synthetic_panel: Path,
):
    """Round-MLops-review fix: a row at ``period_month == cutoff`` has
    target = SMP(cutoff + horizon), which is in the FUTURE relative to
    the simulated retraining moment. Including it would leak future
    labels into the training fit. The strict-correct training filter
    must exclude such rows.
    """
    df = pd.read_parquet(synthetic_panel)
    cutoff = pd.Timestamp("2021-12-01")

    # Strict filter (h=1): row at 2021-12 is dropped because its target
    # is SMP(2022-01) — not observable at end-of-2021-12.
    strict = filter_to_cutoff(df, data_cutoff_month=cutoff, horizon_months=1)
    assert pd.to_datetime(strict["period_month"]).max() == pd.Timestamp("2021-11-01"), (
        "Strict-mode filter must drop the boundary row whose target leaks "
        "into the simulated future."
    )

    # And every row in the strict pool has target_month <= cutoff
    pm = pd.to_datetime(strict["period_month"])
    target_month = pm + pd.DateOffset(months=1)
    assert (target_month <= cutoff).all(), (
        "Strict filter let through rows with future target months."
    )

    # Loose filter (h=0) keeps the boundary row — used for the
    # canary leak-safety tests above and for test-set selection where
    # the labels are observable in retrospect.
    loose = filter_to_cutoff(df, data_cutoff_month=cutoff)
    assert pd.to_datetime(loose["period_month"]).max() == cutoff


def test_strict_pool_actually_used_by_smoke_test_training(
    smoke_results: pd.DataFrame, synthetic_panel: Path,
):
    """End-to-end: confirm the smoke test's reported n_train for each
    historical version equals the count of rows whose TARGET month is
    observable at cutoff — not the count of rows whose period_month is."""
    df = pd.read_parquet(synthetic_panel)
    for vcfg_name, cutoff_str in [
        ("v1", "2021-12-01"), ("v2", "2022-12-01"),
        ("v3", "2023-12-01"), ("v4", "2024-12-01"),
    ]:
        cutoff = pd.Timestamp(cutoff_str)
        strict_pool = filter_to_cutoff(df, data_cutoff_month=cutoff, horizon_months=1)
        expected_n_train = int(strict_pool["target_smp_t_plus_h_months"].notna().sum())
        # Pick any fixed_holdout row for that version (n_train is the same
        # across models in a version since models share the same panel filter)
        actual_n_train = int(
            smoke_results[(smoke_results["version"] == vcfg_name)]["n_train"].max()
        )
        assert actual_n_train == expected_n_train, (
            f"{vcfg_name}: smoke test reported n_train={actual_n_train} "
            f"but strict pool has {expected_n_train} rows. The training "
            f"filter is not using the target-observability bound."
        )


def test_v5_not_marked_as_fixed_holdout_without_future_labels(
    smoke_results: pd.DataFrame, tmp_path: Path,
):
    """v5 must not be tagged as fixed_holdout / historical_backtest — it
    has no future-month test labels, so claiming a forward-holdout score
    would be misleading.
    """
    v5 = smoke_results[smoke_results["version"] == "v5"]
    assert (v5["evaluation_mode"] == "latest_rolling_validation").all()
    # registry_status should be latest_candidate (not historical_backtest)
    statuses = set(v5["registry_status"].unique())
    assert "historical_backtest" not in statuses, statuses
    assert "latest_candidate" in statuses, statuses
    # Registry JSON same check
    for f in (tmp_path / "registry").glob("*_registry.json"):
        for r in json.loads(f.read_text())["records"]:
            if r["version"] == "v5":
                assert r["evaluation_mode"] == "latest_rolling_validation"
                assert r["promotion_status"] != "historical_backtest"
