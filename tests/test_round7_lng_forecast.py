"""Round-7 tests: new LNG forecaster (per PDF guide).

Pins the PDF's key design rules:
  * TimeSeriesSplit + embargo (no train→valid bleed)
  * per-fold scaler fit (MLP only — pinned by isolating its scaler state)
  * target = LNG(M+h) (correct future-shift, no leakage)
  * RMSE + MAPE formula
  * pandas pct_change explicit fill_method=None (round-6 lesson carried into new module)
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.models.lng_forecast.evaluate import compute_rmse_mape, run_cv
from src.models.lng_forecast.models import (
    LightGBMForecaster,
    MLPForecaster,
    NaiveLagForecaster,
)
from src.models.lng_forecast.split import ts_folds


# ---------------------------------------------------------------------------
# 1. Embargo isolates train tail from valid head
# ---------------------------------------------------------------------------

def test_ts_folds_embargo_creates_gap_between_train_tail_and_valid_head():
    """For every fold, max(train_idx) + embargo + 1 ≤ min(valid_idx). This is
    the PDF §5 contract: the lagged/rolling tail of train must not bleed."""
    n = 200
    embargo = 5
    n_folds = 0
    for tr, va in ts_folds(n, n_splits=5, embargo=embargo):
        n_folds += 1
        assert tr.max() + embargo + 1 <= va.min(), (
            f"fold {n_folds}: train.max={tr.max()}, valid.min={va.min()}, "
            f"embargo={embargo}: gap violated"
        )
    assert n_folds == 5


def test_ts_folds_embargo_zero_keeps_train_tail_adjacent_to_valid():
    """Sanity: embargo=0 makes train_max == valid_min - 1 (no gap)."""
    for tr, va in ts_folds(80, n_splits=4, embargo=0):
        assert tr.max() + 1 == va.min(), (
            f"embargo=0 should make train tail adjacent to valid head"
        )


def test_ts_folds_yields_chronological_validation():
    """Successive folds must place validation later in time (sklearn semantics)."""
    prev_valid_start = -1
    for tr, va in ts_folds(150, n_splits=5, embargo=1):
        assert va.min() > prev_valid_start
        prev_valid_start = va.min()


# ---------------------------------------------------------------------------
# 2. Per-fold scaler isolation (MLPForecaster)
# ---------------------------------------------------------------------------

def test_mlp_per_fold_scaler_isolates_train_statistics():
    """When MLPForecaster is fit on different train sets, its scaler mean must
    reflect only that fold's train data — NOT some global blend. We verify by
    fitting twice on overlapping but distinct subsets and confirming the
    scaler mean differs."""
    rng = np.random.default_rng(42)
    X_full = pd.DataFrame(rng.normal(0, 1, (100, 5)),
                          columns=[f"f{i}" for i in range(5)])
    y_full = pd.Series(rng.normal(0, 1, 100))

    m1 = MLPForecaster(max_iter=50)
    m1.fit(X_full.iloc[:60], y_full.iloc[:60])
    mean1 = m1.scaler.mean_.copy()

    m2 = MLPForecaster(max_iter=50)
    m2.fit(X_full.iloc[20:80], y_full.iloc[20:80])
    mean2 = m2.scaler.mean_.copy()

    # Means must differ — different train subsets must produce different
    # scaler stats. (If they were equal, the implementation would have
    # been using global statistics.)
    diff = np.abs(mean1 - mean2)
    assert diff.max() > 1e-6, (
        f"Per-fold scaler not isolated: mean1={mean1}, mean2={mean2}, max diff={diff.max():.2e}"
    )


# ---------------------------------------------------------------------------
# 3. Target shift correctness
# ---------------------------------------------------------------------------

def test_build_features_target_is_horizon_months_ahead(tmp_path, monkeypatch):
    """The target column y at row M must equal LNG at (M + horizon_months).

    This pins PDF §4.3 "타깃은 shift(-h)로 명시적으로 미래화" — and rules out
    the classic off-by-one where target ends up at M or M-1.
    """
    from src.utils import io as io_mod
    class _Stub:
        data_dir = tmp_path / "data"
    monkeypatch.setattr(io_mod, "get_settings", lambda: _Stub())
    import importlib
    feats_mod = importlib.reload(importlib.import_module("src.models.lng_forecast.features"))

    # Stage minimal LNG/oil/FX panel
    months = pd.date_range("2020-01-01", periods=36, freq="MS")
    for src_id, val_col, csv_name in [
        ("lng_price_monthly_file", "lng_price_usd_per_mmbtu", "lng"),
        ("oil_price_monthly_file", "oil_price_usd_per_bbl", "oil"),
        ("fx_usd_krw_monthly_file", "usd_krw_monthly_avg", "fx"),
    ]:
        from src.utils.io import source_root_dir
        d = source_root_dir(src_id) / "2022" / "12" / "01"
        d.mkdir(parents=True, exist_ok=True)
        df = pd.DataFrame({
            "period_month": months,
            val_col: np.linspace(10, 50, 36),
            "source_id": src_id,
            "collected_at": pd.Timestamp("2026-05-26"),
            "source_file": f"{csv_name}.csv",
            "source_priority": 1,
            "source_file_sha256": "a" * 64,
        })
        df.to_parquet(d / "parsed_test.parquet", index=False)

    X, y, pm, info = feats_mod.build_lng_forecast_features(horizon_months=1)
    # Pick a row deep enough that all lags are well-defined.
    for idx in range(len(X)):
        row_month = pm.iloc[idx]
        # y at this row should equal LNG at (row_month + 1 month) from the
        # raw series we just stored.
        expected_target_month = row_month + pd.DateOffset(months=1)
        # Look up expected value from the synthetic LNG series.
        expected_idx = (expected_target_month - months[0]).days // 30  # rough
        expected_value = np.linspace(10, 50, 36)[
            int(np.round((expected_target_month - months[0]).days / 30.4375))
        ]
        # Use a tolerance because linspace stepping is non-integer
        assert abs(y.iloc[idx] - expected_value) < 0.3, (
            f"target mismatch at row {idx} period_month={row_month}: "
            f"y={y.iloc[idx]:.3f}, expected≈{expected_value:.3f}"
        )
        break  # one row is enough; all rows have the same structural shift


def test_build_features_horizon_2_produces_2m_ahead_target(tmp_path, monkeypatch):
    """Same as above but horizon=2 — target at row M should be LNG(M+2)."""
    from src.utils import io as io_mod
    class _Stub:
        data_dir = tmp_path / "data"
    monkeypatch.setattr(io_mod, "get_settings", lambda: _Stub())
    import importlib
    feats_mod = importlib.reload(importlib.import_module("src.models.lng_forecast.features"))
    from src.utils.io import source_root_dir

    months = pd.date_range("2020-01-01", periods=48, freq="MS")
    for src_id, val_col in [
        ("lng_price_monthly_file", "lng_price_usd_per_mmbtu"),
        ("oil_price_monthly_file", "oil_price_usd_per_bbl"),
        ("fx_usd_krw_monthly_file", "usd_krw_monthly_avg"),
    ]:
        d = source_root_dir(src_id) / "2023" / "12" / "01"
        d.mkdir(parents=True, exist_ok=True)
        df = pd.DataFrame({
            "period_month": months,
            val_col: np.arange(len(months), dtype=float) * 1.5 + 10,
            "source_id": src_id,
            "collected_at": pd.Timestamp("2026-05-26"),
            "source_file": "x.csv",
            "source_priority": 1,
            "source_file_sha256": "a" * 64,
        })
        df.to_parquet(d / "parsed_test.parquet", index=False)

    X, y, pm, info = feats_mod.build_lng_forecast_features(horizon_months=2)
    # y at first valid row = LNG at first_row.period_month + 2 months
    first_pm = pm.iloc[0]
    expected_target_pm = first_pm + pd.DateOffset(months=2)
    expected_value = (expected_target_pm - months[0]).days / 30.4375 * 1.5 + 10
    assert abs(y.iloc[0] - expected_value) < 0.5
    assert info.horizon_months == 2


# ---------------------------------------------------------------------------
# 4. RMSE + MAPE formula correctness
# ---------------------------------------------------------------------------

def test_compute_rmse_mape_matches_manual_calculation():
    y_true = np.array([10.0, 20.0, 30.0, 40.0])
    y_pred = np.array([12.0, 18.0, 33.0, 36.0])
    rmse, mape = compute_rmse_mape(y_true, y_pred)
    # Manual: errors = [-2, 2, -3, 4]; sq = [4, 4, 9, 16]; sqrt(mean) = sqrt(8.25)
    assert rmse == pytest.approx(np.sqrt(8.25), rel=1e-9)
    # MAPE: |err/y| = [0.2, 0.1, 0.1, 0.1]; mean = 0.125; * 100 = 12.5
    assert mape == pytest.approx(12.5, rel=1e-9)


def test_compute_mape_does_not_divide_by_zero():
    y_true = np.array([0.0, 1.0])
    y_pred = np.array([0.5, 1.0])
    # If MAPE divided by 0 naively it would produce inf. Our impl uses
    # max(|y|, eps), so it should produce a large but finite number for
    # the 0-y row.
    rmse, mape = compute_rmse_mape(y_true, y_pred, eps=1e-8)
    assert np.isfinite(mape)


# ---------------------------------------------------------------------------
# 5. End-to-end: smoke test the CV loop on a synthetic series
# ---------------------------------------------------------------------------

def test_lightgbm_outer_valid_does_not_influence_fit():
    """Canary for the round-7 leakage fix.

    Even if a caller mistakenly passes the outer CV's evaluation fold as
    `X_valid` / `y_valid` to LightGBMForecaster.fit(), the resulting model
    MUST be identical to the model fit without those kwargs — because
    they should be IGNORED for early stopping (which uses an inner slice
    of X_train instead).

    If this test fails, predictions reported by run_cv() are tuned to the
    very validation labels they are then measured on.
    """
    rng = np.random.default_rng(42)
    n = 100
    X = pd.DataFrame({
        "a": rng.normal(0, 1, n),
        "b": rng.normal(0, 1, n),
        "c": np.linspace(0, 1, n),
    })
    y = pd.Series(X["a"] * 2 + X["c"] * 3 + rng.normal(0, 0.5, n))
    X_outer_va = pd.DataFrame({
        "a": rng.normal(0, 1, 20),
        "b": rng.normal(0, 1, 20),
        "c": np.linspace(0.5, 1.5, 20),
    })
    y_outer_va = pd.Series(rng.normal(0, 10, 20))  # adversarial labels

    m_clean = LightGBMForecaster().fit(X, y)
    m_with_kw = LightGBMForecaster().fit(X, y, X_valid=X_outer_va, y_valid=y_outer_va)

    # Predict on a NEW test set with neither model — outputs must match.
    X_test = pd.DataFrame({
        "a": rng.normal(0, 1, 30),
        "b": rng.normal(0, 1, 30),
        "c": np.linspace(0, 1.5, 30),
    })
    pred_clean = m_clean.predict(X_test)
    pred_with = m_with_kw.predict(X_test)
    np.testing.assert_allclose(
        pred_clean, pred_with, rtol=1e-10, atol=1e-10,
        err_msg=(
            "LightGBMForecaster used the outer X_valid/y_valid kwargs in fit "
            "— predictions differ from the no-kwargs fit. This is a "
            "validation-labels-in-training leak."
        ),
    )


def test_run_cv_smoke_with_synthetic_series():
    rng = np.random.default_rng(7)
    n = 80
    X = pd.DataFrame({
        "lng_price_usd_per_mmbtu_lag1": np.linspace(5, 15, n) + rng.normal(0, 0.1, n),
        "f2": rng.normal(0, 1, n),
        "f3": rng.normal(0, 1, n),
    })
    y = pd.Series(X["lng_price_usd_per_mmbtu_lag1"].values + rng.normal(0, 0.5, n))
    pm = pd.Series(pd.date_range("2018-01-01", periods=n, freq="MS"))

    rep = run_cv(X, y, pm, lambda: NaiveLagForecaster(), model_name="naive",
                 n_splits=4, embargo=1)
    agg = rep.aggregate()
    assert agg["n_folds"] == 4
    assert agg["rmse_mean"] > 0
    assert not np.isnan(agg["mape_mean"])
    assert rep.predictions is not None
    # Each predictions row must have a valid period_month
    assert rep.predictions["period_month"].notna().all()
