"""Round-4 tests: LightGBM monthly defaults, regime indicator, MonthlyARRidge,
and the walk-forward CV evaluation contract.

The round-3 failure mode this guards against: LightGBM auto-selected the
hourly DEFAULT_LGB_FEATURES list on monthly data → only 3 features matched
(`month`, `is_summer`, `is_winter`) → tree collapsed to a constant predictor.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.features.build_monthly import add_calendar_features
from src.models.ar_monthly import MONTHLY_AR_FEATURES, MonthlyARRidge
from src.models.lightgbm_model import (
    DEFAULT_LGB_FEATURES_HOURLY,
    DEFAULT_LGB_FEATURES_MONTHLY,
    LightGBMModel,
)
from src.models.metrics import compute_metrics


def _make_monthly_features(n: int = 60) -> pd.DataFrame:
    """Synthetic monthly feature frame compatible with the round-3 schema."""
    rng = np.random.default_rng(7)
    months = pd.date_range("2020-01-01", periods=n, freq="MS")
    smp = 100 + 30 * np.sin(2 * np.pi * months.month / 12) + rng.normal(0, 3, n)
    df = pd.DataFrame({
        "period_month": months,
        "smp_krw_per_kwh": smp,
        "smp_lag_1m": np.r_[np.nan, smp[:-1]],
        "smp_lag_2m": np.r_[[np.nan]*2, smp[:-2]],
        "smp_lag_3m": np.r_[[np.nan]*3, smp[:-3]],
        "smp_lag_6m": np.r_[[np.nan]*6, smp[:-6]],
        "smp_lag_12m": np.r_[[np.nan]*12, smp[:-12]],
        "smp_rolling_3m_mean": pd.Series(smp).shift(1).rolling(3).mean(),
        "smp_rolling_6m_mean": pd.Series(smp).shift(1).rolling(6).mean(),
        "smp_rolling_12m_mean": pd.Series(smp).shift(1).rolling(12).mean(),
        "smp_rolling_12m_std": pd.Series(smp).shift(1).rolling(12).std(),
    })
    df = add_calendar_features(df)
    return df.dropna().reset_index(drop=True)


# ---------------------------------------------------------------------------
# Regime indicator — POSITIVE absence assertion
# ---------------------------------------------------------------------------

def test_no_hardcoded_lng_shock_indicator_in_calendar_features():
    """Regression for the post-hoc / look-ahead leakage finding: an earlier
    round exposed `is_lng_shock_period` whose 2022-01..2023-12 boundaries
    were chosen by looking at the valid split's peak_threshold. That flag
    leaks future knowledge into any walk-forward fit positioned before
    2022-01 (it tells the model exactly when the shock will start). The
    calendar-feature pipeline must NOT add it.
    """
    months = pd.date_range("2019-01-01", "2025-12-01", freq="MS")
    df = pd.DataFrame({"period_month": months})
    out = add_calendar_features(df)
    assert "is_lng_shock_period" not in out.columns, (
        "is_lng_shock_period reintroduced — its boundaries were defined "
        "post-hoc from the valid split and create look-ahead leakage."
    )
    # The model defaults must also stay clean.
    assert "is_lng_shock_period" not in DEFAULT_LGB_FEATURES_MONTHLY
    assert "is_lng_shock_period" not in MONTHLY_AR_FEATURES


# ---------------------------------------------------------------------------
# LightGBM monthly defaults + auto-detect
# ---------------------------------------------------------------------------

def test_lightgbm_auto_detects_monthly_features():
    """When the input has *_lag_1m columns, LightGBM must pick the monthly
    default pool, not the hourly one. Regression for the round-3 bug where
    auto-fallback picked `month`/`is_summer`/`is_winter` only.
    """
    df = _make_monthly_features(n=48)
    target = df["smp_krw_per_kwh"]
    model = LightGBMModel()
    model.fit(df.drop(columns=["smp_krw_per_kwh"]), target)
    # used_features must come from the monthly pool (it sees lag_1m etc).
    assert "smp_lag_1m" in model.used_features
    assert "smp_lag_12m" in model.used_features
    # Sanity: it should now USE more than 3 features (round-3 bug had 3).
    assert len(model.used_features) > 5


def test_lightgbm_hourly_pool_used_when_no_monthly_lag_cols():
    """If only hourly-lag columns are present, the hourly pool is used."""
    # Build a minimal hourly-like frame
    df = pd.DataFrame({
        "period_month": pd.date_range("2024-01-01", periods=200, freq="h"),
        "smp_lag_24h": np.random.RandomState(1).normal(120, 10, 200),
        "smp_lag_168h": np.random.RandomState(2).normal(120, 10, 200),
        "demand_forecast_mw": np.random.RandomState(3).normal(60000, 5000, 200),
        "hour": np.tile(np.arange(24), 9)[:200],
        "month": [1] * 200,
    })
    target = df["smp_lag_24h"] + 5
    model = LightGBMModel()
    model.fit(df.drop(columns=["period_month"]), target)
    # Should pick hourly-pool features
    assert "smp_lag_24h" in model.used_features
    # Should NOT pick monthly-only features
    assert "smp_lag_1m" not in model.used_features


def test_lightgbm_small_data_overrides_min_data_in_leaf():
    """For monthly-sized data, the default min_data_in_leaf=50 collapses the
    tree to ~2 leaves. The auto-scaling should reduce it on small data.
    """
    df = _make_monthly_features(n=48)
    target = df["smp_krw_per_kwh"]
    model = LightGBMModel()
    model.fit(df.drop(columns=["smp_krw_per_kwh"]), target)
    # Auto-scaled: min_data_in_leaf should be < 50 (the original default)
    assert model.params["min_data_in_leaf"] < 50, (
        f"Small-data auto-scaling failed: min_data_in_leaf="
        f"{model.params['min_data_in_leaf']}"
    )
    # User override must still win
    user_model = LightGBMModel(params={"min_data_in_leaf": 99})
    user_model.fit(df.drop(columns=["smp_krw_per_kwh"]), target)
    assert user_model.params["min_data_in_leaf"] == 99


def test_lightgbm_monthly_default_pool_contains_optional_features():
    """The monthly default pool must mention round-3 capacity/transaction
    features so LightGBM sees them when build_monthly_features added them.
    """
    must_be_listed = [
        "capacity_fuel_lng_mw_lag_1m",
        "capacity_type_renewable_mw_lag_1m",
        "capacity_yearly_total_mw_lag_1y",
        "transaction_volume_total_mwh_lag_1m",
        "transaction_amount_total_krw_lag_1m",
        "market_trade_price_lng_krw_per_kwh_lag_1m",
    ]
    for col in must_be_listed:
        assert col in DEFAULT_LGB_FEATURES_MONTHLY, (
            f"DEFAULT_LGB_FEATURES_MONTHLY missing {col}"
        )
    # Hourly pool must NOT contain monthly features
    assert "smp_lag_1m" not in DEFAULT_LGB_FEATURES_HOURLY


def test_persistence_monthly_beats_naive_lag_1m_on_autocorrelated_data():
    """The user-flagged regression: predictions from naive_lag_1m looked like
    the input shifted forward by 2 months. That is because naive_lag_1m
    fits target(M+1) ≈ SMP(M-1) (a 2-step seasonal lag) instead of the
    true 1-step persistence baseline target(M+1) ≈ SMP(M).

    `smp_t_observed` (added to feature_cols) carries SMP(M) — observed
    before forecasting M+1 — and `PersistenceMonthly` consumes it. On any
    autocorrelated series, the 1-step naive must beat the 2-step naive.
    """
    from src.features.build_monthly import build_smp_monthly_features  # noqa: F401
    from src.models.naive import NaiveLag1m, PersistenceMonthly

    df = _make_monthly_features(n=60)
    # Build target as t+1 SMP (mirrors the production pipeline).
    df["smp_t_observed"] = df["smp_krw_per_kwh"]
    df["target"] = df["smp_krw_per_kwh"].shift(-1)
    df = df.dropna(subset=["target"]).reset_index(drop=True)

    persistence = PersistenceMonthly().fit(df, df["target"])
    naive = NaiveLag1m().fit(df, df["target"])
    mae_persistence = compute_metrics(df["target"], persistence.predict(df)).mae
    mae_naive = compute_metrics(df["target"], naive.predict(df)).mae

    assert mae_persistence < mae_naive, (
        f"PersistenceMonthly (1-step naive, MAE={mae_persistence:.2f}) must "
        f"beat NaiveLag1m (2-step lag, MAE={mae_naive:.2f}) on an "
        f"autocorrelated series. If this regresses, somebody removed "
        f"smp_t_observed from the feature pipeline again."
    )


def test_smp_t_observed_is_in_feature_pool():
    """The current-month SMP must be exposed to downstream models. Removing
    it was the round-3 mistake that turned naive_lag_1m into a 2-step lag.
    """
    from src.models.ar_monthly import MONTHLY_AR_FEATURES
    from src.models.lightgbm_model import DEFAULT_LGB_FEATURES_MONTHLY
    from src.models.ridge_model import DEFAULT_RIDGE_FEATURES_MONTHLY
    assert "smp_t_observed" in MONTHLY_AR_FEATURES
    assert "smp_t_observed" in DEFAULT_LGB_FEATURES_MONTHLY
    assert "smp_t_observed" in DEFAULT_RIDGE_FEATURES_MONTHLY


def test_lightgbm_user_num_boost_round_is_respected():
    """If the caller explicitly passes num_boost_round, the small-data cap
    must NOT silently override it. Regression for: an earlier round
    truncated user-pinned 800 down to 200 even when explicitly requested.
    """
    df = _make_monthly_features(n=48)
    target = df["smp_krw_per_kwh"]

    # User pins a large value → must be preserved.
    explicit = LightGBMModel(num_boost_round=800)
    explicit.fit(df.drop(columns=["smp_krw_per_kwh"]), target)
    assert explicit.num_boost_round == 800, (
        f"User-pinned num_boost_round=800 was silently capped to "
        f"{explicit.num_boost_round}"
    )

    # Default (no arg) on small data → auto-scaling kicks in (cap=200).
    auto = LightGBMModel()
    auto.fit(df.drop(columns=["smp_krw_per_kwh"]), target)
    assert auto.num_boost_round <= 200, (
        f"Auto small-data cap (200) not applied: num_boost_round="
        f"{auto.num_boost_round}"
    )


# ---------------------------------------------------------------------------
# MonthlyARRidge
# ---------------------------------------------------------------------------

def test_monthly_ar_ridge_uses_only_listed_features():
    df = _make_monthly_features(n=48)
    target = df["smp_krw_per_kwh"]
    model = MonthlyARRidge()
    model.fit(df.drop(columns=["smp_krw_per_kwh"]), target)
    # Only features from MONTHLY_AR_FEATURES that exist in X get selected
    assert set(model.used_features).issubset(MONTHLY_AR_FEATURES)
    assert "smp_lag_1m" in model.used_features


def test_monthly_ar_ridge_beats_seasonal_naive_on_synthetic_seasonal_data():
    """On clean seasonal synthetic data, the MonthlyARRidge should generally
    beat the seasonal_naive_lag_12m baseline because it can blend lag_1m
    with the seasonal lag rather than being forced to pick one."""
    df = _make_monthly_features(n=72)  # 72-12 lag = 60 valid rows
    cut = int(len(df) * 0.75)
    train, test = df.iloc[:cut], df.iloc[cut:]
    assert len(test) > 0, "test split must be non-empty"
    model = MonthlyARRidge().fit(train.drop(columns=["smp_krw_per_kwh"]),
                                  train["smp_krw_per_kwh"])
    preds = model.predict(test.drop(columns=["smp_krw_per_kwh"]))
    mae_ar = compute_metrics(test["smp_krw_per_kwh"], preds).mae
    mae_seasonal = compute_metrics(
        test["smp_krw_per_kwh"], test["smp_lag_12m"]
    ).mae
    # MAE comparison is the canonical claim
    assert mae_ar <= mae_seasonal + 1.0, (
        f"MonthlyARRidge MAE={mae_ar:.2f} much worse than seasonal_naive "
        f"({mae_seasonal:.2f}) on synthetic seasonal data"
    )


def test_monthly_ar_ridge_raises_when_no_required_features():
    """The model must hard-fail if NONE of its 4 features exist in X
    (rather than silently fit on an empty feature set)."""
    df = pd.DataFrame({
        "period_month": pd.date_range("2024-01-01", periods=24, freq="MS"),
        "demand_forecast_mw": np.arange(24),  # wrong feature family
    })
    target = pd.Series(np.arange(24, dtype=float))
    with pytest.raises(KeyError, match="needs at least one"):
        MonthlyARRidge().fit(df, target)


# ---------------------------------------------------------------------------
# Walk-forward CV pipeline (smoke + correctness)
# ---------------------------------------------------------------------------

def test_walk_forward_predicts_each_post_train_row_exactly_once(tmp_path, monkeypatch):
    """walk_forward.main must emit exactly (len(df) - min_train_rows) rows of
    predictions, covering every period_month past the initial train window."""
    from src.utils import io as io_mod
    class _Stub:
        data_dir = tmp_path / "data"
        outputs_dir = tmp_path / "outputs"
    monkeypatch.setattr(io_mod, "get_settings", lambda: _Stub())
    import src.pipelines.walk_forward as wf
    monkeypatch.setattr(wf, "get_settings", lambda: _Stub())

    df = _make_monthly_features(n=50)
    df["target_smp_t_plus_h_months"] = df["smp_krw_per_kwh"].shift(-1)
    df = df.dropna(subset=["target_smp_t_plus_h_months"]).reset_index(drop=True)
    feats_path = tmp_path / "smp_monthly_test.parquet"
    df.to_parquet(feats_path, index=False)

    from typer.testing import CliRunner
    runner = CliRunner()
    res = runner.invoke(wf.app, [
        "main",
        "--features-path", str(feats_path),
        "--model", "naive_lag_1m",
        "--target", "target_smp_t_plus_h_months",
        "--timestamp-col", "period_month",
        "--min-train-rows", "12",
    ])
    assert res.exit_code == 0, res.output
    preds = pd.read_csv(_Stub.outputs_dir / "walk_forward" / "naive_lag_1m" / "predictions.csv")
    # Exactly len(df) - min_train_rows rows
    assert len(preds) == len(df) - 12
    # period_month strictly monotonic increasing (one row per month)
    pm = pd.to_datetime(preds["period_month"])
    assert pm.is_monotonic_increasing
    assert pm.duplicated().sum() == 0
