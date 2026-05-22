from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.features.build import build_smp_hourly_features
from src.models.lightgbm_model import LightGBMModel
from src.models.metrics import compute_metrics
from src.models.naive import NaiveLag24h, SeasonalNaiveLag168h
from src.models.ridge_model import RidgeModel


@pytest.fixture
def feature_table(synthetic_smp_dataframe):
    return build_smp_hourly_features(synthetic_smp_dataframe, forecast_horizon_hours=24)


def _split(df):
    n = len(df)
    split = int(n * 0.7)
    return df.iloc[:split], df.iloc[split:]


def test_naive_predicts_lag_column(feature_table):
    train, test = _split(feature_table)
    m = NaiveLag24h().fit(train, train["target_smp_t_plus_h"])
    preds = m.predict(test)
    np.testing.assert_allclose(preds.to_numpy(), test["smp_lag_24h"].to_numpy())


def test_seasonal_naive_uses_168h_lag(feature_table):
    train, test = _split(feature_table)
    m = SeasonalNaiveLag168h().fit(train, train["target_smp_t_plus_h"])
    preds = m.predict(test)
    np.testing.assert_allclose(preds.to_numpy(), test["smp_lag_168h"].to_numpy())


def test_ridge_runs_and_returns_metrics(feature_table):
    train, test = _split(feature_table)
    feature_cols = [c for c in train.columns if c not in {"target_smp_t_plus_h", "interval_end", "area", "season"}]
    m = RidgeModel(feature_cols=[c for c in feature_cols if c in [
        "demand_forecast_mw", "smp_lag_1h", "smp_lag_24h", "smp_lag_168h",
        "smp_rolling_24h_mean", "hour_sin", "hour_cos", "dow_sin", "dow_cos", "month",
    ]]).fit(train, train["target_smp_t_plus_h"])
    preds = m.predict(test)
    metrics = compute_metrics(test["target_smp_t_plus_h"], preds)
    assert metrics.mae > 0
    assert metrics.rmse >= metrics.mae


def test_lightgbm_trains_and_predicts(feature_table):
    train, test = _split(feature_table)
    m = LightGBMModel(num_boost_round=50, early_stopping_rounds=10)
    m.fit(train, train["target_smp_t_plus_h"], X_valid=test, y_valid=test["target_smp_t_plus_h"])
    preds = m.predict(test)
    assert len(preds) == len(test)
    imp = m.feature_importance()
    assert not imp.empty
    assert {"feature", "importance_gain"}.issubset(imp.columns)
