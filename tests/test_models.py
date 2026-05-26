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


def test_ridge_default_pool_excludes_sparse_jkm_daily_columns():
    """Round-9-review fix: the JKM investing.com daily history starts in
    2013, so `jkm_daily_*_lag_1m` columns are NaN for SMP months before
    that. Ridge rejects NaN, so these columns MUST stay out of the default
    auto-selected pool. (LightGBM tolerates NaN and gets these via its own
    feature list — different model, different default.)
    """
    import numpy as np
    from src.models.ridge_model import RidgeModel, DEFAULT_RIDGE_FEATURES_MONTHLY

    sparse_jkm = {
        "jkm_daily_mean_lag_1m",
        "jkm_daily_std_lag_1m",
        "jkm_daily_range_lag_1m",
    }
    assert sparse_jkm.isdisjoint(DEFAULT_RIDGE_FEATURES_MONTHLY), (
        "Sparse JKM daily-derived columns leaked back into the Ridge default pool. "
        "These cause Ridge fits to fail with NaN errors on the pre-2013 portion of "
        "the SMP monthly panel."
    )

    # Monthly frame that includes a sparse JKM column → Ridge auto-selects
    # only the clean features and the fit succeeds.
    n = 60
    X = pd.DataFrame({
        "smp_t_observed": np.linspace(80, 120, n),
        "smp_lag_1m":     np.linspace(80, 120, n),
        "jkm_daily_mean_lag_1m": [np.nan] * 20 + list(np.linspace(15, 20, 40)),
    })
    y = pd.Series(np.linspace(90, 130, n))
    m = RidgeModel().fit(X, y)
    assert "jkm_daily_mean_lag_1m" not in m.used_features
    assert set(m.used_features) == {"smp_t_observed", "smp_lag_1m"}
    # And the predict path works end-to-end
    preds = m.predict(X)
    assert len(preds) == n and not preds.isna().any()


def test_ridge_pipeline_imputes_nans_when_explicit_features_have_them():
    """Defense-in-depth: if a caller explicitly asks Ridge to use a column
    that happens to contain NaN (e.g. feature-group ablation hands us a
    set that includes a sparse column), the SimpleImputer step in Ridge's
    pipeline imputes the NaN rather than letting sklearn raise.
    """
    import numpy as np
    from src.models.ridge_model import RidgeModel

    n = 40
    X = pd.DataFrame({
        "smp_t_observed":  np.linspace(80, 120, n),
        "some_sparse_col": [np.nan] * 10 + list(np.linspace(1.0, 5.0, 30)),
    })
    y = pd.Series(np.linspace(90, 130, n))
    # Caller-specified features force the sparse column in
    m = RidgeModel(feature_cols=["smp_t_observed", "some_sparse_col"]).fit(X, y)
    preds = m.predict(X)
    assert len(preds) == n
    assert not preds.isna().any()
