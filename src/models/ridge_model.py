"""Ridge regression baseline."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


DEFAULT_RIDGE_FEATURES_HOURLY = [
    "demand_forecast_mw",
    "demand_lag_24h",
    "demand_lag_168h",
    "smp_lag_1h",
    "smp_lag_24h",
    "smp_lag_168h",
    "smp_rolling_24h_mean",
    "smp_rolling_7d_mean",
    "hour_sin",
    "hour_cos",
    "dow_sin",
    "dow_cos",
    "month",
]

DEFAULT_RIDGE_FEATURES_MONTHLY = [
    "smp_lag_1m",
    "smp_lag_2m",
    "smp_lag_3m",
    "smp_lag_6m",
    "smp_lag_12m",
    "smp_rolling_3m_mean",
    "smp_rolling_6m_mean",
    "smp_rolling_12m_mean",
    "smp_rolling_12m_std",
    "month_sin",
    "month_cos",
    "quarter",
    # Settlement (wide-by-fuel, lag_1m) — added when settlement monthly is loaded.
    "settlement_unit_price_total_lag_1m",
    "settlement_unit_price_nuclear_lag_1m",
    "settlement_unit_price_coal_total_lag_1m",
    "settlement_unit_price_lng_lag_1m",
    "settlement_unit_price_renewable_lag_1m",
]

# Kept for backward-compat
DEFAULT_RIDGE_FEATURES = DEFAULT_RIDGE_FEATURES_HOURLY


class RidgeModel:
    name = "ridge"

    def __init__(self, alpha: float = 1.0, feature_cols: list[str] | None = None) -> None:
        self.alpha = alpha
        self.feature_cols = feature_cols
        self.pipeline: Pipeline | None = None
        self.used_features: list[str] = []

    def _select_features(self, X: pd.DataFrame) -> list[str]:
        if self.feature_cols is not None:
            missing = [c for c in self.feature_cols if c not in X.columns]
            if missing:
                raise KeyError(f"Ridge: missing requested features {missing}")
            return list(self.feature_cols)
        # Auto-detect granularity: prefer monthly defaults when monthly lags exist.
        if any(c.endswith("_lag_1m") or c.endswith("_lag_12m") for c in X.columns):
            return [c for c in DEFAULT_RIDGE_FEATURES_MONTHLY if c in X.columns]
        return [c for c in DEFAULT_RIDGE_FEATURES_HOURLY if c in X.columns]

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "RidgeModel":
        self.used_features = self._select_features(X)
        self.pipeline = Pipeline(
            [("scaler", StandardScaler()), ("ridge", Ridge(alpha=self.alpha))]
        )
        self.pipeline.fit(X[self.used_features].to_numpy(dtype=float), np.asarray(y, dtype=float))
        return self

    def predict(self, X: pd.DataFrame) -> pd.Series:
        if self.pipeline is None:
            raise RuntimeError("RidgeModel.fit() has not been called yet")
        preds = self.pipeline.predict(X[self.used_features].to_numpy(dtype=float))
        return pd.Series(preds, index=X.index, name="prediction")
