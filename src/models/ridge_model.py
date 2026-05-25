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
    # `smp_t_observed` = SMP at row M (the issue date). Observed before we
    # forecast M+1 — not leakage. This is the dominant predictive signal
    # and was previously withheld from features, which forced "naive"
    # models to lean on smp_lag_1m (= SMP at M-1) and effectively become
    # 2-step-lag baselines instead of true persistence baselines.
    "smp_t_observed",
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
    # NB: removed `is_lng_shock_period` (post-hoc / look-ahead leakage —
    # boundaries were defined retrospectively from the valid peak threshold).
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
        # Auto-detect granularity. We mark a frame as "monthly" when ANY of
        # these holds: it carries a monthly lag (_lag_1m / _lag_12m / _lag_1y)
        # OR the explicit smp_t_observed observed-at-origin column. The
        # smp_t_observed branch was added in round 5 because feature-group
        # ablation handed Ridge subsets containing smp_t_observed alone (no
        # lag columns) and the old heuristic silently picked the hourly pool
        # — yielding an empty intersection and a StandardScaler error.
        is_monthly = (
            "smp_t_observed" in X.columns
            or any(
                c.endswith("_lag_1m") or c.endswith("_lag_12m") or c.endswith("_lag_1y")
                for c in X.columns
            )
        )
        default_pool = (
            DEFAULT_RIDGE_FEATURES_MONTHLY if is_monthly else DEFAULT_RIDGE_FEATURES_HOURLY
        )
        chosen = [c for c in default_pool if c in X.columns]
        if chosen:
            return chosen
        # Last-resort fallback: if the default pool didn't intersect with X
        # (e.g. ablation handed us a feature group composed entirely of
        # capacity_* columns Ridge's default doesn't list), use every numeric
        # column in X. Better than silently raising 0-feature errors.
        return list(X.columns)

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
