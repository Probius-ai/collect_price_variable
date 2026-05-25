"""Small-feature monthly AR baseline.

The point: the full-default Ridge (17 features) and LightGBM (50+ features)
both regress poorly on the post-shock test split (MAE > 14). The naive
`smp_lag_1m` gets MAE 9.40 just by trusting the previous month.

This model splits the difference: a Ridge restricted to the handful of
features that have always been informative AND robust across regimes —

  * smp_lag_1m              (the dominant autocorrelation signal)
  * smp_rolling_3m_mean     (3-month average to smooth single-month noise)
  * smp_lag_12m             (year-on-year — preserves seasonality where it
                             still holds)

It deliberately omits the 10 settlement_unit_price_*_lag_1m columns whose
post-shock decoupling drove the full-Ridge into negative R^2 on the test
split. It also intentionally OMITS any hardcoded regime indicator: an
earlier round shipped `is_lng_shock_period` but that flag's boundaries
were picked retrospectively from the valid peak threshold, which leaks
future knowledge into a walk-forward fit.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


MONTHLY_AR_FEATURES = [
    # smp_t_observed = SMP at issue date M (no leakage when forecasting M+1).
    # This is the dominant 1-step-ahead signal — including it converts the
    # baseline from a 2-step lag to a true persistence-plus-residual model.
    "smp_t_observed",
    "smp_lag_1m",
    "smp_rolling_3m_mean",
    "smp_lag_12m",
    # LNG leading-indicator (round 6). FRED JKM Asia monthly avg is
    # published ~mid-M+1 → LNG(M) NOT observable at SMP info_cutoff=end-of-M.
    # Only lag_1m (= LNG(M-1)) is fair game. Unlagged column removed after
    # round-6 leakage review.
    "lng_price_usd_per_mmbtu_lag_1m",
    "lng_price_chg_1m_lag_1m",
]


class MonthlyARRidge:
    name = "monthly_ar_ridge"

    def __init__(
        self, alpha: float = 1.0, feature_cols: list[str] | None = None
    ) -> None:
        self.alpha = alpha
        # When ``feature_cols`` is provided (e.g. by the feature-group
        # ablation pipeline), it OVERRIDES the hardcoded MONTHLY_AR_FEATURES
        # selection — otherwise ablation results would silently ignore the
        # extra capacity/transaction columns and report identical MAE
        # across every group that contains smp_t_observed.
        self.feature_cols = feature_cols
        self.pipeline: Pipeline | None = None
        self.used_features: list[str] = []

    def _select(self, X: pd.DataFrame) -> list[str]:
        if self.feature_cols is not None:
            missing = [c for c in self.feature_cols if c not in X.columns]
            if missing:
                raise KeyError(
                    f"MonthlyARRidge: requested feature_cols not in X: {missing}"
                )
            return list(self.feature_cols)
        return [c for c in MONTHLY_AR_FEATURES if c in X.columns]

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "MonthlyARRidge":
        self.used_features = self._select(X)
        if not self.used_features:
            raise KeyError(
                f"MonthlyARRidge needs at least one of {MONTHLY_AR_FEATURES} "
                f"in X; got columns {list(X.columns)[:8]}..."
            )
        self.pipeline = Pipeline(
            [("scaler", StandardScaler()), ("ridge", Ridge(alpha=self.alpha))]
        )
        self.pipeline.fit(
            X[self.used_features].to_numpy(dtype=float),
            np.asarray(y, dtype=float),
        )
        return self

    def predict(self, X: pd.DataFrame) -> pd.Series:
        if self.pipeline is None:
            raise RuntimeError("MonthlyARRidge.fit() has not been called yet")
        preds = self.pipeline.predict(
            X[self.used_features].to_numpy(dtype=float)
        )
        return pd.Series(preds, index=X.index, name="prediction")
