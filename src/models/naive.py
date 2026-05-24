"""Naive baselines: lag-24h and lag-168h (seasonal weekly)."""

from __future__ import annotations

import pandas as pd


class NaiveLag24h:
    """Predicts target_t+h as smp_lag_24h carried forward.

    For the standard 24h-horizon SMP task this is "predict tomorrow same hour
    equals SMP at the issue time" because feature `smp_lag_24h` (which sits
    on the row at time t) equals SMP at t-24h, and target at row t = SMP at
    t+24h. The strong baseline of "yesterday at this hour" instead uses the
    current observed SMP, but per Plan.md §9.1 we follow the lag_24h naming
    that names *which lag column* drives the prediction.
    """

    name = "naive_lag_24h"

    def __init__(self, lag_column: str = "smp_lag_24h") -> None:
        self.lag_column = lag_column

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "NaiveLag24h":
        if self.lag_column not in X.columns:
            raise KeyError(f"NaiveLag24h needs '{self.lag_column}' in X")
        return self

    def predict(self, X: pd.DataFrame) -> pd.Series:
        if self.lag_column not in X.columns:
            raise KeyError(f"NaiveLag24h needs '{self.lag_column}' in X")
        return X[self.lag_column].astype(float)


class SeasonalNaiveLag168h(NaiveLag24h):
    """Predicts target as same-hour-last-week SMP."""

    name = "seasonal_naive_lag_168h"

    def __init__(self, lag_column: str = "smp_lag_168h") -> None:
        super().__init__(lag_column=lag_column)


class NaiveLag1m(NaiveLag24h):
    """Monthly naive baseline: predicts target as previous-month SMP."""

    name = "naive_lag_1m"

    def __init__(self, lag_column: str = "smp_lag_1m") -> None:
        super().__init__(lag_column=lag_column)


class SeasonalNaiveLag12m(NaiveLag24h):
    """Monthly seasonal baseline: same-month-last-year SMP."""

    name = "seasonal_naive_lag_12m"

    def __init__(self, lag_column: str = "smp_lag_12m") -> None:
        super().__init__(lag_column=lag_column)
