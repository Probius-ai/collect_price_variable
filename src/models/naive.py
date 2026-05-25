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
    """Monthly baseline driven by ``smp_lag_1m``.

    Important: with target = SMP at M+1 and feature smp_lag_1m = SMP at M-1,
    this model fits ``SMP(M+1) ≈ SMP(M-1)`` — a **2-step seasonal lag**, NOT
    a true persistence baseline. The visual symptom on the dashboard is that
    the predicted line looks like the actual SMP shifted forward by 2
    months. Use ``PersistenceMonthly`` below for the true "next month equals
    current month" baseline.
    """

    name = "naive_lag_1m"

    def __init__(self, lag_column: str = "smp_lag_1m") -> None:
        super().__init__(lag_column=lag_column)


class PersistenceMonthly(NaiveLag24h):
    """True persistence baseline: predicts target_(M+1) as SMP at M.

    Uses ``smp_t_observed`` — the current-month SMP — which is observed
    before we forecast M+1 (no leakage). This is the correct "last known
    value" baseline and is what most forecasting literature means by
    "naive" for a 1-step-ahead monthly forecast.
    """

    name = "persistence_monthly"

    def __init__(self, lag_column: str = "smp_t_observed") -> None:
        super().__init__(lag_column=lag_column)


class SeasonalNaiveLag12m(NaiveLag24h):
    """Monthly seasonal baseline: same-month-last-year SMP."""

    name = "seasonal_naive_lag_12m"

    def __init__(self, lag_column: str = "smp_lag_12m") -> None:
        super().__init__(lag_column=lag_column)
