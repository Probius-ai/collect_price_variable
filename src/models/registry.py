"""Classification of monthly forecasting models.

Baseline models predict from a single rule (or no learnable parameters),
so they serve as the floor any trainable model must beat to justify its
complexity. Trainable models fit parameters to history.

We expose two sets so the dashboard / evaluation pipelines can highlight
the comparison: "trainable model X beat baseline Y by Z KRW/kWh".
"""

from __future__ import annotations

# Hourly + monthly baselines (no learnable parameters; rule-driven).
BASELINE_MODELS: frozenset[str] = frozenset({
    "naive",                    # hourly: smp_lag_24h
    "seasonal_naive",           # hourly: smp_lag_168h
    "persistence_monthly",      # monthly: predicts SMP(M+1) = SMP(M)
    "naive_lag_1m",             # monthly: predicts SMP(M+1) = SMP(M-1) (2-step lag)
    "seasonal_naive_lag_12m",   # monthly: SMP(M+1) = SMP(M-11)
})

# Trainable models with learnable parameters. Each should be compared to
# the strongest applicable baseline (persistence_monthly for the monthly
# task) — beating it by a meaningful margin is the bar.
TRAINABLE_MODELS: frozenset[str] = frozenset({
    "ridge",
    "lightgbm",
    "monthly_ar_ridge",
    "delta_ridge",
    "delta_lightgbm",
    "delta_ar_ridge",
})

# Strong reference baseline for the monthly 1-step task. Dashboards and
# evaluation tables overlay this against trainable models.
STRONG_MONTHLY_BASELINE: str = "persistence_monthly"

# Convenience default the dashboard should pick when the user has not
# selected a model yet. We deliberately pick a trainable model so the
# default view shows what the project's modelling adds beyond the
# persistence baseline.
DEFAULT_DASHBOARD_MODEL: str = "monthly_ar_ridge"


def classify(model_name: str) -> str:
    """Return 'baseline', 'trainable', or 'unknown' for a model name."""
    if model_name in BASELINE_MODELS:
        return "baseline"
    if model_name in TRAINABLE_MODELS:
        return "trainable"
    return "unknown"
