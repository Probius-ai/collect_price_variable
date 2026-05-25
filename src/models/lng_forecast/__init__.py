"""LNG monthly price forecasting package — round 7.

Implements the design from `docs/에너지가격예측_파이프라인_가이드.pdf` and
`docs/energy-price-forecast-pipeline-guide.pdf`:

  * inputs:  LNG history + Dubai oil + USD/KRW (+ calendar)
  * split:   TimeSeriesSplit (expanding window) + embargo
  * preprocessing: per-fold scaler fit (train only)
  * models:  LightGBM + MLPRegressor (single-fold + cross-validated)
  * metrics: RMSE + MAPE, fold-level mean ± std

Public API:
  * build_lng_forecast_features(...)  — panel + feature engineering
  * ts_folds(n, n_splits, embargo)    — leakage-safe fold generator
  * run_cv(features, target, ...)     — end-to-end CV loop
  * LightGBMForecaster / MLPForecaster
"""

from src.models.lng_forecast.features import build_lng_forecast_features
from src.models.lng_forecast.split import ts_folds
from src.models.lng_forecast.models import LightGBMForecaster, MLPForecaster
from src.models.lng_forecast.evaluate import compute_rmse_mape, run_cv

__all__ = [
    "build_lng_forecast_features",
    "ts_folds",
    "LightGBMForecaster",
    "MLPForecaster",
    "compute_rmse_mape",
    "run_cv",
]
