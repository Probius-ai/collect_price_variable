"""LightGBM regressor for SMP forecasting."""

from __future__ import annotations

import lightgbm as lgb
import numpy as np
import pandas as pd


DEFAULT_LGB_FEATURES = [
    "demand_forecast_mw",
    "demand_lag_1h",
    "demand_lag_24h",
    "demand_lag_168h",
    "demand_rolling_24h_mean",
    "demand_rolling_7d_mean",
    "smp_lag_1h",
    "smp_lag_2h",
    "smp_lag_3h",
    "smp_lag_24h",
    "smp_lag_48h",
    "smp_lag_168h",
    "smp_rolling_24h_mean",
    "smp_rolling_24h_std",
    "smp_rolling_7d_mean",
    "smp_rolling_7d_std",
    "hour",
    "hour_sin",
    "hour_cos",
    "day_of_week",
    "dow_sin",
    "dow_cos",
    "month",
    "is_weekend",
    "is_holiday",
    "is_summer",
    "is_winter",
    "is_peak_load_season",
]


class LightGBMModel:
    name = "lightgbm"

    def __init__(
        self,
        feature_cols: list[str] | None = None,
        params: dict | None = None,
        num_boost_round: int = 800,
        early_stopping_rounds: int = 50,
        seed: int = 42,
    ) -> None:
        self.feature_cols = feature_cols
        self.params = {
            "objective": "regression",
            "metric": "rmse",
            "learning_rate": 0.05,
            "num_leaves": 64,
            "min_data_in_leaf": 50,
            "feature_fraction": 0.9,
            "bagging_fraction": 0.9,
            "bagging_freq": 5,
            "verbosity": -1,
            "seed": seed,
            **(params or {}),
        }
        self.num_boost_round = num_boost_round
        self.early_stopping_rounds = early_stopping_rounds
        self.booster: lgb.Booster | None = None
        self.used_features: list[str] = []

    def _select_features(self, X: pd.DataFrame) -> list[str]:
        if self.feature_cols is not None:
            missing = [c for c in self.feature_cols if c not in X.columns]
            if missing:
                raise KeyError(f"LightGBM: missing requested features {missing}")
            return list(self.feature_cols)
        return [c for c in DEFAULT_LGB_FEATURES if c in X.columns]

    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        *,
        X_valid: pd.DataFrame | None = None,
        y_valid: pd.Series | None = None,
    ) -> "LightGBMModel":
        self.used_features = self._select_features(X)
        train_set = lgb.Dataset(
            X[self.used_features].astype(float), label=np.asarray(y, dtype=float)
        )
        valid_sets = [train_set]
        valid_names = ["train"]
        if X_valid is not None and y_valid is not None:
            valid_sets.append(
                lgb.Dataset(
                    X_valid[self.used_features].astype(float),
                    label=np.asarray(y_valid, dtype=float),
                    reference=train_set,
                )
            )
            valid_names.append("valid")
        callbacks = []
        if X_valid is not None:
            callbacks.append(lgb.early_stopping(self.early_stopping_rounds, verbose=False))
        callbacks.append(lgb.log_evaluation(period=0))
        self.booster = lgb.train(
            self.params,
            train_set,
            num_boost_round=self.num_boost_round,
            valid_sets=valid_sets,
            valid_names=valid_names,
            callbacks=callbacks,
        )
        return self

    def predict(self, X: pd.DataFrame) -> pd.Series:
        if self.booster is None:
            raise RuntimeError("LightGBMModel.fit() has not been called yet")
        preds = self.booster.predict(X[self.used_features].astype(float))
        return pd.Series(preds, index=X.index, name="prediction")

    def feature_importance(self) -> pd.DataFrame:
        if self.booster is None:
            raise RuntimeError("LightGBMModel.fit() has not been called yet")
        return pd.DataFrame(
            {
                "feature": self.used_features,
                "importance_gain": self.booster.feature_importance(importance_type="gain"),
                "importance_split": self.booster.feature_importance(importance_type="split"),
            }
        ).sort_values("importance_gain", ascending=False)
