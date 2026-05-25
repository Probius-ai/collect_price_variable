"""LightGBM regressor for SMP forecasting."""

from __future__ import annotations

import lightgbm as lgb
import numpy as np
import pandas as pd


DEFAULT_LGB_FEATURES_HOURLY = [
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

# Monthly defaults mirror the structure of DEFAULT_RIDGE_FEATURES_MONTHLY but
# include the full set of round-3 optional-but-lagged features (capacity /
# transaction / market-price) which Ridge intentionally omits. LightGBM
# tolerates NaN and handles many sparse columns far better than Ridge, so
# giving it the full menu is fine — and was the root cause of LightGBM
# previously training on only `month`.
DEFAULT_LGB_FEATURES_MONTHLY = [
    # SMP autoregressive — smp_t_observed (SMP at issue date M) is THE
    # dominant signal for forecasting M+1; it is observed before we
    # forecast and therefore not a leak. Including it lets LightGBM behave
    # like a true persistence baseline plus residual learning.
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
    # Calendar / cyclic
    "month",
    "quarter",
    "month_sin",
    "month_cos",
    "is_summer",
    "is_winter",
    "is_peak_season",
    # NB: removed `is_lng_shock_period` (post-hoc / look-ahead leakage).
    # LNG leading-indicator (round 6 — solar_beam DB → file, FRED JKM Asia)
    # See ridge_model.py for the publish-timing discipline: LNG(M) is NOT
    # observable at SMP info_cutoff=end-of-M, so only lagged columns are used.
    "lng_price_usd_per_mmbtu_lag_1m",
    "lng_price_usd_per_mmbtu_lag_2m",
    "lng_price_chg_1m_lag_1m",
    # JKM daily-derived monthly volatility (round 8).
    "jkm_daily_mean_lag_1m",
    "jkm_daily_std_lag_1m",
    "jkm_daily_range_lag_1m",
    "jkm_daily_last_lag_1m",
    # Settlement (lag_1m, full canonical fuel set)
    "settlement_unit_price_coal_anthracite_lag_1m",
    "settlement_unit_price_coal_bituminous_lag_1m",
    "settlement_unit_price_coal_total_lag_1m",
    "settlement_unit_price_lng_lag_1m",
    "settlement_unit_price_nuclear_lag_1m",
    "settlement_unit_price_oil_lag_1m",
    "settlement_unit_price_pumped_storage_lag_1m",
    "settlement_unit_price_renewable_lag_1m",
    "settlement_unit_price_other_lag_1m",
    "settlement_unit_price_total_lag_1m",
    # Capacity — fuel breakdown (lag_1m)
    "capacity_fuel_nuclear_mw_lag_1m",
    "capacity_fuel_lng_mw_lag_1m",
    "capacity_fuel_coal_bituminous_mw_lag_1m",
    "capacity_fuel_coal_total_mw_lag_1m",
    "capacity_fuel_renewable_total_mw_lag_1m",
    "capacity_fuel_total_mw_lag_1m",
    "capacity_fuel_nuclear_share_lag_1m",
    "capacity_fuel_lng_share_lag_1m",
    "capacity_fuel_renewable_total_share_lag_1m",
    # Capacity — generation type (lag_1m)
    "capacity_type_nuclear_mw_lag_1m",
    "capacity_type_steam_total_mw_lag_1m",
    "capacity_type_combined_cycle_total_mw_lag_1m",
    "capacity_type_renewable_mw_lag_1m",
    "capacity_type_total_mw_lag_1m",
    # Capacity — yearly broadcast (lag_1y)
    "capacity_yearly_nuclear_mw_lag_1y",
    "capacity_yearly_renewable_mw_lag_1y",
    "capacity_yearly_total_mw_lag_1y",
    # Transaction — volume + amount + derived price (lag_1m)
    "transaction_volume_total_mwh_lag_1m",
    "transaction_volume_lng_share_lag_1m",
    "transaction_volume_renewable_share_lag_1m",
    "transaction_amount_total_krw_lag_1m",
    "market_trade_price_lng_krw_per_kwh_lag_1m",
    "market_trade_price_coal_total_krw_per_kwh_lag_1m",
]

# Backward-compat alias — older code/tests may import DEFAULT_LGB_FEATURES
# expecting the hourly list. Keep it pointing at the hourly defaults so
# nothing breaks; the new code path uses the granularity-aware selector.
DEFAULT_LGB_FEATURES = DEFAULT_LGB_FEATURES_HOURLY


class LightGBMModel:
    name = "lightgbm"

    DEFAULT_NUM_BOOST_ROUND = 800

    def __init__(
        self,
        feature_cols: list[str] | None = None,
        params: dict | None = None,
        num_boost_round: int | None = None,
        early_stopping_rounds: int = 50,
        seed: int = 42,
    ) -> None:
        self.feature_cols = feature_cols
        # Track which keys the user explicitly set so the small-dataset
        # auto-scaling in fit() only overrides values the user did NOT pin.
        self._user_set_keys = set((params or {}).keys())
        # `num_boost_round=None` is the sentinel meaning "use the default and
        # allow auto-scaling on small data". A concrete integer means the
        # user is pinning the value and small-data auto-scaling must NOT
        # silently cap it.
        self._user_set_num_boost_round = num_boost_round is not None
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
        self.num_boost_round = (
            num_boost_round if num_boost_round is not None
            else self.DEFAULT_NUM_BOOST_ROUND
        )
        self.early_stopping_rounds = early_stopping_rounds
        self.booster: lgb.Booster | None = None
        self.used_features: list[str] = []
        # Filled in `fit()` when a validation set is supplied. Schema:
        #   {dataset_name: {metric_name: [value_at_iter_0, ...]}}
        # E.g. {'train': {'rmse': [...]}, 'valid': {'rmse': [...]}}
        # Used by the MLOps smoke test to log proper per-iteration
        # learning curves to MLflow (vs the single end-of-fit point that
        # non-iterative models produce).
        self.eval_history: dict[str, dict[str, list[float]]] = {}

    def _select_features(self, X: pd.DataFrame) -> list[str]:
        if self.feature_cols is not None:
            missing = [c for c in self.feature_cols if c not in X.columns]
            if missing:
                raise KeyError(f"LightGBM: missing requested features {missing}")
            return list(self.feature_cols)
        # Auto-detect granularity. smp_t_observed counts as a monthly marker
        # too — see RidgeModel._select_features for the round-5 reasoning.
        is_monthly = (
            "smp_t_observed" in X.columns
            or any(
                c.endswith("_lag_1m") or c.endswith("_lag_12m") or c.endswith("_lag_1y")
                for c in X.columns
            )
        )
        default_pool = (
            DEFAULT_LGB_FEATURES_MONTHLY if is_monthly else DEFAULT_LGB_FEATURES_HOURLY
        )
        chosen = [c for c in default_pool if c in X.columns]
        if chosen:
            return chosen
        # Fallback: use every column we were given. LightGBM tolerates
        # arbitrary numeric inputs and NaNs.
        return list(X.columns)

    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        *,
        X_valid: pd.DataFrame | None = None,
        y_valid: pd.Series | None = None,
    ) -> "LightGBMModel":
        self.used_features = self._select_features(X)
        # Auto-scale tree-size params for small monthly datasets. With the
        # default min_data_in_leaf=50 on ~120 train rows, the tree can fit
        # at most 2-3 leaves and gradient boosting collapses to a constant
        # predictor (this was the round-3 LightGBM failure mode). If the
        # user did not explicitly pin these params and training data is
        # small, fall back to sane small-data values.
        if len(X) < 500:
            overrides = {}
            # Empirically tuned for ~120 monthly rows: min_data_in_leaf=10
            # and num_leaves=15 keep the tree shallow enough to avoid
            # memorising. Larger values led to LightGBM training to ~2 MAE
            # while test MAE stayed at 17 (overfit).
            if "min_data_in_leaf" not in self._user_set_keys:
                overrides["min_data_in_leaf"] = max(10, len(X) // 15)
            if "num_leaves" not in self._user_set_keys:
                overrides["num_leaves"] = max(7, min(15, len(X) // 12))
            if "learning_rate" not in self._user_set_keys:
                overrides["learning_rate"] = 0.03   # slower learn for stability
            if "feature_fraction" not in self._user_set_keys:
                overrides["feature_fraction"] = 0.6  # stronger column sampling
            if overrides:
                self.params = {**self.params, **overrides}
            # Cap num_boost_round too — 800 rounds on 120 rows is wasteful and
            # also makes walk-forward CV (150 refits) painfully slow. But
            # ONLY when the user did not explicitly pin num_boost_round.
            if not self._user_set_num_boost_round:
                self.num_boost_round = min(self.num_boost_round, 200)
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
        # Capture per-iteration metrics so the MLOps smoke test can log
        # them to MLflow as a learning curve. Reset between fits so a
        # second `.fit()` doesn't pile onto stale data.
        self.eval_history = {}
        callbacks.append(lgb.record_evaluation(self.eval_history))
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
