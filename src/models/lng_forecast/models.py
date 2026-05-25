"""Two forecaster classes mirroring the PDF guide §6.

Both wrap fit/predict with a common interface so the CV loop can swap them.
The MLP uses sklearn's MLPRegressor (small, dependency-free) — it satisfies
the PDF's "ANN, polished comparison baseline" intent without requiring a
TensorFlow install.

Per-fold scaler discipline (PDF §4.4): `_PerFoldScaler` is intentionally
RE-FITTED inside each fold's `fit(X_train, y_train)`. Calling `transform`
on validation data therefore uses the train fold's statistics only.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler

# Use the project's existing LightGBM wrapper for hyperparameter consistency.
import lightgbm as lgb


class LightGBMForecaster:
    """LightGBM regressor with PDF-recommended hyperparameters.

    Tree models are scale-invariant — no scaler fit needed.
    """

    name = "lng_lightgbm"

    def __init__(
        self,
        n_estimators: int = 800,
        learning_rate: float = 0.03,
        num_leaves: int = 31,
        min_data_in_leaf: int = 10,
        feature_fraction: float = 0.9,
        bagging_fraction: float = 0.9,
        bagging_freq: int = 1,
        seed: int = 42,
        early_stopping_rounds: int = 100,
    ) -> None:
        self.params = dict(
            objective="regression",
            metric="rmse",
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            num_leaves=num_leaves,
            min_data_in_leaf=min_data_in_leaf,
            feature_fraction=feature_fraction,
            bagging_fraction=bagging_fraction,
            bagging_freq=bagging_freq,
            verbose=-1,
            seed=seed,
        )
        self.early_stopping_rounds = early_stopping_rounds
        self.model: lgb.LGBMRegressor | None = None
        self.used_features: list[str] = []

    def fit(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_valid: pd.DataFrame | None = None,
        y_valid: pd.Series | None = None,
        inner_valid_frac: float = 0.15,
        min_train_rows_for_es: int = 30,
    ) -> "LightGBMForecaster":
        """Fit LightGBM.

        IMPORTANT — outer-CV leakage avoidance: the ``X_valid`` / ``y_valid``
        kwargs in this signature are kept ONLY for interface compatibility
        with sklearn-style fit; **they are deliberately ignored** for early
        stopping. Using the outer CV's evaluation fold as the
        ``eval_set`` would let ``early_stopping`` pick the iteration that
        minimises loss on the very rows we later report metrics on — a
        classic 'validation labels in training' leak the PDF guide §8
        understates by reusing the same fold.

        Instead, we carve an INNER validation slice from the chronological
        tail of ``X_train`` (last ``inner_valid_frac``) and use it for
        ``early_stopping``. The outer valid fold remains completely held
        out until ``predict`` time.

        For very small folds (< ``min_train_rows_for_es``) we skip early
        stopping entirely and use a fixed ``n_estimators`` — this keeps
        the smallest folds reproducible and avoids spurious early-stop
        decisions on noisy 20-row tails.
        """
        self.used_features = list(X_train.columns)

        if len(X_train) >= min_train_rows_for_es:
            split = int(len(X_train) * (1.0 - inner_valid_frac))
            split = max(1, min(split, len(X_train) - 2))
            X_inner_tr = X_train.iloc[:split]
            y_inner_tr = y_train.iloc[:split]
            X_inner_va = X_train.iloc[split:]
            y_inner_va = y_train.iloc[split:]
            es_model = lgb.LGBMRegressor(**self.params)
            es_model.fit(
                X_inner_tr, y_inner_tr,
                eval_set=[(X_inner_va, y_inner_va)],
                eval_metric="rmse",
                callbacks=[
                    lgb.early_stopping(self.early_stopping_rounds, verbose=False),
                    lgb.log_evaluation(period=0),
                ],
            )
            best_iter = max(1, int(getattr(es_model, "best_iteration_", None)
                                   or self.params["n_estimators"]))
            # Refit on the FULL train fold at the inner-CV-selected iteration
            # count. This way the outer valid fold has never been seen.
            final_params = {**self.params, "n_estimators": best_iter}
            self.model = lgb.LGBMRegressor(**final_params)
            self.model.fit(X_train, y_train,
                           callbacks=[lgb.log_evaluation(period=0)])
        else:
            # Train fold too small for a sensible inner split — fall back to
            # the configured n_estimators with no early stopping.
            self.model = lgb.LGBMRegressor(**self.params)
            self.model.fit(X_train, y_train,
                           callbacks=[lgb.log_evaluation(period=0)])
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("LightGBMForecaster.fit() not called")
        return self.model.predict(X)

    def feature_importance(self) -> pd.DataFrame:
        if self.model is None:
            raise RuntimeError("LightGBMForecaster.fit() not called")
        booster = self.model.booster_
        return pd.DataFrame({
            "feature": self.used_features,
            "gain": booster.feature_importance(importance_type="gain"),
            "split": booster.feature_importance(importance_type="split"),
        }).sort_values("gain", ascending=False)


class MLPForecaster:
    """Small MLP — PDF guide §6.2 "ANN" counterpart.

    Per-fold scaler is fit in `fit()`. `predict()` reuses the same scaler
    on the same row order.

    Default hyperparameters tuned for ~150-200 monthly rows (small data):
      * shallow (one hidden layer, 32 units) to fight overfitting
      * adam + early_stopping (10% validation slice) to halt automatically
      * fixed seed for reproducibility
    """

    name = "lng_mlp"

    def __init__(
        self,
        hidden_layer_sizes: tuple[int, ...] = (64, 32),
        learning_rate_init: float = 1e-3,
        alpha: float = 1e-3,
        max_iter: int = 1000,
        random_state: int = 42,
    ) -> None:
        self.hidden_layer_sizes = hidden_layer_sizes
        self.learning_rate_init = learning_rate_init
        self.alpha = alpha
        self.max_iter = max_iter
        self.random_state = random_state
        self.scaler: StandardScaler | None = None
        self.model: MLPRegressor | None = None
        self.used_features: list[str] = []

    def fit(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_valid: pd.DataFrame | None = None,
        y_valid: pd.Series | None = None,
    ) -> "MLPForecaster":
        # PER-FOLD SCALER FIT — fit on train ONLY, transform both.
        self.used_features = list(X_train.columns)
        self.scaler = StandardScaler().fit(X_train.values)
        X_train_s = self.scaler.transform(X_train.values)
        self.model = MLPRegressor(
            hidden_layer_sizes=self.hidden_layer_sizes,
            activation="relu",
            solver="adam",
            learning_rate_init=self.learning_rate_init,
            alpha=self.alpha,
            max_iter=self.max_iter,
            random_state=self.random_state,
            early_stopping=True,
            validation_fraction=0.10,  # internal early-stopping slice
            n_iter_no_change=20,
        )
        self.model.fit(X_train_s, y_train.values)
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if self.model is None or self.scaler is None:
            raise RuntimeError("MLPForecaster.fit() not called")
        return self.model.predict(self.scaler.transform(X.values))


class NaiveLagForecaster:
    """Per-PDF baseline: prediction = LNG(M) (= lag1 column at row M)."""

    name = "lng_naive_lag1"

    def __init__(self, lag1_col: str = "lng_price_usd_per_mmbtu_lag1") -> None:
        self.lag1_col = lag1_col

    def fit(self, X_train, y_train, X_valid=None, y_valid=None) -> "NaiveLagForecaster":
        if self.lag1_col not in X_train.columns:
            raise KeyError(f"NaiveLag needs '{self.lag1_col}' in X")
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return X[self.lag1_col].to_numpy(dtype=float)
