"""Iterative-training models for the MLOps smoke test.

These models expose a per-iteration / per-epoch learning curve (via the
common `eval_history` attribute) so the smoke test can log them to
MLflow and the user sees real convergence lines in the Compare-runs
view — not single-point bars.

Models in this module:
  * `MLPMonthlyModel`    — sklearn MLPRegressor with SimpleImputer/Scaler
                           pipeline (fixes the "Input X contains NaN"
                           failure mode of the prior MLPForecaster on
                           the SMP monthly panel)
  * `XGBoostMonthlyModel` — XGBoost regressor with `evals_result` capture
  * `TorchMLPModel`      — small feed-forward net in PyTorch, full
                           per-epoch loss tracking
  * `TorchLSTMModel`     — LSTM over `smp_lag_{1..12}m` lag features as
                           a 12-step sequence; per-epoch loss tracking

Each model's `fit()` accepts optional `X_valid`/`y_valid` and, when
present, logs the validation metric (RMSE) per iteration/epoch under
`eval_history['valid']`. The smoke test's `_extract_learning_curve`
prefers the `valid` series; falls back to `train` when no validation
set is supplied.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler


# ---------------------------------------------------------------------------
# 1) sklearn MLP for the monthly panel — the one we couldn't run before
#    because the panel has NaN columns.
# ---------------------------------------------------------------------------

class MLPMonthlyModel:
    """sklearn MLPRegressor with NaN imputation for the SMP monthly panel.

    The prior `MLPForecaster` (in `lng_forecast.models`) was tuned for
    the LNG panel which is dense by construction. The SMP monthly panel
    has sparse columns (JKM lag features only available from 2013+,
    some capacity / settlement columns with publication gaps), so a
    plain `StandardScaler().fit(X)` blows up with "Input X contains NaN".

    Pipeline: SimpleImputer(median) → StandardScaler → MLPRegressor.
    """

    name = "mlp_monthly"
    DEFAULT_MAX_ITER = 500

    def __init__(
        self,
        hidden_layer_sizes: tuple[int, ...] = (64, 32),
        learning_rate_init: float = 1e-3,
        alpha: float = 1e-3,
        max_iter: int | None = None,
        random_state: int = 42,
    ) -> None:
        self.hidden_layer_sizes = hidden_layer_sizes
        self.learning_rate_init = learning_rate_init
        self.alpha = alpha
        self.max_iter = max_iter if max_iter is not None else self.DEFAULT_MAX_ITER
        self.random_state = random_state
        self.imputer: SimpleImputer | None = None
        self.scaler: StandardScaler | None = None
        self.model: MLPRegressor | None = None
        self.used_features: list[str] = []
        # Common surface for `_extract_learning_curve`.
        self.eval_history: dict[str, dict[str, list[float]]] = {}

    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        *,
        X_valid: pd.DataFrame | None = None,
        y_valid: pd.Series | None = None,
    ) -> "MLPMonthlyModel":
        self.used_features = list(X.columns)
        # Fit imputer + scaler on TRAIN ONLY (per-fold isolation) so the
        # transform never sees future statistics.
        self.imputer = SimpleImputer(strategy="median")
        X_imp = self.imputer.fit_transform(X.values)
        self.scaler = StandardScaler().fit(X_imp)
        X_s = self.scaler.transform(X_imp)

        # NB: sklearn's MLPRegressor early-stopping uses its OWN internal
        # validation slice (`validation_fraction=0.10`); we do not pass
        # X_valid directly. The user's holdout is for downstream metrics.
        self.model = MLPRegressor(
            hidden_layer_sizes=self.hidden_layer_sizes,
            activation="relu",
            solver="adam",
            learning_rate_init=self.learning_rate_init,
            alpha=self.alpha,
            max_iter=self.max_iter,
            random_state=self.random_state,
            early_stopping=True,
            validation_fraction=0.10,
            n_iter_no_change=20,
        )
        self.model.fit(X_s, y.values.astype(float))

        # Surface the per-epoch loss curve under the common shape so
        # `_extract_learning_curve` can find it without special-casing.
        # sklearn exposes `loss_curve_` (the TRAINING MSE per iteration).
        if hasattr(self.model, "loss_curve_"):
            train_loss = list(self.model.loss_curve_)
            # Convert MSE → RMSE for unit-consistency with LightGBM's `rmse`
            train_rmse = [float(np.sqrt(max(0.0, v))) for v in train_loss]
            self.eval_history = {"train": {"rmse": train_rmse}}
            # sklearn also has `validation_scores_` (the internal early-
            # stopping slice's R²). Log it as `valid_r2` so the user can
            # see both convergence views in MLflow.
            if hasattr(self.model, "validation_scores_") and self.model.validation_scores_:
                self.eval_history["valid"] = {
                    "r2": [float(v) for v in self.model.validation_scores_],
                }
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if self.model is None or self.scaler is None or self.imputer is None:
            raise RuntimeError("MLPMonthlyModel.fit() has not been called")
        X_imp = self.imputer.transform(X[self.used_features].values)
        X_s = self.scaler.transform(X_imp)
        return self.model.predict(X_s)


# ---------------------------------------------------------------------------
# 2) XGBoost — direct counterpart to LightGBM; eval_history shape identical
# ---------------------------------------------------------------------------

class XGBoostMonthlyModel:
    """XGBoost regressor — boosting iteration learning curve.

    Output `eval_history` has the same shape as LightGBM's
    (`{dataset_name: {metric_name: [values]}}`) so the existing
    extraction logic works without changes.
    """

    name = "xgboost"
    DEFAULT_NUM_BOOST_ROUND = 300

    def __init__(
        self,
        num_boost_round: int | None = None,
        early_stopping_rounds: int = 50,
        learning_rate: float = 0.05,
        max_depth: int = 4,
        subsample: float = 0.9,
        colsample_bytree: float = 0.9,
        seed: int = 42,
    ) -> None:
        self.num_boost_round = num_boost_round or self.DEFAULT_NUM_BOOST_ROUND
        self.early_stopping_rounds = early_stopping_rounds
        self.params = {
            "objective": "reg:squarederror",
            "eval_metric": "rmse",
            "learning_rate": learning_rate,
            "max_depth": max_depth,
            "subsample": subsample,
            "colsample_bytree": colsample_bytree,
            "verbosity": 0,
            "seed": seed,
        }
        self.booster: Any = None
        self.used_features: list[str] = []
        self.eval_history: dict[str, dict[str, list[float]]] = {}

    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        *,
        X_valid: pd.DataFrame | None = None,
        y_valid: pd.Series | None = None,
    ) -> "XGBoostMonthlyModel":
        import xgboost as xgb

        self.used_features = list(X.columns)
        # Median-impute NaN — XGBoost handles missing natively (using
        # `missing=nan`) but the imputer keeps results comparable with
        # the other models that don't.
        X_train = X.astype(float).fillna(X.median(numeric_only=True))
        y_train_np = y.astype(float).to_numpy()
        dtrain = xgb.DMatrix(X_train.values, label=y_train_np)

        evals = [(dtrain, "train")]
        if X_valid is not None and y_valid is not None:
            X_v = X_valid[self.used_features].astype(float).fillna(
                X.median(numeric_only=True)
            )
            dvalid = xgb.DMatrix(X_v.values, label=y_valid.astype(float).to_numpy())
            evals.append((dvalid, "valid"))

        self.eval_history = {}
        kwargs: dict[str, Any] = {
            "evals": evals,
            "evals_result": self.eval_history,
            "verbose_eval": False,
        }
        if X_valid is not None:
            kwargs["early_stopping_rounds"] = self.early_stopping_rounds

        self.booster = xgb.train(
            self.params,
            dtrain,
            num_boost_round=self.num_boost_round,
            **kwargs,
        )
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        import xgboost as xgb
        if self.booster is None:
            raise RuntimeError("XGBoostMonthlyModel.fit() has not been called")
        X_imp = X[self.used_features].astype(float).fillna(0.0)
        return self.booster.predict(xgb.DMatrix(X_imp.values))


# ---------------------------------------------------------------------------
# 3) PyTorch feed-forward MLP — full control over the training loop so
#    we can log per-epoch loss properly.
# ---------------------------------------------------------------------------

class TorchMLPModel:
    """Two-layer feed-forward net in PyTorch with per-epoch loss tracking.

    Why a second MLP when sklearn's is already here? Two reasons:
      1. Educational — shows the explicit training loop and how `eval_history`
         is populated step-by-step (vs sklearn's opaque `loss_curve_`).
      2. Clean per-epoch RMSE on BOTH train and validation (sklearn only
         tracks training loss + an R² on its internal early-stopping slice).
    """

    name = "torch_mlp"
    DEFAULT_EPOCHS = 300

    def __init__(
        self,
        hidden_size: int = 64,
        n_layers: int = 2,
        epochs: int | None = None,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-4,
        seed: int = 42,
        device: str = "cpu",  # tiny dataset — CPU is faster than GPU spin-up
    ) -> None:
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.epochs = epochs or self.DEFAULT_EPOCHS
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.seed = seed
        self.device = device
        self.imputer: SimpleImputer | None = None
        self.scaler: StandardScaler | None = None
        self.model: Any = None
        self.used_features: list[str] = []
        self.eval_history: dict[str, dict[str, list[float]]] = {}

    def _build(self, n_features: int) -> Any:
        import torch.nn as nn
        layers: list[Any] = []
        in_dim = n_features
        for _ in range(self.n_layers):
            layers += [nn.Linear(in_dim, self.hidden_size), nn.ReLU()]
            in_dim = self.hidden_size
        layers += [nn.Linear(in_dim, 1)]
        return nn.Sequential(*layers)

    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        *,
        X_valid: pd.DataFrame | None = None,
        y_valid: pd.Series | None = None,
    ) -> "TorchMLPModel":
        import torch

        torch.manual_seed(self.seed)
        self.used_features = list(X.columns)
        self.imputer = SimpleImputer(strategy="median")
        X_imp = self.imputer.fit_transform(X.values)
        self.scaler = StandardScaler().fit(X_imp)
        X_s = self.scaler.transform(X_imp)

        X_t = torch.tensor(X_s, dtype=torch.float32, device=self.device)
        y_t = torch.tensor(y.values.astype(float), dtype=torch.float32,
                           device=self.device).view(-1, 1)

        X_v_t = y_v_t = None
        if X_valid is not None and y_valid is not None:
            X_v_imp = self.imputer.transform(X_valid[self.used_features].values)
            X_v_s = self.scaler.transform(X_v_imp)
            X_v_t = torch.tensor(X_v_s, dtype=torch.float32, device=self.device)
            y_v_t = torch.tensor(y_valid.values.astype(float),
                                 dtype=torch.float32,
                                 device=self.device).view(-1, 1)

        self.model = self._build(X_s.shape[1]).to(self.device)
        optim = torch.optim.Adam(
            self.model.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )
        loss_fn = torch.nn.MSELoss()

        train_rmse: list[float] = []
        valid_rmse: list[float] = []
        # Full-batch training (monthly panel is tiny, ~100-200 rows).
        for _epoch in range(self.epochs):
            self.model.train()
            optim.zero_grad()
            pred = self.model(X_t)
            loss = loss_fn(pred, y_t)
            loss.backward()
            optim.step()
            train_rmse.append(float(loss.item() ** 0.5))

            if X_v_t is not None:
                self.model.eval()
                with torch.no_grad():
                    v_loss = loss_fn(self.model(X_v_t), y_v_t).item()
                valid_rmse.append(float(v_loss ** 0.5))

        self.eval_history = {"train": {"rmse": train_rmse}}
        if valid_rmse:
            self.eval_history["valid"] = {"rmse": valid_rmse}
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        import torch
        if self.model is None or self.scaler is None or self.imputer is None:
            raise RuntimeError("TorchMLPModel.fit() has not been called")
        X_imp = self.imputer.transform(X[self.used_features].values)
        X_s = self.scaler.transform(X_imp)
        X_t = torch.tensor(X_s, dtype=torch.float32, device=self.device)
        self.model.eval()
        with torch.no_grad():
            return self.model(X_t).cpu().numpy().reshape(-1)


# ---------------------------------------------------------------------------
# 4) PyTorch LSTM — process the `smp_lag_{1..12}m` columns as a 12-step
#    sequence and predict next-month SMP.
# ---------------------------------------------------------------------------

class TorchLSTMModel:
    """LSTM over the monthly SMP lag-feature sequence (lag_12m → lag_1m).

    Each row's `smp_lag_{12,11,...,1}m` columns are treated as a single
    12-timestep univariate sequence (oldest → newest). The LSTM
    ingests the sequence and emits a single regression output (the
    next month's SMP).

    Caveats for ~100-200 training samples:
      * LSTM is overkill — included for completeness / instructional value
      * `hidden_size=16` and only `n_layers=1` to fight overfit
      * `weight_decay` + dropout would help further but kept minimal
        for clarity
    """

    name = "torch_lstm"
    DEFAULT_EPOCHS = 300
    SEQUENCE_COLS_LAGS = list(range(1, 13))  # lag_1m through lag_12m

    def __init__(
        self,
        hidden_size: int = 16,
        n_layers: int = 1,
        epochs: int | None = None,
        learning_rate: float = 5e-3,
        weight_decay: float = 1e-3,
        seed: int = 42,
        device: str = "cpu",
    ) -> None:
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.epochs = epochs or self.DEFAULT_EPOCHS
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.seed = seed
        self.device = device
        self.imputer: SimpleImputer | None = None
        self.scaler: StandardScaler | None = None
        self.model: Any = None
        self.used_features: list[str] = []
        self.eval_history: dict[str, dict[str, list[float]]] = {}

    def _select_sequence_cols(self, X: pd.DataFrame) -> list[str]:
        # Oldest first (lag_12m) → newest (lag_1m). LSTM sees time
        # going forward through the sequence.
        cols = [f"smp_lag_{k}m" for k in reversed(self.SEQUENCE_COLS_LAGS)
                if f"smp_lag_{k}m" in X.columns]
        if len(cols) < 3:
            raise ValueError(
                "TorchLSTMModel needs ≥3 smp_lag_{k}m columns; got "
                f"{cols}. Run build_smp_monthly_features first."
            )
        return cols

    def _build(self) -> Any:
        import torch.nn as nn

        class _LSTMRegressor(nn.Module):
            def __init__(self, hidden_size: int, n_layers: int):
                super().__init__()
                self.lstm = nn.LSTM(
                    input_size=1, hidden_size=hidden_size,
                    num_layers=n_layers, batch_first=True,
                )
                self.head = nn.Linear(hidden_size, 1)

            def forward(self, x):
                # x: (batch, seq_len, 1) → use the LAST timestep's hidden state
                out, _ = self.lstm(x)
                return self.head(out[:, -1, :])

        return _LSTMRegressor(self.hidden_size, self.n_layers)

    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        *,
        X_valid: pd.DataFrame | None = None,
        y_valid: pd.Series | None = None,
    ) -> "TorchLSTMModel":
        import torch

        torch.manual_seed(self.seed)
        self.used_features = self._select_sequence_cols(X)
        X_seq = X[self.used_features].values  # shape: (n, seq_len)
        # Impute + scale on the flat (n, seq_len) view; rescaling per
        # timestep would distort the temporal pattern.
        self.imputer = SimpleImputer(strategy="median")
        X_imp = self.imputer.fit_transform(X_seq)
        self.scaler = StandardScaler().fit(X_imp)
        X_s = self.scaler.transform(X_imp)
        # Reshape to (n, seq_len, 1) for LSTM input
        X_t = torch.tensor(X_s, dtype=torch.float32,
                           device=self.device).unsqueeze(-1)
        y_t = torch.tensor(y.values.astype(float), dtype=torch.float32,
                           device=self.device).view(-1, 1)

        X_v_t = y_v_t = None
        if X_valid is not None and y_valid is not None:
            X_v_seq = X_valid[self.used_features].values
            X_v_imp = self.imputer.transform(X_v_seq)
            X_v_s = self.scaler.transform(X_v_imp)
            X_v_t = torch.tensor(X_v_s, dtype=torch.float32,
                                 device=self.device).unsqueeze(-1)
            y_v_t = torch.tensor(y_valid.values.astype(float),
                                 dtype=torch.float32,
                                 device=self.device).view(-1, 1)

        self.model = self._build().to(self.device)
        optim = torch.optim.Adam(
            self.model.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )
        loss_fn = torch.nn.MSELoss()

        train_rmse: list[float] = []
        valid_rmse: list[float] = []
        for _epoch in range(self.epochs):
            self.model.train()
            optim.zero_grad()
            pred = self.model(X_t)
            loss = loss_fn(pred, y_t)
            loss.backward()
            optim.step()
            train_rmse.append(float(loss.item() ** 0.5))
            if X_v_t is not None:
                self.model.eval()
                with torch.no_grad():
                    v_loss = loss_fn(self.model(X_v_t), y_v_t).item()
                valid_rmse.append(float(v_loss ** 0.5))

        self.eval_history = {"train": {"rmse": train_rmse}}
        if valid_rmse:
            self.eval_history["valid"] = {"rmse": valid_rmse}
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        import torch
        if self.model is None or self.scaler is None or self.imputer is None:
            raise RuntimeError("TorchLSTMModel.fit() has not been called")
        X_seq = X[self.used_features].values
        X_imp = self.imputer.transform(X_seq)
        X_s = self.scaler.transform(X_imp)
        X_t = torch.tensor(X_s, dtype=torch.float32,
                           device=self.device).unsqueeze(-1)
        self.model.eval()
        with torch.no_grad():
            return self.model(X_t).cpu().numpy().reshape(-1)
