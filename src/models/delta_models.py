"""Delta-target wrappers — learn (target − smp_t_observed) and reconstruct.

For a 1-month horizon forecast we have:

    target = SMP at M+1
    smp_t_observed = SMP at M (the last observed value at forecast origin)
    y_delta_1m = target − smp_t_observed

A delta model fits ``y_delta_1m`` from explanatory features (excluding
smp_t_observed itself, which has been absorbed into the target), then
reconstructs the SMP prediction by adding smp_t_observed back:

    reconstructed_prediction = smp_t_observed + predicted_delta_1m

The persistence baseline corresponds to ``predicted_delta = 0``. A delta
model that has any genuine signal MUST beat ``persistence_monthly`` on
the reconstructed-price MAE; otherwise it has learnt only noise.

Each wrapper:
  * `.fit(X, y)` — y is the level SMP target; we form the delta target
    internally, drop smp_t_observed from X, and fit the underlying model
  * `.predict(X)` — returns reconstructed level prediction (KRW/kWh)
  * `.predict_delta(X)` — returns the raw delta prediction
  * `.delta_target(X, y)` — exposes the delta target for evaluation

Compatible base models: any class with sklearn-style fit/predict on a
pandas DataFrame. Currently:

    delta_ridge      → RidgeModel
    delta_lightgbm   → LightGBMModel
    delta_ar_ridge   → MonthlyARRidge
"""

from __future__ import annotations

from typing import Callable

import pandas as pd

from src.models.ar_monthly import MonthlyARRidge
from src.models.lightgbm_model import LightGBMModel
from src.models.ridge_model import RidgeModel


OBSERVED_COL = "smp_t_observed"


class _DeltaWrapped:
    """Common machinery for the three delta variants."""

    name: str = "delta_base"

    def __init__(self, base_factory: Callable[[], object]) -> None:
        self._base_factory = base_factory
        self.base = base_factory()
        self.used_features: list[str] = []

    # ---- target transformation ----------------------------------------
    @staticmethod
    def delta_target(X: pd.DataFrame, y: pd.Series) -> pd.Series:
        if OBSERVED_COL not in X.columns:
            raise KeyError(
                f"Delta models require '{OBSERVED_COL}' in X to compute "
                f"y_delta_1m = target − {OBSERVED_COL}."
            )
        return y.to_numpy() - X[OBSERVED_COL].to_numpy()

    def _feature_view(self, X: pd.DataFrame) -> pd.DataFrame:
        """Return X without smp_t_observed (it's absorbed into the target)."""
        return X.drop(columns=[OBSERVED_COL]) if OBSERVED_COL in X.columns else X

    # ---- fit / predict --------------------------------------------------
    def fit(self, X: pd.DataFrame, y: pd.Series) -> "_DeltaWrapped":
        if OBSERVED_COL not in X.columns:
            raise KeyError(
                f"Delta models require '{OBSERVED_COL}' in X. Run "
                f"build_smp_monthly_features (round 5+) to expose it."
            )
        y_delta = pd.Series(self.delta_target(X, y), index=X.index, name="delta")
        feat_view = self._feature_view(X)
        self.base.fit(feat_view, y_delta)
        self.used_features = getattr(self.base, "used_features", list(feat_view.columns))
        return self

    def predict_delta(self, X: pd.DataFrame) -> pd.Series:
        feat_view = self._feature_view(X)
        delta = self.base.predict(feat_view)
        return pd.Series(delta, index=X.index, name="predicted_delta_1m")

    def predict(self, X: pd.DataFrame) -> pd.Series:
        if OBSERVED_COL not in X.columns:
            raise KeyError(
                f"Delta models need '{OBSERVED_COL}' at predict time to "
                f"reconstruct the level forecast."
            )
        delta = self.predict_delta(X)
        recon = X[OBSERVED_COL].to_numpy() + delta.to_numpy()
        return pd.Series(recon, index=X.index, name="prediction")


def _strip_observed(cols: list[str] | None) -> list[str] | None:
    """Drop smp_t_observed from a user-supplied feature_cols list — the
    delta wrapper absorbs it into the target, so the base model must not
    see it again."""
    if cols is None:
        return None
    return [c for c in cols if c != OBSERVED_COL]


class DeltaRidge(_DeltaWrapped):
    name = "delta_ridge"

    def __init__(self, alpha: float = 1.0, feature_cols: list[str] | None = None) -> None:
        cleaned = _strip_observed(feature_cols)
        super().__init__(lambda: RidgeModel(alpha=alpha, feature_cols=cleaned))


class DeltaLightGBM(_DeltaWrapped):
    name = "delta_lightgbm"

    def __init__(self, feature_cols: list[str] | None = None, **kw) -> None:
        cleaned = _strip_observed(feature_cols)
        super().__init__(lambda: LightGBMModel(feature_cols=cleaned, **kw))


class DeltaARRidge(_DeltaWrapped):
    name = "delta_ar_ridge"

    def __init__(self, alpha: float = 1.0, feature_cols: list[str] | None = None) -> None:
        cleaned = _strip_observed(feature_cols)
        super().__init__(lambda: MonthlyARRidge(alpha=alpha, feature_cols=cleaned))


# ---------------------------------------------------------------------------
# Delta-evaluation helpers
# ---------------------------------------------------------------------------

def compute_delta_metrics(
    y_true_level: pd.Series,
    y_pred_level: pd.Series,
    smp_observed: pd.Series,
) -> dict:
    """Compute the metric set the spec asks for, on top of compute_metrics.

    Returns:
        mae_level        — MAE on reconstructed SMP
        rmse_level       — RMSE on reconstructed SMP
        mape_level       — MAPE on reconstructed SMP
        delta_mae        — MAE on the delta (target_delta vs predicted_delta)
        delta_direction_accuracy — fraction of rows where sign(true_delta)
                                   == sign(predicted_delta)
    """
    import numpy as np
    from src.models.metrics import compute_metrics

    level = compute_metrics(y_true_level, y_pred_level)
    true_delta = y_true_level.to_numpy() - smp_observed.to_numpy()
    pred_delta = y_pred_level.to_numpy() - smp_observed.to_numpy()
    delta_mae = float(np.mean(np.abs(true_delta - pred_delta)))

    # Sign-agreement: NaN target deltas or zero true delta rows are excluded
    # from the denominator so the metric is well-defined.
    mask = (true_delta != 0)
    if mask.sum() == 0:
        direction = float("nan")
    else:
        direction = float(
            (np.sign(true_delta[mask]) == np.sign(pred_delta[mask])).mean()
        )
    return {
        "mae_level": level.mae,
        "rmse_level": level.rmse,
        "mape_level": level.mape,
        "delta_mae": delta_mae,
        "delta_direction_accuracy": direction,
    }
