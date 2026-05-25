"""CV evaluation + RMSE/MAPE — PDF guide §7 and §8.

Reports fold-level RMSE/MAPE and aggregates (mean ± std). Persists raw
predictions per fold so a caller can plot residuals or rerun comparisons
without retraining.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np
import pandas as pd

from src.models.lng_forecast.split import ts_folds


def compute_rmse_mape(y_true: np.ndarray, y_pred: np.ndarray,
                      eps: float = 1e-8) -> tuple[float, float]:
    """RMSE on raw scale, MAPE in percent — PDF §7 formulas."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
    mape = float(np.mean(np.abs((y_true - y_pred) / np.maximum(np.abs(y_true), eps))) * 100)
    return rmse, mape


@dataclass
class FoldResult:
    fold: int
    train_n: int
    valid_n: int
    train_period: tuple[str, str]
    valid_period: tuple[str, str]
    rmse: float
    mape: float
    mae: float


@dataclass
class CVReport:
    model_name: str
    n_folds: int
    embargo: int
    folds: list[FoldResult] = field(default_factory=list)
    predictions: pd.DataFrame | None = None  # all valid-fold preds concatenated

    def aggregate(self) -> dict:
        rmses = np.array([f.rmse for f in self.folds])
        mapes = np.array([f.mape for f in self.folds])
        maes = np.array([f.mae for f in self.folds])
        return {
            "model": self.model_name,
            "n_folds": self.n_folds,
            "embargo": self.embargo,
            "rmse_mean": float(rmses.mean()),
            "rmse_std": float(rmses.std()),
            "mape_mean": float(mapes.mean()),
            "mape_std": float(mapes.std()),
            "mae_mean": float(maes.mean()),
            "mae_std": float(maes.std()),
        }


def run_cv(
    X: pd.DataFrame,
    y: pd.Series,
    period_months: pd.Series,
    model_factory: Callable[[], object],
    *,
    model_name: str,
    n_splits: int = 5,
    embargo: int = 1,
) -> CVReport:
    """Run TimeSeriesSplit CV over (X, y).

    `model_factory` is a callable returning a fresh model with `.fit(...)`
    and `.predict(...)`. We call it once per fold so each fold's model is
    independent (no parameter bleed between folds).
    """
    n = len(X)
    if n != len(y) or n != len(period_months):
        raise ValueError(
            f"length mismatch X={n}, y={len(y)}, periods={len(period_months)}"
        )

    fold_results: list[FoldResult] = []
    pred_rows: list[pd.DataFrame] = []
    for fi, (tr, va) in enumerate(ts_folds(n, n_splits=n_splits, embargo=embargo)):
        X_tr, X_va = X.iloc[tr], X.iloc[va]
        y_tr, y_va = y.iloc[tr], y.iloc[va]

        model = model_factory()
        # CRITICAL: the outer valid fold (X_va, y_va) MUST NOT be passed to
        # the model's fit(). Doing so would leak the evaluation labels into
        # early-stopping / hyperparameter selection, and the RMSE/MAPE we
        # then report on (X_va, y_va) would be optimistic.
        #
        # Forecasters that want early stopping should carve an inner
        # validation slice from X_tr (see LightGBMForecaster.fit's
        # `inner_valid_frac` parameter for the canonical pattern).
        model.fit(X_tr, y_tr)
        y_pred = np.asarray(model.predict(X_va), dtype=float)

        rmse, mape = compute_rmse_mape(y_va.values, y_pred)
        mae = float(np.mean(np.abs(y_va.values - y_pred)))

        fold_results.append(FoldResult(
            fold=fi + 1,
            train_n=int(len(tr)),
            valid_n=int(len(va)),
            train_period=(str(period_months.iloc[tr[0]])[:10],
                          str(period_months.iloc[tr[-1]])[:10]),
            valid_period=(str(period_months.iloc[va[0]])[:10],
                          str(period_months.iloc[va[-1]])[:10]),
            rmse=rmse, mape=mape, mae=mae,
        ))
        pred_rows.append(pd.DataFrame({
            "fold": fi + 1,
            "period_month": period_months.iloc[va].values,
            "y_true": y_va.values,
            "y_pred": y_pred,
        }))

    report = CVReport(
        model_name=model_name,
        n_folds=len(fold_results),
        embargo=embargo,
        folds=fold_results,
        predictions=pd.concat(pred_rows, ignore_index=True) if pred_rows else None,
    )
    return report
