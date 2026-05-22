"""Regression + directional + spike-recall metrics for SMP evaluation."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class ForecastMetrics:
    mae: float
    rmse: float
    mape: float
    smape: float
    r2: float
    directional_accuracy: float | None
    peak_precision: float | None
    peak_recall: float | None
    peak_f1: float | None
    peak_threshold: float | None
    n_observations: int

    def to_dict(self) -> dict[str, float | int | None]:
        return self.__dict__.copy()


def _safe_mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    mask = np.abs(y_true) > 1e-9
    if not mask.any():
        return float("nan")
    return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100.0)


def _smape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    denom = (np.abs(y_true) + np.abs(y_pred)) / 2.0
    mask = denom > 1e-9
    if not mask.any():
        return float("nan")
    return float(np.mean(np.abs(y_true[mask] - y_pred[mask]) / denom[mask]) * 100.0)


def _r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))
    if ss_tot < 1e-12:
        return float("nan")
    return 1.0 - ss_res / ss_tot


def _directional_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float | None:
    if y_true.shape[0] < 2:
        return None
    dy = np.sign(np.diff(y_true))
    dp = np.sign(np.diff(y_pred))
    return float(np.mean(dy == dp))


def _peak_scores(
    y_true: np.ndarray, y_pred: np.ndarray, percentile: float = 90.0
) -> tuple[float | None, float | None, float | None, float]:
    threshold = float(np.percentile(y_true, percentile))
    true_peak = y_true >= threshold
    pred_peak = y_pred >= threshold
    tp = int(np.sum(true_peak & pred_peak))
    fp = int(np.sum(~true_peak & pred_peak))
    fn = int(np.sum(true_peak & ~pred_peak))
    if tp + fp == 0 and tp + fn == 0:
        return None, None, None, threshold
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )
    return precision, recall, f1, threshold


def compute_metrics(
    y_true: pd.Series | np.ndarray,
    y_pred: pd.Series | np.ndarray,
    *,
    spike_percentile: float = 90.0,
) -> ForecastMetrics:
    yt = np.asarray(y_true, dtype=float)
    yp = np.asarray(y_pred, dtype=float)
    mask = ~(np.isnan(yt) | np.isnan(yp))
    yt = yt[mask]
    yp = yp[mask]
    if yt.size == 0:
        raise ValueError("No valid (non-NaN) prediction pairs to score")

    mae = float(np.mean(np.abs(yt - yp)))
    rmse = float(np.sqrt(np.mean((yt - yp) ** 2)))
    mape = _safe_mape(yt, yp)
    smape = _smape(yt, yp)
    r2 = _r2(yt, yp)
    da = _directional_accuracy(yt, yp)
    pp, pr, pf1, threshold = _peak_scores(yt, yp, percentile=spike_percentile)
    return ForecastMetrics(
        mae=mae,
        rmse=rmse,
        mape=mape,
        smape=smape,
        r2=r2,
        directional_accuracy=da,
        peak_precision=pp,
        peak_recall=pr,
        peak_f1=pf1,
        peak_threshold=threshold,
        n_observations=int(yt.size),
    )
