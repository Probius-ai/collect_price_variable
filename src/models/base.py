"""Tiny model protocol so every estimator (naive, Ridge, LightGBM) is callable
the same way from train.py / evaluate.py."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import pandas as pd


@runtime_checkable
class ForecastModel(Protocol):
    name: str

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "ForecastModel": ...

    def predict(self, X: pd.DataFrame) -> pd.Series: ...
