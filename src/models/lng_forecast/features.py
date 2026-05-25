"""Feature engineering for the LNG monthly forecaster.

Per the PDF guide:
  * monthly panel of LNG + Dubai oil + USD/KRW
  * lag features (1, 2, 3, 6, 12 months)
  * rolling stats (3m mean, 6m mean, 6m std) — `shift(1).rolling(...)` so the
    window never includes the current month
  * pct-change with `fill_method=None` (round-6 lesson; pandas 2.x default
    pads NaN and fabricates 0-change at the grid edges)
  * calendar (month_sin/cos, quarter)
  * target = LNG(M+h) for forecast horizon h (default 1)

All features at row M are derived from values whose period_month ≤ M, with
no internal lookahead. The pipeline produces (X, y, period_months) such
that fitting on indices [a..b] of X and y guarantees no future leakage.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.features.build_monthly import _load_one_source_parquet


# Source IDs we expect on disk (loaded by `src.pipelines.load_files`).
_SRC_LNG = "lng_price_monthly_file"
_SRC_OIL = "oil_price_monthly_file"
_SRC_FX = "fx_usd_krw_monthly_file"


@dataclass
class PanelInfo:
    """Summary of the assembled LNG/oil/FX panel."""

    rows: int
    period_min: pd.Timestamp
    period_max: pd.Timestamp
    n_features: int
    horizon_months: int
    n_dropped_for_nan: int


def _load_monthly_series(source_id: str, value_col: str) -> pd.Series:
    """Read the latest snapshot for a monthly source and dedup on period_month."""
    df = _load_one_source_parquet(source_id)
    if df.empty:
        return pd.Series(dtype=float, name=value_col)
    df["period_month"] = pd.to_datetime(df["period_month"])
    if "collected_at" in df.columns:
        df = df.sort_values(["period_month", "collected_at"])
    else:
        df = df.sort_values("period_month")
    df = df.drop_duplicates(subset="period_month", keep="last")
    return df.set_index("period_month")[value_col].astype(float).sort_index()


def _build_monthly_panel() -> pd.DataFrame:
    """Inner join of LNG, oil, FX on period_month — monthly grid."""
    lng = _load_monthly_series(_SRC_LNG, "lng_price_usd_per_mmbtu")
    oil = _load_monthly_series(_SRC_OIL, "oil_price_usd_per_bbl")
    fx = _load_monthly_series(_SRC_FX, "usd_krw_monthly_avg")
    panel = pd.concat([lng, oil, fx], axis=1)
    panel.index.name = "period_month"
    return panel


def build_lng_forecast_features(
    horizon_months: int = 1,
    *,
    keep_dropna: bool = True,
) -> tuple[pd.DataFrame, pd.Series, pd.Series, PanelInfo]:
    """Assemble LNG/oil/FX features + target.

    Returns:
        X — feature matrix
        y — target = LNG(period_month + horizon_months)
        period_months — sortable index aligned with X/y rows
        info — PanelInfo metadata snapshot
    """
    if horizon_months <= 0:
        raise ValueError(f"horizon_months must be > 0 (got {horizon_months})")

    panel = _build_monthly_panel()
    if panel.empty:
        raise FileNotFoundError(
            "No LNG/oil/FX parsed parquet on disk. "
            "Run `python -m src.pipelines.load_files --source "
            "{lng,oil,fx}_*_monthly_file` first."
        )

    feats = pd.DataFrame(index=panel.index)

    # Lag + rolling features per macro variable.
    for col in ["lng_price_usd_per_mmbtu", "oil_price_usd_per_bbl",
                "usd_krw_monthly_avg"]:
        series = panel[col]
        # Lags: at row M, lag_L = value at month M-L.
        for L in [1, 2, 3, 6, 12]:
            feats[f"{col}_lag{L}"] = series.shift(L)
        # Rolling: shift(1) BEFORE rolling so the window covers [M-window..M-1].
        # NOT center=True (would include M+1..M+window/2 — future leak).
        feats[f"{col}_roll3_mean"] = series.shift(1).rolling(3).mean()
        feats[f"{col}_roll6_mean"] = series.shift(1).rolling(6).mean()
        feats[f"{col}_roll6_std"] = series.shift(1).rolling(6).std()
        # 1-period pct change of the (shifted) series — pandas 2.x default
        # fill_method='pad' would fabricate 0-change at grid edges; pass
        # fill_method=None explicitly (round-6 lesson).
        feats[f"{col}_ret1_lag1"] = series.pct_change(fill_method=None).shift(1)

    # Calendar features.
    feats["month"] = feats.index.month
    feats["month_sin"] = np.sin(2 * np.pi * feats["month"] / 12)
    feats["month_cos"] = np.cos(2 * np.pi * feats["month"] / 12)
    feats["quarter"] = feats.index.quarter

    # Cross-variable interaction (PDF example).
    feats["lng_oil_ratio_lag1"] = (
        panel["lng_price_usd_per_mmbtu"].shift(1)
        / panel["oil_price_usd_per_bbl"].shift(1)
    )

    # Target = LNG at (period_month + horizon_months).
    y = panel["lng_price_usd_per_mmbtu"].shift(-horizon_months)
    y.name = "lng_target"

    period_months = pd.Series(feats.index, index=feats.index, name="period_month")
    full = feats.join(y).join(period_months.to_frame())

    rows_before = len(full)
    if keep_dropna:
        full = full.dropna().reset_index(drop=True)

    X = full.drop(columns=["lng_target", "period_month"])
    y_out = full["lng_target"]
    pm = pd.to_datetime(full["period_month"])

    info = PanelInfo(
        rows=int(len(full)),
        period_min=pd.Timestamp(pm.min()),
        period_max=pd.Timestamp(pm.max()),
        n_features=int(X.shape[1]),
        horizon_months=horizon_months,
        n_dropped_for_nan=int(rows_before - len(full)),
    )
    return X, y_out, pm, info
