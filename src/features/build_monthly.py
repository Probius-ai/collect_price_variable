"""Monthly feature builder for the pre-approval MVP (round 2).

Schema change: monthly inputs are now LONG-format with explicit `source_id`,
`period_month`, and (for SMP) `area` or (for settlement) `fuel_type`. The
feature builder:

  1. Loads each source's parsed_*.parquet files.
  2. Concatenates across collection snapshots, de-duplicating by
     (period_month, area) (or fuel_type for settlement) using the source's
     `source_priority` from sources.yaml — lower number wins.
  3. Pivots to a wide per-period feature row before applying lag/rolling.

The no-future-leakage contract is unchanged: features at month M are derived
only from values whose period_month is strictly < M.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from src.config.settings import get_source_config
from src.utils.io import source_root_dir


# ---------------------------------------------------------------------------
# Lag / rolling specs (unchanged, but ts_col defaults to period_month)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class MonthlyLagSpec:
    column: str
    periods: int       # months (positive = past)
    new_name: str | None = None

    def output_name(self) -> str:
        return self.new_name or f"{self.column}_lag_{self.periods}m"


@dataclass(frozen=True)
class MonthlyRollingSpec:
    column: str
    window_months: int
    agg: str
    new_name: str | None = None

    def output_name(self) -> str:
        return self.new_name or f"{self.column}_rolling_{self.window_months}m_{self.agg}"


def _reindex_monthly(df: pd.DataFrame, ts_col: str = "period_month") -> pd.DataFrame:
    df = df.sort_values(ts_col).copy()
    df[ts_col] = pd.to_datetime(df[ts_col])
    full = pd.date_range(df[ts_col].min(), df[ts_col].max(), freq="MS")
    df = df.set_index(ts_col).reindex(full)
    df.index.name = ts_col
    return df.reset_index()


def add_monthly_lags(
    df: pd.DataFrame,
    specs: Iterable[MonthlyLagSpec],
    *,
    ts_col: str = "period_month",
) -> pd.DataFrame:
    df = _reindex_monthly(df, ts_col)
    for spec in specs:
        if spec.column not in df.columns:
            raise KeyError(f"Lag source column missing: {spec.column}")
        if spec.periods <= 0:
            raise ValueError(
                f"MonthlyLagSpec.periods must be > 0 (got {spec.periods}) — "
                "no-future-leakage rule."
            )
        df[spec.output_name()] = df[spec.column].shift(spec.periods)
    return df


def add_monthly_rollings(
    df: pd.DataFrame,
    specs: Iterable[MonthlyRollingSpec],
    *,
    ts_col: str = "period_month",
) -> pd.DataFrame:
    df = _reindex_monthly(df, ts_col)
    for spec in specs:
        if spec.column not in df.columns:
            raise KeyError(f"Rolling source column missing: {spec.column}")
        if spec.window_months <= 0:
            raise ValueError(f"window_months must be > 0 (got {spec.window_months})")
        shifted = df[spec.column].shift(1)
        rolled = shifted.rolling(window=spec.window_months, min_periods=spec.window_months)
        if spec.agg == "mean":
            df[spec.output_name()] = rolled.mean()
        elif spec.agg == "std":
            df[spec.output_name()] = rolled.std()
        elif spec.agg == "min":
            df[spec.output_name()] = rolled.min()
        elif spec.agg == "max":
            df[spec.output_name()] = rolled.max()
        else:
            raise ValueError(f"Unsupported agg {spec.agg!r}")
    return df


def add_calendar_features(
    df: pd.DataFrame, ts_col: str = "period_month"
) -> pd.DataFrame:
    out = df.copy()
    ts = pd.to_datetime(out[ts_col])
    out["year"] = ts.dt.year
    out["month"] = ts.dt.month
    out["quarter"] = ts.dt.quarter
    out["is_summer"] = out["month"].isin([6, 7, 8])
    out["is_winter"] = out["month"].isin([12, 1, 2])
    out["is_peak_season"] = out["is_summer"] | out["is_winter"]
    out["month_sin"] = np.sin(2 * np.pi * out["month"] / 12)
    out["month_cos"] = np.cos(2 * np.pi * out["month"] / 12)
    return out


# ---------------------------------------------------------------------------
# Loading per-source parsed_*.parquet (long-format)
# ---------------------------------------------------------------------------

# Monthly SMP source ids (ordered by source_priority once dedup'd).
SMP_SOURCES = ["kpx_smp_monthly_kepco_file", "kpx_smp_monthly_home_avg_file"]


def _load_one_source_parquet(source_name: str) -> pd.DataFrame:
    root = source_root_dir(source_name)
    files = sorted(root.rglob("parsed_*.parquet"))
    if not files:
        return pd.DataFrame()
    return pd.concat([pd.read_parquet(p) for p in files], ignore_index=True)


def _source_priority(source_id: str) -> int:
    try:
        cfg = get_source_config(source_id)
    except KeyError:
        return 99
    return int(cfg.get("source_priority", 99))


def load_smp_monthly_long(sources: list[str] | None = None) -> pd.DataFrame:
    """Read all monthly SMP sources into a single long-format frame.

    Duplicates on (period_month, area) are resolved by source_priority
    (lower number wins). The resolution decisions are recorded as side-info
    on the returned dataframe via ``df.attrs['_priority_dedup_log']``.
    """
    sources = sources or SMP_SOURCES
    frames: list[pd.DataFrame] = []
    for src in sources:
        df = _load_one_source_parquet(src)
        if df.empty:
            continue
        df["_priority"] = _source_priority(src)
        frames.append(df)
    if not frames:
        raise FileNotFoundError(
            "No parsed monthly SMP data on disk. Run "
            "`python -m src.pipelines.load_files --source <kpx_smp_monthly_*>` first."
        )
    long = pd.concat(frames, ignore_index=True)
    long["period_month"] = pd.to_datetime(long["period_month"])
    long = long.sort_values(["period_month", "area", "_priority"]).reset_index(drop=True)

    # Build a dedup log capturing any (period_month, area) where multiple
    # sources disagreed BEFORE we keep the winner.
    dup_mask = long.duplicated(subset=["period_month", "area"], keep=False)
    dup_log = long.loc[dup_mask, [
        "period_month", "area", "source_id", "_priority", "smp_krw_per_kwh"
    ]].copy()

    deduped = long.drop_duplicates(subset=["period_month", "area"], keep="first").drop(
        columns=["_priority"]
    )
    deduped.attrs["_priority_dedup_log"] = dup_log
    return deduped


def load_settlement_monthly_long() -> pd.DataFrame:
    df = _load_one_source_parquet("kpx_settlement_monthly_file")
    if not df.empty:
        df["period_month"] = pd.to_datetime(df["period_month"])
        df = df.sort_values(["period_month", "fuel_type"]).drop_duplicates(
            subset=["period_month", "fuel_type"], keep="last"
        )
    return df.reset_index(drop=True)


# ---------------------------------------------------------------------------
# Wide pivot for one area
# ---------------------------------------------------------------------------

def _smp_wide_for_area(smp_long: pd.DataFrame, area: str) -> pd.DataFrame:
    if area not in {"mainland", "jeju", "integrated"}:
        raise ValueError(f"Unknown area {area!r}")
    chunk = smp_long[smp_long["area"] == area]
    if chunk.empty:
        raise FileNotFoundError(
            f"No monthly SMP rows for area={area!r}. Sources on disk had: "
            f"{sorted(smp_long['area'].unique()) if not smp_long.empty else []}"
        )
    return chunk[["period_month", "smp_krw_per_kwh"]].reset_index(drop=True)


# ---------------------------------------------------------------------------
# Exogenous lag helper (carried over from round 1 — robust for gaps + dupes)
# ---------------------------------------------------------------------------

def _exogenous_lag_1m(
    df: pd.DataFrame,
    *,
    value_cols: list[str],
    ts_col: str = "period_month",
) -> pd.DataFrame:
    """Produce ``<col>_lag_1m`` columns for an exogenous monthly series.

    Aggregates same-month duplicates by mean (deterministic, order-
    independent) and reindexes to a contiguous month-start grid before
    shifting so gaps become NaN instead of silently joining to the previous
    available month.
    """
    lag_cols = [f"{c}_lag_1m" for c in value_cols]
    if df.empty or not value_cols:
        return pd.DataFrame(columns=[ts_col, *lag_cols])
    work = df[[ts_col, *value_cols]].copy()
    work[ts_col] = pd.to_datetime(work[ts_col])
    work = (
        work.groupby(ts_col, as_index=False, sort=True)[value_cols]
        .mean(numeric_only=True)
    )
    full = pd.date_range(work[ts_col].min(), work[ts_col].max(), freq="MS")
    work = work.set_index(ts_col).reindex(full)
    work.index.name = ts_col
    for col in value_cols:
        work[f"{col}_lag_1m"] = work[col].shift(1)
    return work.reset_index()[[ts_col, *lag_cols]]


# ---------------------------------------------------------------------------
# Top-level builder
# ---------------------------------------------------------------------------

DEFAULT_MONTHLY_LAGS = [
    MonthlyLagSpec("smp_krw_per_kwh", 1, "smp_lag_1m"),
    MonthlyLagSpec("smp_krw_per_kwh", 2, "smp_lag_2m"),
    MonthlyLagSpec("smp_krw_per_kwh", 3, "smp_lag_3m"),
    MonthlyLagSpec("smp_krw_per_kwh", 6, "smp_lag_6m"),
    MonthlyLagSpec("smp_krw_per_kwh", 12, "smp_lag_12m"),
]
DEFAULT_MONTHLY_ROLLINGS = [
    MonthlyRollingSpec("smp_krw_per_kwh", 3, "mean", "smp_rolling_3m_mean"),
    MonthlyRollingSpec("smp_krw_per_kwh", 6, "mean", "smp_rolling_6m_mean"),
    MonthlyRollingSpec("smp_krw_per_kwh", 12, "mean", "smp_rolling_12m_mean"),
    MonthlyRollingSpec("smp_krw_per_kwh", 12, "std", "smp_rolling_12m_std"),
]


def build_smp_monthly_features(
    area: str = "mainland",
    *,
    horizon_months: int = 1,
    target_col: str = "target_smp_t_plus_h_months",
    include_settlement: bool = True,
) -> tuple[pd.DataFrame, dict]:
    """Build monthly features for a single area.

    Returns the feature dataframe AND a side-info dict useful for the DQ
    report (priority dedup decisions, settlement fuel coverage).
    """
    smp_long = load_smp_monthly_long()
    wide = _smp_wide_for_area(smp_long, area)
    feats = add_monthly_lags(wide, DEFAULT_MONTHLY_LAGS)
    feats = add_monthly_rollings(feats, DEFAULT_MONTHLY_ROLLINGS)
    feats = add_calendar_features(feats)
    feats["area"] = area

    settlement_fuels: list[str] = []
    if include_settlement:
        sett = load_settlement_monthly_long()
        if not sett.empty:
            # Pivot fuel_type -> wide columns named like
            # settlement_unit_price_<fuel>_lag_1m.
            wide_sett = sett.pivot_table(
                index="period_month",
                columns="fuel_type",
                values="settlement_unit_price_krw_per_kwh",
                aggfunc="mean",
            ).reset_index()
            settlement_fuels = [c for c in wide_sett.columns if c != "period_month"]
            renamed_cols = {f: f"settlement_unit_price_{f}" for f in settlement_fuels}
            wide_sett = wide_sett.rename(columns=renamed_cols)
            lag_cols = list(renamed_cols.values())
            sett_lags = _exogenous_lag_1m(wide_sett, value_cols=lag_cols)
            if not sett_lags.empty:
                feats = feats.merge(sett_lags, on="period_month", how="left")

    if horizon_months <= 0:
        raise ValueError("horizon_months must be > 0")
    feats[target_col] = feats["smp_krw_per_kwh"].shift(-horizon_months)

    feature_cols = [
        c for c in feats.columns
        if c not in {target_col, "period_month", "area", "smp_krw_per_kwh"}
        and feats[c].dtype != object
    ]
    feats = feats.dropna(subset=feature_cols + [target_col]).reset_index(drop=True)
    _assert_monthly_no_leakage(feats, target_col=target_col, feature_cols=feature_cols)

    side_info = {
        "area": area,
        "rows": int(len(feats)),
        "feature_cols": feature_cols,
        "settlement_fuels": sorted(settlement_fuels),
        "smp_priority_dedup_log": smp_long.attrs.get("_priority_dedup_log"),
    }
    return feats, side_info


def _assert_monthly_no_leakage(
    df: pd.DataFrame,
    *,
    target_col: str,
    feature_cols: Iterable[str],
) -> None:
    target = pd.to_numeric(df[target_col], errors="coerce")
    suspects: list[str] = []
    for col in feature_cols:
        if col not in df.columns:
            continue
        feat = pd.to_numeric(df[col], errors="coerce")
        mask = feat.notna() & target.notna()
        if mask.sum() < 6:
            continue
        if (feat[mask].to_numpy() == target[mask].to_numpy()).all():
            suspects.append(col)
    if suspects:
        raise AssertionError(f"Monthly feature(s) leak future target: {suspects}")
