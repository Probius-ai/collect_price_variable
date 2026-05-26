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
    """Add purely deterministic calendar features.

    NB: we deliberately do NOT add a hardcoded LNG-shock indicator. An
    earlier round shipped `is_lng_shock_period = (2022-01..2023-12)` but
    those boundaries were picked AFTER observing the validation split's
    peak_threshold (246.89 KRW/kWh) — i.e. with knowledge of which months
    turned out to be the shock. In a walk-forward CV at, say, August 2021,
    the model would already "know" the shock starts in 2022 — that is
    look-ahead leakage by post-hoc feature engineering. A leak-free regime
    proxy can be derived from `smp_rolling_*` z-scores at runtime if needed
    (computable from lag features only), but is intentionally not added by
    default.
    """
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


REVISION_META_COLUMNS = [
    "collected_at",
    "source_file_sha256",
    "source_file",
    "parsed_path",
    "source_id",
    "source_priority",
]


def _load_one_source_parquet(source_name: str) -> pd.DataFrame:
    """Read every parsed snapshot for a source and stamp each row with the
    revision-metadata needed for deterministic de-duplication.

    Older parquets may pre-date the loader change that began writing
    ``source_file_sha256``, ``source_priority``, and ``parsed_path`` into the
    parsed dataframe. For those rows we fill the missing fields from
    ``sources.yaml`` and the parquet path itself so that the new dedup
    contract still applies uniformly.
    """
    root = source_root_dir(source_name)
    files = sorted(root.rglob("parsed_*.parquet"))
    if not files:
        return pd.DataFrame()
    priority_fallback = _source_priority(source_name)
    frames: list[pd.DataFrame] = []
    for p in files:
        df = pd.read_parquet(p)
        if df.empty:
            continue
        # parsed_path is intrinsic to the file just read — always overwrite.
        df["parsed_path"] = str(p)
        if "source_priority" not in df.columns:
            df["source_priority"] = priority_fallback
        else:
            df["source_priority"] = (
                df["source_priority"].fillna(priority_fallback).astype("int64")
            )
        if "source_file_sha256" not in df.columns:
            df["source_file_sha256"] = ""
        else:
            df["source_file_sha256"] = df["source_file_sha256"].fillna("").astype(str)
        if "source_id" not in df.columns:
            df["source_id"] = source_name
        frames.append(df)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def _source_priority(source_id: str) -> int:
    try:
        cfg = get_source_config(source_id)
    except KeyError:
        return 99
    return int(cfg.get("source_priority", 99))


def load_smp_monthly_long(sources: list[str] | None = None) -> pd.DataFrame:
    """Read all monthly SMP sources into a single long-format frame.

    De-duplication contract for (period_month, area):
      1. Lower ``source_priority`` wins (KEPCO < HOME in our config).
      2. Within the same priority, the snapshot with the LATEST
         ``collected_at`` wins — this means a corrected re-load supersedes
         the older snapshot for the same period.
      3. Ties are broken deterministically by ``source_file_sha256`` then
         by ``parsed_path``, both ascending.

    The resolution decisions are recorded on the returned dataframe via
    ``df.attrs['_priority_dedup_log']``. Each conflicting (period_month, area)
    contributes ONE row per candidate (selected + dropped) with columns:

        period_month, area, source_id, source_priority, smp_krw_per_kwh,
        collected_at, source_file_sha256, parsed_path, role, reason

    ``role`` is one of ``"selected"`` / ``"dropped"``. ``reason`` is
    ``"lower_priority"`` if the row lost to a higher-priority source, or
    ``"older_revision"`` if it lost to a same-priority newer snapshot.
    The legacy ``_priority`` alias column is preserved for backward compat.
    """
    sources = sources or SMP_SOURCES
    frames: list[pd.DataFrame] = []
    for src in sources:
        df = _load_one_source_parquet(src)
        if df.empty:
            continue
        frames.append(df)
    if not frames:
        raise FileNotFoundError(
            "No parsed monthly SMP data on disk. Run "
            "`python -m src.pipelines.load_files --source <kpx_smp_monthly_*>` first."
        )
    long = pd.concat(frames, ignore_index=True)
    long["period_month"] = pd.to_datetime(long["period_month"])
    long["source_priority"] = long["source_priority"].astype("int64")
    long["source_file_sha256"] = long["source_file_sha256"].fillna("").astype(str)
    long["parsed_path"] = long["parsed_path"].astype(str)

    # Build the collected_at rank using an INTERNAL UTC-normalised series.
    # We deliberately do NOT overwrite long["collected_at"] with the tz-aware
    # version: the external contract (returned dataframe + _priority_dedup_log
    # + side-info JSON consumers) expects UTC-naive timestamps, matching the
    # collectors that wrote them.
    #
    # utc=True interprets tz-naive values as UTC and converts tz-aware values
    # to UTC, so the rank works uniformly regardless of how each parquet was
    # written. After computing the rank we normalise long["collected_at"]
    # back to UTC-naive for the rest of the function.
    collected_utc = pd.to_datetime(long["collected_at"], utc=True, errors="coerce")
    # Full native precision (microseconds in pandas ≥ 3, nanoseconds in older
    # pandas). Do NOT divide to seconds: same-second-different-microsecond
    # snapshots must still be distinguishable, otherwise the lexicographic
    # sha/path tie-breaker would silently keep an older revision.
    # NaT is forced to int64.min so missing collected_at sorts LAST under
    # descending-by-rank ordering (older / unknown timing always loses).
    collected_int = collected_utc.astype("int64")
    long["_collected_at_rank"] = collected_int.where(
        collected_utc.notna(), other=np.iinfo(np.int64).min
    )
    # Restore the external contract: UTC-naive collected_at.
    long["collected_at"] = collected_utc.dt.tz_convert("UTC").dt.tz_localize(None)
    long_sorted = long.sort_values(
        by=[
            "period_month",
            "area",
            "source_priority",
            "_collected_at_rank",
            "source_file_sha256",
            "parsed_path",
        ],
        ascending=[True, True, True, False, True, True],
        kind="mergesort",  # stable
    ).reset_index(drop=True)

    # Build the revision log BEFORE dropping duplicates.
    dup_mask = long_sorted.duplicated(subset=["period_month", "area"], keep=False)
    candidates = long_sorted.loc[dup_mask].copy()

    if candidates.empty:
        dedup_log = pd.DataFrame(
            columns=[
                "period_month", "area", "source_id", "source_priority",
                "smp_krw_per_kwh", "collected_at", "source_file_sha256",
                "parsed_path", "role", "reason", "_priority",
            ]
        )
    else:
        # The first row in each (period_month, area) group is the winner because
        # of the multi-key sort above.
        first_pos = candidates.groupby(
            ["period_month", "area"], sort=False
        ).head(1).index
        candidates["role"] = "dropped"
        candidates.loc[first_pos, "role"] = "selected"

        selected_priority = (
            candidates.loc[candidates["role"] == "selected"]
            .set_index(["period_month", "area"])["source_priority"]
            .to_dict()
        )

        def _reason(row: pd.Series) -> str:
            if row["role"] == "selected":
                return ""
            sel_p = selected_priority.get((row["period_month"], row["area"]))
            if sel_p is None or row["source_priority"] > sel_p:
                return "lower_priority"
            return "older_revision"

        candidates["reason"] = candidates.apply(_reason, axis=1)
        candidates["_priority"] = candidates["source_priority"]
        dedup_log = candidates[[
            "period_month", "area", "source_id", "source_priority",
            "smp_krw_per_kwh", "collected_at", "source_file_sha256",
            "parsed_path", "role", "reason", "_priority",
        ]].reset_index(drop=True)

    deduped = (
        long_sorted.drop_duplicates(subset=["period_month", "area"], keep="first")
        .drop(columns=["_collected_at_rank"])
        .reset_index(drop=True)
    )
    deduped.attrs["_priority_dedup_log"] = dedup_log
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
# Round-3 optional loaders: capacity (yearly + monthly) + transaction
# ---------------------------------------------------------------------------

def load_capacity_monthly_by_fuel_long() -> pd.DataFrame:
    """Monthly fuel-capacity long-format frame (already filtered to all-합계)."""
    df = _load_one_source_parquet("kpx_capacity_monthly_by_fuel_home_file")
    if not df.empty:
        df["period_month"] = pd.to_datetime(df["period_month"])
        df = df.sort_values(["period_month", "fuel_type"]).drop_duplicates(
            subset=["period_month", "fuel_type"], keep="last"
        )
    return df.reset_index(drop=True)


def load_capacity_monthly_by_generation_type_long() -> pd.DataFrame:
    df = _load_one_source_parquet("kpx_capacity_monthly_by_generation_type_home_file")
    if not df.empty:
        df["period_month"] = pd.to_datetime(df["period_month"])
        df = df.sort_values(
            ["period_month", "capacity_type_canonical"]
        ).drop_duplicates(
            subset=["period_month", "capacity_type_canonical"], keep="last"
        )
    return df.reset_index(drop=True)


def load_capacity_yearly_by_energy_source_long() -> pd.DataFrame:
    df = _load_one_source_parquet("kpx_capacity_yearly_by_energy_source_home_file")
    if not df.empty:
        df = df.sort_values(
            ["period_year", "row_category",
             "capacity_category_level1", "capacity_category_level2"]
        )
    return df.reset_index(drop=True)


def load_transaction_volume_hourly_long() -> pd.DataFrame:
    df = _load_one_source_parquet("kpx_transaction_volume_hourly_by_fuel_file")
    if not df.empty:
        df["interval_end"] = pd.to_datetime(df["interval_end"])
    return df.reset_index(drop=True)


def load_transaction_amount_daily_long() -> pd.DataFrame:
    df = _load_one_source_parquet("kpx_transaction_amount_daily_by_fuel_file")
    if not df.empty:
        df["trade_date"] = pd.to_datetime(df["trade_date"])
    return df.reset_index(drop=True)


def load_jkm_daily_history_long() -> pd.DataFrame:
    """investing.com JKM daily OHLC history (2013-01 onwards, ~2900 rows)."""
    df = _load_one_source_parquet("jkm_lng_daily_history_file")
    if not df.empty:
        df["trade_date"] = pd.to_datetime(df["trade_date"])
    return df.reset_index(drop=True)


def build_jkm_daily_volatility_features(period_months: pd.Series) -> tuple[pd.DataFrame, dict]:
    """Aggregate JKM daily OHLC into monthly mean / std / range (lag_1m).

    At forecast_origin_month M, the daily JKM prices for month M-1 are
    fully observable (daily futures settles publish on the same day). We
    expose:

      * `jkm_daily_mean_lag_1m`   = mean of daily close in month M-1
      * `jkm_daily_std_lag_1m`    = std of daily close (volatility signal)
      * `jkm_daily_range_lag_1m`  = max-min over month M-1
      * `jkm_daily_last_lag_1m`   = last trade in M-1 (end-of-month snap)

    Lag_1m is mandatory here because at info_cutoff = end-of-M only
    daily JKM through M-1 has had a chance to fully settle (the very last
    daily print of M might not be fully cleared yet).
    """
    long = load_jkm_daily_history_long()
    info = {
        "source_id": "jkm_lng_daily_history_file",
        "feature_origin_frequency": "daily_aggregated_to_monthly",
        "rows_in_source": int(len(long)),
        "period_range": None,
        "missing_periods_in_smp_index": 0,
    }
    if long.empty:
        return pd.DataFrame(columns=["period_month"]), info

    daily = long[["trade_date", "close"]].dropna().sort_values("trade_date").copy()
    daily["period_month"] = daily["trade_date"].dt.to_period("M").dt.to_timestamp()
    g = daily.groupby("period_month")["close"]
    monthly = pd.DataFrame({
        "jkm_daily_mean": g.mean(),
        "jkm_daily_std": g.std(ddof=1),
        "jkm_daily_max": g.max(),
        "jkm_daily_min": g.min(),
        "jkm_daily_last": g.last(),
    }).reset_index()
    monthly["jkm_daily_range"] = monthly["jkm_daily_max"] - monthly["jkm_daily_min"]

    # Contiguous month grid extended by 1 (same trick as the round-6 LNG fix)
    full = pd.date_range(monthly["period_month"].min(),
                         monthly["period_month"].max() + pd.DateOffset(months=1),
                         freq="MS")
    monthly = monthly.set_index("period_month").reindex(full)
    monthly.index.name = "period_month"
    out = pd.DataFrame({
        "jkm_daily_mean_lag_1m": monthly["jkm_daily_mean"].shift(1),
        "jkm_daily_std_lag_1m": monthly["jkm_daily_std"].shift(1),
        "jkm_daily_range_lag_1m": monthly["jkm_daily_range"].shift(1),
        "jkm_daily_last_lag_1m": monthly["jkm_daily_last"].shift(1),
    }).reset_index()

    info["period_range"] = (str(out["period_month"].min()), str(out["period_month"].max()))
    info["missing_periods_in_smp_index"] = int(
        (~period_months.isin(out["period_month"])).sum()
    )
    return out, info


def load_jkm_futures_curve_long() -> pd.DataFrame:
    """CME JKM forward curve snapshots (one row per (quote_date, contract_month))."""
    df = _load_one_source_parquet("jkm_lng_futures_curve_file")
    if not df.empty:
        df["quote_date"] = pd.to_datetime(df["quote_date"])
        df["contract_month"] = pd.to_datetime(df["contract_month"])
    return df.reset_index(drop=True)


def build_jkm_forward_features(period_months: pd.Series) -> tuple[pd.DataFrame, dict]:
    """True-leading indicator from the CME JKM forward curve.

    For each row at SMP forecast_origin_month M, we look up the most recent
    forward-curve snapshot whose ``quote_date`` is ≤ end-of-M (information_cutoff)
    and emit:

      * `jkm_forward_next_month_usd_per_mmbtu`  = price for contract whose
        contract_month = M+1 (i.e. the market's expectation of LNG at M+1
        as observed at forecast origin).
      * `jkm_forward_M_plus_2_usd_per_mmbtu`    = M+2 contract.
      * `jkm_forward_slope_1_2`                  = (M+2 − M+1) / M+1 (curve slope).

    These are **legitimate leading features** for the SMP forecast at M+1
    because they reflect forward-looking market consensus that IS available
    at end-of-M.

    Backtesting caveat: we usually have a *single* forward-curve snapshot
    (the latest one, captured by the user). For historical SMP rows we
    have no forward curve, so the values default to NaN. Practical impact:
    only the most recent few SMP rows get filled — useful for INFERENCE,
    not for training.
    """
    long = load_jkm_futures_curve_long()
    info = {
        "source_id": "jkm_lng_futures_curve_file",
        "feature_origin_frequency": "monthly_forward_lookup",
        "rows_in_source": int(len(long)),
        "snapshots_available": (
            sorted(set(map(str, long["quote_date"].dt.date.unique())))
            if not long.empty else []
        ),
        "missing_periods_in_smp_index": 0,
    }
    if long.empty:
        return pd.DataFrame(columns=["period_month"]), info

    out_rows: list[dict] = []
    for pm in period_months.drop_duplicates():
        info_cutoff = pd.Timestamp(pm) + pd.offsets.MonthEnd(0)
        # Pick the LATEST snapshot whose quote_date ≤ info_cutoff. This is
        # the "what did the market know at end-of-M" lookup.
        eligible = long[long["quote_date"] <= info_cutoff]
        if eligible.empty:
            continue
        last_qd = eligible["quote_date"].max()
        snap = eligible[eligible["quote_date"] == last_qd]
        m_plus_1 = pd.Timestamp(pm) + pd.DateOffset(months=1)
        m_plus_2 = pd.Timestamp(pm) + pd.DateOffset(months=2)
        row1 = snap[snap["contract_month"] == m_plus_1]
        row2 = snap[snap["contract_month"] == m_plus_2]
        v1 = float(row1["prior_settle"].iloc[0]) if len(row1) else None
        v2 = float(row2["prior_settle"].iloc[0]) if len(row2) else None
        slope = (v2 - v1) / v1 if (v1 and v2) else None
        out_rows.append({
            "period_month": pd.Timestamp(pm),
            "jkm_forward_next_month_usd_per_mmbtu": v1,
            "jkm_forward_M_plus_2_usd_per_mmbtu": v2,
            "jkm_forward_slope_1_2": slope,
        })

    out = pd.DataFrame(out_rows)
    info["missing_periods_in_smp_index"] = int(
        (~period_months.isin(out["period_month"])).sum()
    ) if not out.empty else int(len(period_months))
    return out, info


def load_lng_price_monthly_long() -> pd.DataFrame:
    """FRED Asia LNG monthly price (USD/MMBtu), via solar_beam DB export.

    The month label = the calendar month for which the JKM monthly average
    was computed. We treat that value as **observable at end-of-month**, i.e.
    available at the SMP forecast's information_cutoff for the same month.
    """
    df = _load_one_source_parquet("lng_price_monthly_file")
    if not df.empty:
        df["period_month"] = pd.to_datetime(df["period_month"])
        df = df.sort_values("period_month").drop_duplicates(
            subset=["period_month"], keep="last"
        )
    return df.reset_index(drop=True)


def build_lng_features(period_months: pd.Series) -> tuple[pd.DataFrame, dict]:
    """LNG price features for the SMP feature table.

    **Publish-timing discipline (round 6 leakage fix):**
    FRED PNGASJPUSDM (JKM Asia monthly LNG average) for month M is the
    average of all daily prices within M, and is **published around the
    15th of M+1** (after World Bank receives the data). Therefore at the
    SMP forecast's ``information_cutoff = end-of-M``, the value for
    month M is NOT yet observable — only LNG up to month M−1 is.

    We therefore expose ONLY lagged columns at row M:

      * `lng_price_usd_per_mmbtu_lag_1m`   = LNG(M−1)   (observable at end-of-M)
      * `lng_price_usd_per_mmbtu_lag_2m`   = LNG(M−2)
      * `lng_price_chg_1m_lag_1m`          = (LNG(M−1)−LNG(M−2))/LNG(M−2)
                                             — month-over-month change as of
                                               the latest observable point

    The unlagged current-month column is INTENTIONALLY DROPPED. Including
    it would leak future information across the forecast cutoff (the
    previous round-6 implementation did this and was caught by review).
    A leakage-canary test in tests/test_round6_lng_integration.py pins
    this contract.

    Returns (DataFrame keyed on period_month, info dict).
    """
    long = load_lng_price_monthly_long()
    info = {
        "source_id": "lng_price_monthly_file",
        "feature_origin_frequency": "monthly",
        "rows_in_source": int(len(long)),
        "period_range": None,
        "missing_periods_in_smp_index": 0,
        "publish_timing_note": (
            "FRED PNGASJPUSDM monthly avg published ~mid-M+1; therefore at "
            "SMP info_cutoff=end-of-M only LNG up to M-1 is observable. "
            "Current-month column is NOT exposed as a feature."
        ),
    }
    if long.empty:
        return pd.DataFrame(columns=["period_month"]), info

    base = long[["period_month", "lng_price_usd_per_mmbtu"]].copy()
    base = base.sort_values("period_month").reset_index(drop=True)
    # Reindex to a contiguous month grid, **extended by MAX_LAG months past
    # raw.max** so the newest usable lag is exposed at the corresponding row.
    # Without this extension the LNG output stops at raw.max, and when the
    # SMP feature table extends one month past LNG raw (e.g. SMP refreshed
    # ahead of the next LNG publication), the SMP row at raw.max+1 silently
    # gets NaN for lag_1m even though LNG(raw.max) IS observable at that
    # row's information_cutoff. Round-3 fixed the same pattern in
    # `_exogenous_lag_1m`; this is the LNG-specific counterpart.
    # MAX_LAG = 2 because we emit lag_1m and lag_2m.
    _MAX_LAG_MONTHS = 2
    extended_max = base["period_month"].max() + pd.DateOffset(months=_MAX_LAG_MONTHS)
    full = pd.date_range(base["period_month"].min(), extended_max, freq="MS")
    base = base.set_index("period_month").reindex(full)
    base.index.name = "period_month"

    # Build OUTPUT with ONLY lagged columns. The raw LNG(M) is used to
    # derive the lags but never exposed as a feature.
    raw = base["lng_price_usd_per_mmbtu"]
    # IMPORTANT: pass fill_method=None explicitly. In pandas 2.x the
    # deprecated default `fill_method='pad'` forward-fills NaN BEFORE
    # computing pct_change — which would silently fabricate `chg=0` at the
    # extended-grid positions (Tmax+1, Tmax+2) created by the round-6 grid
    # extension fix. We want those positions to stay NaN until the real
    # LNG value lands.
    out = pd.DataFrame({
        # observable at end-of-M = info_cutoff
        "lng_price_usd_per_mmbtu_lag_1m": raw.shift(1),
        "lng_price_usd_per_mmbtu_lag_2m": raw.shift(2),
        # change between M-2 and M-1 (i.e., the most-recent observable Δ)
        "lng_price_chg_1m_lag_1m": raw.pct_change(1, fill_method=None).shift(1),
    })
    out = out.reset_index()

    info["period_range"] = (
        str(out["period_month"].min()), str(out["period_month"].max())
    )
    info["missing_periods_in_smp_index"] = int(
        (~period_months.isin(out["period_month"])).sum()
    )
    return out, info


# ---------------------------------------------------------------------------
# Round-3 feature-block builders (each emits a per-period_month wide table)
# ---------------------------------------------------------------------------

# Fuel sub-categories whose sum gives `renewable_total` for capacity.
RENEWABLE_SUB_FUELS_CAPACITY = (
    "renewable_solar", "renewable_wind", "renewable_fuel_cell",
    "renewable_hydro", "renewable_marine", "renewable_bio",
    "renewable_waste", "renewable_igcc",
)
# For transactions (different sub-fuel set because amount file has biomass etc).
RENEWABLE_SUB_FUELS_TX = (
    "renewable_solar", "renewable_wind", "renewable_fuel_cell",
    "renewable_hydro", "renewable_marine", "renewable_bio",
    "renewable_biogas", "renewable_biomass", "renewable_bio_srf",
    "renewable_waste", "renewable_igcc",
)


def _pivot_fuel_wide(
    long: pd.DataFrame,
    value_col: str,
    *,
    index_col: str = "period_month",
    fuel_col: str = "fuel_type",
) -> pd.DataFrame:
    """Pivot a long (period, fuel, value) frame into one row per period."""
    if long.empty:
        return pd.DataFrame(columns=[index_col])
    return long.pivot_table(
        index=index_col, columns=fuel_col, values=value_col, aggfunc="sum"
    ).reset_index()


def build_capacity_fuel_features(period_months: pd.Series) -> tuple[pd.DataFrame, dict]:
    """capacity_fuel_*_mw + share features, lagged 1 month and keyed on
    period_month.

    Lag 1 month is required by the no-future-leakage contract: a feature at
    row M may only use values whose underlying period_month is strictly < M.
    The lag also gives capacity publication time (KPX HOME data comes out
    with a M+1 delay).
    """
    long = load_capacity_monthly_by_fuel_long()
    info = {
        "source_id": "kpx_capacity_monthly_by_fuel_home_file",
        "feature_origin_frequency": "monthly",
        "lag_applied": "1m",
        "rows_in_source": int(len(long)),
        "period_range": None,
        "missing_periods_in_smp_index": 0,
    }
    if long.empty:
        return pd.DataFrame(columns=["period_month"]), info
    wide = _pivot_fuel_wide(long, value_col="capacity_mw")
    # Compute renewable_total as sum of available sub-fuels.
    sub_cols = [c for c in RENEWABLE_SUB_FUELS_CAPACITY if c in wide.columns]
    wide["renewable_total"] = wide[sub_cols].sum(axis=1) if sub_cols else 0.0
    rename = {
        "nuclear": "capacity_fuel_nuclear_mw",
        "coal_bituminous": "capacity_fuel_coal_bituminous_mw",
        "coal_anthracite": "capacity_fuel_coal_anthracite_mw",
        "oil": "capacity_fuel_oil_mw",
        "lng": "capacity_fuel_lng_mw",
        "pumped_storage": "capacity_fuel_pumped_storage_mw",
        "renewable_solar": "capacity_fuel_renewable_solar_mw",
        "renewable_wind": "capacity_fuel_renewable_wind_mw",
        "renewable_fuel_cell": "capacity_fuel_renewable_fuel_cell_mw",
        "renewable_total": "capacity_fuel_renewable_total_mw",
        "other": "capacity_fuel_other_mw",
        "total": "capacity_fuel_total_mw",
    }
    wide = wide.rename(columns={k: v for k, v in rename.items() if k in wide.columns})
    keep = ["period_month"] + [v for v in rename.values() if v in wide.columns]
    current = wide[keep].copy()

    total = current.get("capacity_fuel_total_mw")
    if total is not None:
        denom = total.where(total > 0)
        if "capacity_fuel_nuclear_mw" in current.columns:
            current["capacity_fuel_nuclear_share"] = current["capacity_fuel_nuclear_mw"] / denom
        if "capacity_fuel_lng_mw" in current.columns:
            current["capacity_fuel_lng_share"] = current["capacity_fuel_lng_mw"] / denom
        coal_b = current.get("capacity_fuel_coal_bituminous_mw", 0)
        coal_a = current.get("capacity_fuel_coal_anthracite_mw", 0)
        current["capacity_fuel_coal_total_mw"] = coal_b + coal_a
        current["capacity_fuel_coal_total_share"] = current["capacity_fuel_coal_total_mw"] / denom
        if "capacity_fuel_renewable_total_mw" in current.columns:
            current["capacity_fuel_renewable_total_share"] = (
                current["capacity_fuel_renewable_total_mw"] / denom
            )

    # Apply lag_1m to ALL value columns. This satisfies the contract that
    # row M only uses values whose period_month < M.
    value_cols = [c for c in current.columns if c != "period_month"]
    out = _exogenous_lag_1m(current, value_cols=value_cols)
    info["period_range"] = (
        str(out["period_month"].min()), str(out["period_month"].max())
    )
    info["missing_periods_in_smp_index"] = int(
        (~period_months.isin(out["period_month"])).sum()
    )
    return out, info


def build_capacity_type_features(period_months: pd.Series) -> tuple[pd.DataFrame, dict]:
    """capacity_type_*_mw + share features, lagged 1 month."""
    long = load_capacity_monthly_by_generation_type_long()
    info = {
        "source_id": "kpx_capacity_monthly_by_generation_type_home_file",
        "feature_origin_frequency": "monthly",
        "lag_applied": "1m",
        "rows_in_source": int(len(long)),
        "period_range": None,
        "missing_periods_in_smp_index": 0,
    }
    if long.empty:
        return pd.DataFrame(columns=["period_month"]), info
    wide = long.pivot_table(
        index="period_month", columns="capacity_type_canonical",
        values="capacity_mw", aggfunc="sum",
    ).reset_index()
    # Aggregate sub-types into the canonical groups the prompt asks for.
    steam_cols = [c for c in
                  ["steam_coal_bituminous", "steam_coal_anthracite",
                   "steam_oil", "steam_gas"] if c in wide.columns]
    combined_cols = [c for c in ["combined_oil", "combined_gas"] if c in wide.columns]
    ic_cols = [c for c in ["internal_combustion_oil", "internal_combustion_gas"] if c in wide.columns]

    current = pd.DataFrame({"period_month": wide["period_month"]})
    if "nuclear" in wide.columns:
        current["capacity_type_nuclear_mw"] = wide["nuclear"]
    if steam_cols:
        current["capacity_type_steam_total_mw"] = wide[steam_cols].sum(axis=1)
    if combined_cols:
        current["capacity_type_combined_cycle_total_mw"] = wide[combined_cols].sum(axis=1)
    if ic_cols:
        current["capacity_type_internal_combustion_total_mw"] = wide[ic_cols].sum(axis=1)
    if "pumped_storage" in wide.columns:
        current["capacity_type_pumped_storage_mw"] = wide["pumped_storage"]
    if "renewable" in wide.columns:
        current["capacity_type_renewable_mw"] = wide["renewable"]
    if "total" in wide.columns:
        current["capacity_type_total_mw"] = wide["total"]

    total = current.get("capacity_type_total_mw")
    if total is not None:
        denom = total.where(total > 0)
        if "capacity_type_nuclear_mw" in current.columns:
            current["capacity_type_nuclear_share"] = current["capacity_type_nuclear_mw"] / denom
        if "capacity_type_combined_cycle_total_mw" in current.columns:
            current["capacity_type_combined_cycle_share"] = (
                current["capacity_type_combined_cycle_total_mw"] / denom
            )
        if "capacity_type_renewable_mw" in current.columns:
            current["capacity_type_renewable_share"] = (
                current["capacity_type_renewable_mw"] / denom
            )

    value_cols = [c for c in current.columns if c != "period_month"]
    out = _exogenous_lag_1m(current, value_cols=value_cols)
    info["period_range"] = (
        str(out["period_month"].min()), str(out["period_month"].max())
    )
    info["missing_periods_in_smp_index"] = int(
        (~period_months.isin(out["period_month"])).sum()
    )
    return out, info


def build_capacity_yearly_features(period_months: pd.Series) -> tuple[pd.DataFrame, dict]:
    """Broadcast yearly totals using a 1-year LAG (period_month.year - 1).

    The no-future-leakage contract requires that a row M only see values
    whose source period is strictly < M. Yearly aggregates for year Y are
    only published after Y ends (typically Q1 of Y+1), so for any month in
    year Y we use the year Y-1 capacity total. Feature names carry an
    explicit ``_lag_1y`` suffix so downstream readers know the lag.
    """
    long = load_capacity_yearly_by_energy_source_long()
    info = {
        "source_id": "kpx_capacity_yearly_by_energy_source_home_file",
        "feature_origin_frequency": "yearly_broadcast",
        "lag_applied": "1y",
        "rows_in_source": int(len(long)),
        "years_available": [],
        "missing_years_in_smp_index": 0,
    }
    if long.empty:
        return pd.DataFrame(columns=["period_month"]), info
    totals = long[long["row_category"] == "총계"].copy()
    if totals.empty:
        totals = long.copy()
    by_year = totals.groupby(
        ["period_year", "capacity_category_level1", "capacity_category_level2"],
        as_index=False,
    )["capacity_mw"].sum()
    by_year["category_key"] = (
        by_year["capacity_category_level1"]
        + by_year["capacity_category_level2"].map(lambda s: f"|{s}" if s else "")
    )
    wide = by_year.pivot_table(
        index="period_year", columns="category_key",
        values="capacity_mw", aggfunc="sum",
    ).reset_index()

    rename = {
        "원자력": "capacity_yearly_nuclear_mw_lag_1y",
        "신재생": "capacity_yearly_renewable_mw_lag_1y",
        "총계": "capacity_yearly_total_mw_lag_1y",
        "기력|계": "capacity_yearly_steam_total_mw_lag_1y",
        "복합화력|계": "capacity_yearly_combined_total_mw_lag_1y",
    }
    wide = wide.rename(columns={k: v for k, v in rename.items() if k in wide.columns})
    keep = ["period_year"] + [v for v in rename.values() if v in wide.columns]
    wide = wide[keep].copy()

    info["years_available"] = sorted(int(y) for y in wide["period_year"].unique())
    smp_years_minus_one = (period_months.dt.year - 1).unique()
    info["missing_years_in_smp_index"] = int(
        sum(y not in info["years_available"] for y in smp_years_minus_one)
    )

    # Broadcast with a 1-year LAG: month in year Y maps to yearly capacity of Y-1.
    out = pd.DataFrame({"period_month": period_months.drop_duplicates()})
    out["_lookup_year"] = out["period_month"].dt.year - 1
    out = out.merge(wide, left_on="_lookup_year", right_on="period_year", how="left")
    out = out.drop(columns=["_lookup_year", "period_year"])
    return out, info


def build_transaction_volume_features(period_months: pd.Series) -> tuple[pd.DataFrame, dict]:
    """Hourly → monthly aggregation, then 1-month lag; emits volume MWh
    per fuel + shares (all with ``_lag_1m`` suffix).

    Monthly aggregation uses ``trade_date`` rather than ``interval_end``: the
    vendor reports trade_date as the physical-flow day, so trade_hour=24 of
    the last day of a month must stay in that month's bucket — under
    interval_end semantics it would silently spill into the next month.
    """
    long = load_transaction_volume_hourly_long()
    info = {
        "source_id": "kpx_transaction_volume_hourly_by_fuel_file",
        "feature_origin_frequency": "hourly_aggregated_to_monthly",
        "lag_applied": "1m",
        "monthly_aggregation_key": "trade_date",
        "market_participating_generators_only": True,
        "rows_in_source": int(len(long)),
        "period_range": None,
        "missing_periods_in_smp_index": 0,
    }
    if long.empty:
        return pd.DataFrame(columns=["period_month"]), info
    long = long.copy()
    long["period_month"] = (
        long["trade_date"].dt.to_period("M").dt.to_timestamp()
    )
    wide = _pivot_fuel_wide(long, value_col="transaction_volume_mwh")
    sub_cols = [c for c in RENEWABLE_SUB_FUELS_TX if c in wide.columns]
    wide["renewable_total"] = wide[sub_cols].sum(axis=1) if sub_cols else 0.0
    rename = {
        "nuclear": "transaction_volume_nuclear_mwh",
        "lng": "transaction_volume_lng_mwh",
        "coal_bituminous": "transaction_volume_coal_bituminous_mwh",
        "coal_anthracite": "transaction_volume_coal_anthracite_mwh",
        "oil": "transaction_volume_oil_mwh",
        "renewable": "transaction_volume_renewable_singlecol_mwh",
        "renewable_total": "transaction_volume_renewable_total_mwh",
    }
    wide = wide.rename(columns={k: v for k, v in rename.items() if k in wide.columns})
    keep_base = [v for v in rename.values() if v in wide.columns]
    wide["transaction_volume_total_mwh"] = wide[keep_base].sum(axis=1, min_count=1)
    keep = ["period_month"] + keep_base + ["transaction_volume_total_mwh"]
    current = wide[keep].copy()

    total = current["transaction_volume_total_mwh"]
    denom = total.where(total > 0)
    for fcol, scol in [
        ("transaction_volume_nuclear_mwh", "transaction_volume_nuclear_share"),
        ("transaction_volume_lng_mwh", "transaction_volume_lng_share"),
        ("transaction_volume_renewable_total_mwh", "transaction_volume_renewable_share"),
    ]:
        if fcol in current.columns:
            current[scol] = current[fcol] / denom
    coal_b = current.get("transaction_volume_coal_bituminous_mwh", 0)
    coal_a = current.get("transaction_volume_coal_anthracite_mwh", 0)
    current["transaction_volume_coal_total_mwh"] = coal_b + coal_a
    current["transaction_volume_coal_total_share"] = (
        current["transaction_volume_coal_total_mwh"] / denom
    )

    value_cols = [c for c in current.columns if c != "period_month"]
    out = _exogenous_lag_1m(current, value_cols=value_cols)
    info["period_range"] = (
        str(out["period_month"].min()), str(out["period_month"].max())
    )
    info["missing_periods_in_smp_index"] = int(
        (~period_months.isin(out["period_month"])).sum()
    )
    return out, info


def build_transaction_amount_features(period_months: pd.Series) -> tuple[pd.DataFrame, dict]:
    """Daily → monthly aggregation (by trade_date) → 1-month lag.

    Aggregation uses trade_date directly (no boundary ambiguity for daily
    data); the result then gets ``_lag_1m`` applied for no-future-leakage.
    """
    long = load_transaction_amount_daily_long()
    info = {
        "source_id": "kpx_transaction_amount_daily_by_fuel_file",
        "feature_origin_frequency": "daily_aggregated_to_monthly",
        "lag_applied": "1m",
        "monthly_aggregation_key": "trade_date",
        "market_participating_generators_only": True,
        "rows_in_source": int(len(long)),
        "period_range": None,
        "missing_periods_in_smp_index": 0,
    }
    if long.empty:
        return pd.DataFrame(columns=["period_month"]), info
    long = long.copy()
    long["period_month"] = (
        long["trade_date"].dt.to_period("M").dt.to_timestamp()
    )
    wide = _pivot_fuel_wide(long, value_col="transaction_amount_krw")
    sub_cols = [c for c in RENEWABLE_SUB_FUELS_TX if c in wide.columns]
    wide["renewable_total"] = wide[sub_cols].sum(axis=1) if sub_cols else 0.0
    rename = {
        "nuclear": "transaction_amount_nuclear_krw",
        "lng": "transaction_amount_lng_krw",
        "coal_bituminous": "transaction_amount_coal_bituminous_krw",
        "coal_anthracite": "transaction_amount_coal_anthracite_krw",
        "oil": "transaction_amount_oil_krw",
        "renewable_total": "transaction_amount_renewable_total_krw",
    }
    wide = wide.rename(columns={k: v for k, v in rename.items() if k in wide.columns})
    keep_base = [v for v in rename.values() if v in wide.columns]
    wide["transaction_amount_total_krw"] = wide[keep_base].sum(axis=1, min_count=1)
    keep = ["period_month"] + keep_base + ["transaction_amount_total_krw"]
    current = wide[keep].copy()
    total = current["transaction_amount_total_krw"]
    denom = total.where(total > 0)
    for fcol, scol in [
        ("transaction_amount_nuclear_krw", "transaction_amount_nuclear_share"),
        ("transaction_amount_lng_krw", "transaction_amount_lng_share"),
        ("transaction_amount_renewable_total_krw", "transaction_amount_renewable_share"),
    ]:
        if fcol in current.columns:
            current[scol] = current[fcol] / denom
    coal_b = current.get("transaction_amount_coal_bituminous_krw", 0)
    coal_a = current.get("transaction_amount_coal_anthracite_krw", 0)
    current["transaction_amount_coal_total_krw"] = coal_b + coal_a
    current["transaction_amount_coal_total_share"] = (
        current["transaction_amount_coal_total_krw"] / denom
    )

    value_cols = [c for c in current.columns if c != "period_month"]
    out = _exogenous_lag_1m(current, value_cols=value_cols)
    info["period_range"] = (
        str(out["period_month"].min()), str(out["period_month"].max())
    )
    info["missing_periods_in_smp_index"] = int(
        (~period_months.isin(out["period_month"])).sum()
    )
    return out, info


def build_transaction_price_features(
    volume_feats: pd.DataFrame, amount_feats: pd.DataFrame
) -> tuple[pd.DataFrame, dict]:
    """Compute KRW/kWh approximations from monthly amount ÷ volume.

    Both inputs are already 1-month-lagged (per the no-future-leakage
    contract), so dividing them gives a 1-month-lagged unit price too —
    feature names carry the explicit ``_lag_1m`` suffix.

    ONLY runs on (period_month) months that appear in BOTH inputs; if there
    is no overlap, returns an empty frame and a warning info dict.
    """
    info = {
        "overlap_months": 0,
        "warning": None,
        "lag_applied": "1m",
    }
    if volume_feats.empty or amount_feats.empty:
        info["warning"] = "missing_volume_or_amount_data"
        return pd.DataFrame(columns=["period_month"]), info
    # NB: volume_feats and amount_feats have ALREADY been lag_1m'd by their
    # respective builders. After lag_1m, _exogenous_lag_1m's `_reindex_monthly`
    # produces a value at row M whose source is M-1. Two `period_month=M` rows
    # with both having non-NaN values means BOTH M-1 source aggregates exist.
    vol_nonnull = volume_feats.dropna(how="all", subset=[
        c for c in volume_feats.columns if c != "period_month"
    ])
    amt_nonnull = amount_feats.dropna(how="all", subset=[
        c for c in amount_feats.columns if c != "period_month"
    ])
    overlap = (
        set(vol_nonnull["period_month"]) & set(amt_nonnull["period_month"])
    )
    info["overlap_months"] = len(overlap)
    if not overlap:
        info["warning"] = (
            "no_overlap_between_transaction_volume_and_amount: "
            "skipped price feature creation"
        )
        return pd.DataFrame(columns=["period_month"]), info
    merged = volume_feats.merge(amount_feats, on="period_month", how="inner")
    pairs = [
        ("lng",
         "transaction_amount_lng_krw_lag_1m", "transaction_volume_lng_mwh_lag_1m",
         "market_trade_price_lng_krw_per_kwh_lag_1m"),
        ("coal_total",
         "transaction_amount_coal_total_krw_lag_1m",
         "transaction_volume_coal_total_mwh_lag_1m",
         "market_trade_price_coal_total_krw_per_kwh_lag_1m"),
        ("nuclear",
         "transaction_amount_nuclear_krw_lag_1m",
         "transaction_volume_nuclear_mwh_lag_1m",
         "market_trade_price_nuclear_krw_per_kwh_lag_1m"),
        ("renewable",
         "transaction_amount_renewable_total_krw_lag_1m",
         "transaction_volume_renewable_total_mwh_lag_1m",
         "market_trade_price_renewable_krw_per_kwh_lag_1m"),
    ]
    out = pd.DataFrame({"period_month": merged["period_month"]})
    for _name, amt, vol, feat in pairs:
        if amt in merged.columns and vol in merged.columns:
            vol_safe = merged[vol].where(merged[vol] > 0)
            out[feat] = (merged[amt] / vol_safe) / 1000.0
    return out, info


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

    The reindex grid is extended by ONE month past the source's last month,
    so the lag of the newest source observation gets exposed at
    ``source_max + 1 month``. Without that extension, the newest source
    value would never appear in any output row (it would be the lag at
    ``source_max + 1`` which would not exist) — that drops genuinely fresh
    data on the floor and was the root cause of optional capacity/
    transaction lag features missing their most recent month.
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
    full = pd.date_range(
        work[ts_col].min(),
        work[ts_col].max() + pd.DateOffset(months=1),
        freq="MS",
    )
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
    include_capacity: bool = True,
    include_transaction: bool = True,
    include_lng: bool = True,
) -> tuple[pd.DataFrame, dict]:
    """Build monthly features for a single area.

    Returns the feature dataframe AND a side-info dict useful for the DQ
    report (priority dedup decisions, settlement fuel coverage, optional
    capacity/transaction join metadata).

    Optional capacity/transaction features are joined AFTER the baseline
    dropna so that they NEVER change the row count of the feature table
    (per the Codex no-overwrite + row-count invariant contract). Missing
    optional values surface as NaN columns.
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

    # Expose the current-month SMP as a feature under a distinct name. At
    # row M, smp_krw_per_kwh is SMP at M — observed before we forecast M+1,
    # so it is NOT leakage to use it. Excluding it (as the original builder
    # did) forces the "naive" models to lean on smp_lag_1m (= SMP at M-1),
    # which makes naive_lag_1m look like a 2-step seasonal lag instead of a
    # true persistence baseline. We add it under an explicit name so it
    # cannot accidentally collide with the smp_krw_per_kwh column the
    # builder reserves as the raw target source.
    feats["smp_t_observed"] = feats["smp_krw_per_kwh"]

    # Baseline (required) feature_cols — these decide which rows survive dropna.
    # smp_krw_per_kwh stays excluded (it is the "raw target source", duplicated
    # into smp_t_observed for downstream features); the rest is open game.
    baseline_feature_cols = [
        c for c in feats.columns
        if c not in {target_col, "period_month", "area", "smp_krw_per_kwh"}
        and feats[c].dtype != object
    ]
    feats = feats.dropna(subset=baseline_feature_cols + [target_col]).reset_index(drop=True)
    n_baseline_rows = len(feats)

    # ---- Optional capacity/transaction joins (must preserve row count) -----
    optional_info: dict = {}
    if include_capacity:
        cap_fuel_feats, cap_fuel_info = build_capacity_fuel_features(feats["period_month"])
        cap_type_feats, cap_type_info = build_capacity_type_features(feats["period_month"])
        cap_yr_feats, cap_yr_info = build_capacity_yearly_features(feats["period_month"])
        for block in (cap_fuel_feats, cap_type_feats, cap_yr_feats):
            if not block.empty:
                # Deduplicate on period_month so left-join can't fan-out rows.
                block = block.drop_duplicates(subset=["period_month"], keep="last")
                feats = feats.merge(block, on="period_month", how="left")
        optional_info["capacity_fuel"] = cap_fuel_info
        optional_info["capacity_type"] = cap_type_info
        optional_info["capacity_yearly"] = cap_yr_info

    tx_volume_block: pd.DataFrame | None = None
    tx_amount_block: pd.DataFrame | None = None
    if include_transaction:
        tx_volume_block, tx_volume_info = build_transaction_volume_features(
            feats["period_month"]
        )
        tx_amount_block, tx_amount_info = build_transaction_amount_features(
            feats["period_month"]
        )
        for block in (tx_volume_block, tx_amount_block):
            if not block.empty:
                block = block.drop_duplicates(subset=["period_month"], keep="last")
                feats = feats.merge(block, on="period_month", how="left")
        # Price feature is conditional on overlap.
        price_block, price_info = build_transaction_price_features(
            tx_volume_block, tx_amount_block
        )
        if not price_block.empty:
            price_block = price_block.drop_duplicates(subset=["period_month"], keep="last")
            feats = feats.merge(price_block, on="period_month", how="left")
        optional_info["transaction_volume"] = tx_volume_info
        optional_info["transaction_amount"] = tx_amount_info
        optional_info["transaction_price"] = price_info

    if include_lng:
        lng_block, lng_info = build_lng_features(feats["period_month"])
        if not lng_block.empty:
            lng_block = lng_block.drop_duplicates(subset=["period_month"], keep="last")
            feats = feats.merge(lng_block, on="period_month", how="left")
        optional_info["lng_price"] = lng_info

        # JKM forward curve — true leading indicator (round 8). At forecast
        # origin M, contract_month=M+1 price IS observable at end-of-M, so
        # safe to use for forecasting SMP(M+1).
        jkm_block, jkm_info = build_jkm_forward_features(feats["period_month"])
        if not jkm_block.empty:
            jkm_block = jkm_block.drop_duplicates(subset=["period_month"], keep="last")
            feats = feats.merge(jkm_block, on="period_month", how="left")
        optional_info["jkm_forward"] = jkm_info

        # JKM daily volatility (lag_1m) — leak-safe leading-ish signal
        # derived from daily JKM history (much richer than monthly FRED).
        jkm_vol_block, jkm_vol_info = build_jkm_daily_volatility_features(feats["period_month"])
        if not jkm_vol_block.empty:
            jkm_vol_block = jkm_vol_block.drop_duplicates(subset=["period_month"], keep="last")
            feats = feats.merge(jkm_vol_block, on="period_month", how="left")
        optional_info["jkm_daily_volatility"] = jkm_vol_info

    # Hard invariant: optional joins must NOT change the SMP row count.
    if len(feats) != n_baseline_rows:
        raise AssertionError(
            f"Optional feature joins changed row count: "
            f"baseline={n_baseline_rows} after_joins={len(feats)}. "
            "Check for duplicate period_month keys in capacity/transaction sources."
        )

    # ---- Forecast-origin metadata --------------------------------------
    # Explicit per-row metadata that documents the forecast contract:
    #   forecast_origin_month: the month after which the prediction is made
    #     (= period_month — features at this row are derived from
    #     observations whose period_month is ≤ this).
    #   target_month: the month the prediction refers to
    #     (= period_month + horizon_months).
    #   information_cutoff: the last instant of observable information
    #     (end-of-day for the last day of forecast_origin_month).
    #   horizon: human-readable horizon descriptor (e.g. "1M").
    # These columns are NOT used as model features (they would either be
    # constant or trivially derived from period_month) but are written
    # alongside every prediction row so downstream consumers can answer
    # "this prediction was made when, for what month?".
    feats["forecast_origin_month"] = feats["period_month"]
    feats["target_month"] = feats["period_month"] + pd.DateOffset(months=horizon_months)
    # information_cutoff = month-end of period_month, at 23:59:59.
    feats["information_cutoff"] = (
        feats["period_month"]
        + pd.offsets.MonthEnd(0)
        + pd.Timedelta(hours=23, minutes=59, seconds=59)
    )
    feats["horizon"] = f"{horizon_months}M"

    # Forecast-metadata columns are NOT features (they are bookkeeping
    # only). Exclude them explicitly so a future dtype tweak can't sneak
    # them into the model input.
    _forecast_meta_cols = {
        "forecast_origin_month", "target_month", "information_cutoff", "horizon",
    }
    feature_cols = [
        c for c in feats.columns
        if c not in {target_col, "period_month", "area", "smp_krw_per_kwh"}
        and c not in _forecast_meta_cols
        and feats[c].dtype != object
    ]
    _assert_monthly_no_leakage(feats, target_col=target_col, feature_cols=baseline_feature_cols)

    side_info = {
        "area": area,
        "rows": int(len(feats)),
        "feature_cols": feature_cols,
        "baseline_feature_cols": baseline_feature_cols,
        "settlement_fuels": sorted(settlement_fuels),
        "smp_priority_dedup_log": smp_long.attrs.get("_priority_dedup_log"),
        "optional_join_info": optional_info,
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
