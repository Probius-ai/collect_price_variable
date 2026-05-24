"""Concrete KPX file loaders — round 2 (pre-approval MVP).

All monthly/weekly file loaders emit a long-format dataframe with a stable
canonical schema, so downstream feature builders never need to know which
vendor file or column layout a row came from. Per the round-2 contract:

* Monthly SMP — columns: source_id, period_month, area, smp_krw_per_kwh,
                          collected_at, source_file
* Monthly settlement — columns: source_id, period_month, fuel_type,
                          settlement_unit_price_krw_per_kwh, collected_at, source_file
* REC weekly — columns: source_id, trade_date, rec_volume, avg_price_krw,
                          total_amount_krw, open_price_krw, close_price_krw,
                          base_price_krw, high_price_krw, low_price_krw,
                          collected_at, source_file

The shared CSV/XLSX reader (`_read_tabular`) supports both single-row and
multi-row headers. Multi-row headers are flattened to "level0|level1" after
forward-filling each level across merged cells.
"""

from __future__ import annotations

from datetime import datetime, date
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.collectors.file_loader import BaseFileLoader, FileLoaderError, TBD
from src.utils.time import collected_now_utc


# ---------------------------------------------------------------------------
# Canonical normalization tables
# ---------------------------------------------------------------------------

AREA_LABELS = {
    "mainland": "mainland",
    "육지": "mainland",
    "jeju": "jeju",
    "제주": "jeju",
    "integrated": "integrated",
    "통합": "integrated",
}

MONTHLY_SMP_COLUMNS = [
    "source_id",
    "period_month",
    "area",
    "smp_krw_per_kwh",
    "collected_at",
    "source_file",
]

# Vendor data-quality marker: KPX/EPSIS exports use literal 0.00 in the SMP
# columns to mean "no value published" for that (period, area). Examples:
#   * KEPCO file: 2015-06-01 → mainland/jeju/integrated all 0.00, between
#     valid neighbours (May=68.72, July=66.79).
#   * HOME file: 2001-04 through 2009-12 → mainland & jeju columns are 0.0
#     because pre-2010 the legacy split into mainland/jeju wasn't tracked
#     separately; only 통합 carried a real value.
# We drop these rows from the canonical schema (real KPX SMP has never been
# 0 KRW/kWh in a published month — min ≈ 30 KRW/kWh since 2001). The raw
# file copy is preserved so anyone can re-verify the vendor's literal value.
ZERO_IS_MISSING_THRESHOLD = 1e-9

MONTHLY_SETTLEMENT_COLUMNS = [
    "source_id",
    "period_month",
    "fuel_type",
    "settlement_unit_price_krw_per_kwh",
    "collected_at",
    "source_file",
]

REC_WEEKLY_COLUMNS = [
    "source_id",
    "trade_date",
    "rec_volume",
    "avg_price_krw",
    "total_amount_krw",
    "open_price_krw",
    "close_price_krw",
    "base_price_krw",
    "high_price_krw",
    "low_price_krw",
    "collected_at",
    "source_file",
]


# ---------------------------------------------------------------------------
# Generic tabular reader (CSV + XLSX, single + multi-row headers)
# ---------------------------------------------------------------------------

def _resolve_format(file_path: Path, config: dict[str, Any]) -> str:
    fmt = (config.get("file_format") or "").lower()
    if fmt in {"", TBD.lower()}:
        fmt = file_path.suffix.lower().lstrip(".")
    return fmt


def _resolve_encoding(config: dict[str, Any]) -> str:
    enc = config.get("encoding") or "utf-8"
    if enc == TBD:
        return "utf-8"
    return enc


def _flatten_multi_header(rows: list[list[Any]], sep: str = "|") -> list[str]:
    """Flatten an N-level header into single column names like 'level0|level1'.

    Only the *parent* rows (every row except the last) are forward-filled —
    that's where vendor CSVs use merged cells with a single label followed
    by blanks. The leaf row is taken literally; we deliberately do NOT ffill
    it so that an empty cell at the rightmost column of the leaf row stays
    empty instead of being filled with the previous column's label. Concrete
    example, KPX HOME SMP export:

        row 0 (parent): 기간, SMP,    ,    , BLMP
        row 1 (leaf):       , 육지, 제주, 통합,
                                                ^ empty — must stay empty so
                                                  the BLMP column doesn't end
                                                  up named "BLMP|통합".
    """
    if not rows:
        return []
    n_cols = max(len(r) for r in rows)
    filled: list[list[str]] = []
    last_row_idx = len(rows) - 1
    for idx, row in enumerate(rows):
        row = list(row) + [None] * (n_cols - len(row))
        last = ""
        out_row: list[str] = []
        for cell in row:
            s = (
                "" if cell is None or (isinstance(cell, float) and np.isnan(cell))
                else str(cell).strip()
            )
            if s in {"", "nan", "NaN", "None"}:
                # Only parent rows get ffill — leaf row leaves blanks blank.
                out_row.append(last if idx < last_row_idx else "")
            else:
                out_row.append(s)
                last = s
        filled.append(out_row)
    return [sep.join(parts) for parts in zip(*filled)]


def _read_tabular(file_path: Path, config: dict[str, Any]) -> pd.DataFrame:
    """Read a CSV/XLSX using sources.yaml settings.

    Supports:
        header_row       : int          (single header row)
        header_rows      : list[int]    (multi-row header; flattened with '|')
        drop_rows_before : int          (rows to skip ABOVE the first header row)
        drop_rows_after_header : int    (separator rows to skip BELOW headers)
        sheet_name       : str | int    (xlsx only)
        encoding         : str          (csv only)
        file_format      : csv | xlsx | xls | tsv  (else inferred)
    """
    fmt = _resolve_format(file_path, config)
    skip = int(config.get("drop_rows_before", 0) or 0)
    drop_after = int(config.get("drop_rows_after_header", 0) or 0)
    multi_rows = config.get("header_rows")

    if multi_rows:
        # We always read with header=None and do flattening ourselves so the
        # behaviour is identical across CSV and XLSX. pd.read_csv(header=[..])
        # does its own munging that doesn't ffill merged cells reliably.
        if fmt in {"csv", "tsv"}:
            sep = "\t" if fmt == "tsv" else ","
            raw = pd.read_csv(
                file_path,
                sep=sep,
                encoding=_resolve_encoding(config),
                header=None,
                skiprows=skip,
            )
        elif fmt in {"xlsx", "xls"}:
            sheet = config.get("sheet_name") if config.get("sheet_name") is not None else 0
            raw = pd.read_excel(file_path, sheet_name=sheet, header=None, skiprows=skip)
        else:
            raise FileLoaderError(f"Unsupported file_format={fmt!r} for {file_path}")

        n_header_rows = len(multi_rows)
        # multi_rows are indices *relative to the data after skiprows*
        # but for the round-2 use case they're contiguous from row 0.
        header_block = [raw.iloc[i].tolist() for i in multi_rows]
        flat_cols = _flatten_multi_header(header_block, sep="|")
        body = raw.iloc[max(multi_rows) + 1:].copy()
        body.columns = flat_cols
        body = body.reset_index(drop=True)
        if drop_after:
            body = body.iloc[drop_after:].reset_index(drop=True)
        return body

    header_row = int(config.get("header_row", 0) or 0)
    if fmt in {"csv", "tsv"}:
        sep = "\t" if fmt == "tsv" else ","
        df = pd.read_csv(
            file_path,
            sep=sep,
            encoding=_resolve_encoding(config),
            skiprows=skip,
            header=header_row,
        )
    elif fmt in {"xlsx", "xls"}:
        sheet = config.get("sheet_name") if config.get("sheet_name") is not None else 0
        df = pd.read_excel(file_path, sheet_name=sheet, skiprows=skip, header=header_row)
    else:
        raise FileLoaderError(f"Unsupported file_format={fmt!r} for {file_path}")

    if drop_after:
        df = df.iloc[drop_after:].reset_index(drop=True)
    df.columns = [str(c).strip() for c in df.columns]
    return df


# ---------------------------------------------------------------------------
# Shared parser helpers
# ---------------------------------------------------------------------------

def _rename_with_mapping(df: pd.DataFrame, mapping: dict[str, str]) -> pd.DataFrame:
    unresolved = [k for k, v in mapping.items() if v == TBD]
    if unresolved:
        raise FileLoaderError(
            f"column_mapping still has TBD entries: {unresolved}. "
            "Open the raw file and fill these in sources.yaml first."
        )
    missing_in_file = [v for v in mapping.values() if v not in df.columns]
    if missing_in_file:
        raise FileLoaderError(
            f"Vendor columns from mapping missing in the actual file: {missing_in_file}. "
            f"Available columns are: {list(df.columns)}"
        )
    return df.rename(columns={vendor: internal for internal, vendor in mapping.items()})


def _coerce_numeric(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    for col in cols:
        if col in df.columns:
            df[col] = pd.to_numeric(
                df[col].astype(str).str.replace(",", "", regex=False),
                errors="coerce",
            )
    return df


def _parse_period(series: pd.Series, fmt: str | None) -> pd.Series:
    """Parse a period string into a month-start Timestamp."""
    if not fmt or fmt == TBD:
        return pd.to_datetime(series, errors="raise").dt.to_period("M").dt.to_timestamp()
    return (
        pd.to_datetime(series.astype(str), format=fmt, errors="raise")
        .dt.to_period("M")
        .dt.to_timestamp()
    )


def _now_str() -> pd.Timestamp:
    return pd.Timestamp(collected_now_utc())


def _melt_and_clean_smp(
    df: pd.DataFrame, file_path: Path, source_name: str
) -> tuple[pd.DataFrame, dict[str, int]]:
    """Melt wide (mainland/jeju/integrated) → long, then drop missing rows.

    Vendor uses 0 as a "no published value" marker (see ZERO_IS_MISSING_THRESHOLD
    above). We replace those with NaN before the final dropna step and return
    a per-area count of zero→missing coercions so the DQ report can surface it.
    """
    value_cols = ["smp_mainland", "smp_jeju", "smp_integrated"]
    area_map = {
        "smp_mainland": "mainland",
        "smp_jeju": "jeju",
        "smp_integrated": "integrated",
    }
    long = df.melt(
        id_vars=["period_month"],
        value_vars=value_cols,
        var_name="area_raw",
        value_name="smp_krw_per_kwh",
    )
    long["area"] = long["area_raw"].map(area_map)
    # Count and coerce the literal-0 rows BEFORE dropping NaNs.
    zero_mask = (
        long["smp_krw_per_kwh"].notna()
        & (long["smp_krw_per_kwh"].abs() < ZERO_IS_MISSING_THRESHOLD)
    )
    zero_summary = (
        long.loc[zero_mask].groupby("area").size().astype(int).to_dict()
    )
    long.loc[zero_mask, "smp_krw_per_kwh"] = np.nan
    long = long.dropna(subset=["smp_krw_per_kwh"])
    long["source_id"] = source_name
    long["collected_at"] = _now_str()
    long["source_file"] = file_path.name
    return (
        long[MONTHLY_SMP_COLUMNS].sort_values(["period_month", "area"]).reset_index(drop=True),
        zero_summary,
    )


# ===========================================================================
# Monthly SMP loaders
# ===========================================================================


class KpxSmpMonthlyKepcoFileLoader(BaseFileLoader):
    """KEPCO official 'mainland/Jeju/integrated monthly SMP' single-CSV loader.

    Vendor schema (single header row, cp949):
        년도, 월, 육지계통한계가격, 제주계통한계가격, 통합계통한계가격

    Output (long format):
        source_id, period_month, area, smp_krw_per_kwh, collected_at, source_file
    """

    source_name = "kpx_smp_monthly_kepco_file"

    def required_columns(self) -> set[str]:
        return set(MONTHLY_SMP_COLUMNS)

    def parse_file(self, file_path: Path) -> pd.DataFrame:
        self.assert_no_tbd_in_config("column_mapping")
        df = _read_tabular(file_path, self.config)
        df = _rename_with_mapping(df, self.config["column_mapping"])
        df = _coerce_numeric(df, ["period_year", "period_month_num"])
        df = df.dropna(subset=["period_year", "period_month_num"]).copy()
        df["period_year"] = df["period_year"].astype(int)
        df["period_month_num"] = df["period_month_num"].astype(int)
        df["period_month"] = pd.to_datetime(
            dict(
                year=df["period_year"],
                month=df["period_month_num"],
                day=1,
            )
        )
        df = _coerce_numeric(df, ["smp_mainland", "smp_jeju", "smp_integrated"])
        long, zero_summary = _melt_and_clean_smp(df, file_path, self.source_name)
        long.attrs["_zero_coerced_rows"] = zero_summary
        return long


class KpxSmpMonthlyHomeAvgFileLoader(BaseFileLoader):
    """HOME (EPSIS) weighted-average monthly SMP file. Multi-row header.

    Vendor schema (2-row header; row 0 has merged 'SMP' across 3 columns):
        row 0: 기간, SMP,  , , BLMP
        row 1:      , 육지, 제주, 통합,
    After forward-fill + 'level0|level1':
        기간|, SMP|육지, SMP|제주, SMP|통합, BLMP|

    Output (long format identical to KEPCO loader).
    """

    source_name = "kpx_smp_monthly_home_avg_file"

    def required_columns(self) -> set[str]:
        return set(MONTHLY_SMP_COLUMNS)

    def parse_file(self, file_path: Path) -> pd.DataFrame:
        self.assert_no_tbd_in_config("column_mapping")
        # The HOME source ships in both .csv and .xlsx. The same column-mapping
        # works because both end up with the same flattened header.
        df = _read_tabular(file_path, self.config)
        df = _rename_with_mapping(df, self.config["column_mapping"])
        # `period` parses as YYYY/MM. Vendor sometimes ships footers ("총계") that
        # don't match the format → coerce errors to NaT and drop them.
        df["period_month"] = pd.to_datetime(
            df["period"].astype(str),
            format=self.config.get("period_format") or "%Y/%m",
            errors="coerce",
        )
        df = _coerce_numeric(df, ["smp_mainland", "smp_jeju", "smp_integrated"])
        df = df.dropna(subset=["period_month"])

        long, zero_summary = _melt_and_clean_smp(df, file_path, self.source_name)
        long.attrs["_zero_coerced_rows"] = zero_summary
        return long


# ===========================================================================
# Monthly settlement (wide → long)
# ===========================================================================


class KpxSettlementMonthlyFileLoader(BaseFileLoader):
    """Wide-by-fuel monthly settlement unit price → long-format loader.

    The vendor file is wide:
        기간, 원자력, 석탄_유연탄, 석탄_무연탄, 석탄_계, 유류, LNG, 양수,
        신재생_연료전지, 신재생_석탄가스화, 신재생_태양, 신재생_풍력,
        신재생_수력, 신재생_해양, 신재생_바이오, 신재생_폐기물,
        신재생_계, 기타, 합계

    We melt to long format and only emit rows whose fuel column is in the
    canonical `fuel_type_mapping`. Renewable subcategories (신재생_연료전지 …
    신재생_폐기물) are excluded — the rollup `신재생_계 → renewable` is kept.
    The set of *actually-dropped* columns is recorded by the DQ report.
    """

    source_name = "kpx_settlement_monthly_file"

    def required_columns(self) -> set[str]:
        return set(MONTHLY_SETTLEMENT_COLUMNS)

    def parse_file(self, file_path: Path) -> pd.DataFrame:
        df = _read_tabular(file_path, self.config)
        period_col = self.config.get("period_column") or "기간"
        fuel_map: dict[str, str] = self.config.get("fuel_type_mapping") or {}
        if not fuel_map:
            raise FileLoaderError(
                "kpx_settlement_monthly_file.fuel_type_mapping is empty. "
                "Populate sources.yaml with the wide column → canonical fuel_type map."
            )
        if period_col not in df.columns:
            raise FileLoaderError(
                f"period column {period_col!r} not in file headers {list(df.columns)}"
            )

        # Mark drop side-info in an attr; the loader entry point persists it
        # via the DQ report writer.
        present_fuels = [c for c in fuel_map if c in df.columns]
        missing_fuels = [c for c in fuel_map if c not in df.columns]
        df.attrs["_dropped_columns"] = sorted(
            set(df.columns) - {period_col} - set(present_fuels)
        )
        df.attrs["_missing_canonical_fuels"] = missing_fuels

        df["period_month"] = _parse_period(
            df[period_col], self.config.get("period_format")
        )
        df = _coerce_numeric(df, present_fuels)
        long = df.melt(
            id_vars=["period_month"],
            value_vars=present_fuels,
            var_name="fuel_type_raw",
            value_name="settlement_unit_price_krw_per_kwh",
        )
        long["fuel_type"] = long["fuel_type_raw"].map(fuel_map)
        long = long.dropna(subset=["fuel_type", "settlement_unit_price_krw_per_kwh"])
        long["source_id"] = self.source_name
        long["collected_at"] = _now_str()
        long["source_file"] = file_path.name
        out = (
            long[MONTHLY_SETTLEMENT_COLUMNS]
            .sort_values(["period_month", "fuel_type"])
            .reset_index(drop=True)
        )
        out.attrs["_dropped_columns"] = df.attrs["_dropped_columns"]
        out.attrs["_missing_canonical_fuels"] = df.attrs["_missing_canonical_fuels"]
        return out


# ===========================================================================
# REC weekly (rows are trade days, despite "weekly" in vendor naming)
# ===========================================================================


class KpxRecWeeklyFileLoader(BaseFileLoader):
    """REC spot trading-day rows with full canonical price schema."""

    source_name = "kpx_rec_weekly_file"

    def required_columns(self) -> set[str]:
        return set(REC_WEEKLY_COLUMNS)

    def parse_file(self, file_path: Path) -> pd.DataFrame:
        self.assert_no_tbd_in_config("column_mapping")
        df = _read_tabular(file_path, self.config)
        df = _rename_with_mapping(df, self.config["column_mapping"])
        fmt = self.config.get("trade_date_format")
        if fmt and fmt != TBD:
            df["trade_date"] = pd.to_datetime(
                df["trade_date"].astype(str), format=fmt, errors="coerce"
            )
        else:
            df["trade_date"] = pd.to_datetime(df["trade_date"], errors="coerce")
        df = df.dropna(subset=["trade_date"])
        numeric_cols = [c for c in REC_WEEKLY_COLUMNS if c not in {
            "source_id", "trade_date", "collected_at", "source_file"
        }]
        df = _coerce_numeric(df, numeric_cols)
        df["source_id"] = self.source_name
        df["collected_at"] = _now_str()
        df["source_file"] = file_path.name
        return (
            df[REC_WEEKLY_COLUMNS]
            .sort_values("trade_date")
            .reset_index(drop=True)
        )


# ===========================================================================
# Yearly SMP — placeholder loader (still TBD this round; no live file)
# ===========================================================================


class KpxSmpYearlyLoader(BaseFileLoader):
    source_name = "kpx_smp_yearly"

    def parse_file(self, file_path: Path) -> pd.DataFrame:
        # Still TBD this round. We intentionally hard-fail rather than silently
        # produce empty data so a user who points this at the *misnamed*
        # HOME_..._SMP.csv (which is actually generation data) gets a clear
        # error instead of an empty parquet.
        raise FileLoaderError(
            "kpx_smp_yearly loader is not implemented this round. "
            "A clean yearly SMP file has not been verified yet — the only "
            "candidate (HOME_전력거래_계통한계가격_가중평균SMP.csv) was misnamed "
            "and is quarantined as kpx_generation_yearly. "
            "Populate sources.yaml::file_sources.kpx_smp_yearly with a verified "
            "yearly SMP export before implementing this loader."
        )

    def required_columns(self) -> set[str]:
        return set()


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

LOADERS: dict[str, type[BaseFileLoader]] = {
    KpxSmpMonthlyKepcoFileLoader.source_name: KpxSmpMonthlyKepcoFileLoader,
    KpxSmpMonthlyHomeAvgFileLoader.source_name: KpxSmpMonthlyHomeAvgFileLoader,
    KpxSettlementMonthlyFileLoader.source_name: KpxSettlementMonthlyFileLoader,
    KpxRecWeeklyFileLoader.source_name: KpxRecWeeklyFileLoader,
    KpxSmpYearlyLoader.source_name: KpxSmpYearlyLoader,
}
