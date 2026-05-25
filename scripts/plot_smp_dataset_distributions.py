"""Distribution plots for the SMP monthly feature dataset.

Reads data/processed/smp_monthly_*_h1m.parquet and writes a multi-panel PNG
to outputs/eda/ summarising target + key feature-group distributions.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
PROC = ROOT / "data" / "processed"
OUT = ROOT / "outputs" / "eda"
OUT.mkdir(parents=True, exist_ok=True)


def load_areas() -> dict[str, pd.DataFrame]:
    areas = {}
    for a in ["mainland", "jeju", "integrated"]:
        p = PROC / f"smp_monthly_{a}_h1m.parquet"
        if p.exists():
            df = pd.read_parquet(p)
            df["period_month"] = pd.to_datetime(df["period_month"])
            areas[a] = df
    return areas


def plot_target_timeseries(areas: dict[str, pd.DataFrame], ax) -> None:
    colors = {"mainland": "#1f77b4", "jeju": "#d62728", "integrated": "#2ca02c"}
    for a, df in areas.items():
        ax.plot(df["period_month"], df["smp_t_observed"], label=a, lw=1.4, color=colors.get(a, None))
    ax.set_title("Monthly SMP (observed) by area")
    ax.set_ylabel("KRW / kWh")
    ax.set_xlabel("period_month")
    ax.legend(loc="upper left")
    ax.grid(alpha=0.3)


def plot_target_hist(areas: dict[str, pd.DataFrame], ax) -> None:
    for a, df in areas.items():
        vals = df["smp_t_observed"].dropna()
        ax.hist(vals, bins=30, alpha=0.45, label=f"{a} (n={len(vals)})")
    ax.set_title("SMP distribution (KRW/kWh)")
    ax.set_xlabel("SMP")
    ax.set_ylabel("count")
    ax.legend()
    ax.grid(alpha=0.3)


def plot_seasonality_box(df: pd.DataFrame, ax) -> None:
    df = df.copy()
    df["month"] = df["period_month"].dt.month
    data = [df.loc[df["month"] == m, "smp_t_observed"].dropna().values for m in range(1, 13)]
    ax.boxplot(data, tick_labels=[str(m) for m in range(1, 13)], showfliers=True)
    ax.set_title("Mainland SMP by calendar month")
    ax.set_xlabel("month")
    ax.set_ylabel("KRW / kWh")
    ax.grid(alpha=0.3)


def _hist_grid(df: pd.DataFrame, cols: list[str], axes, title_prefix: str = "") -> None:
    for ax, col in zip(axes, cols):
        s = df[col].dropna()
        if s.empty:
            ax.set_title(f"{title_prefix}{col}\n(no data)")
            ax.axis("off")
            continue
        if s.dtype == bool:
            s = s.astype(int)
        ax.hist(s.values, bins=30, color="#4c78a8", edgecolor="white")
        ax.set_title(f"{title_prefix}{col}\nn={len(s)}", fontsize=9)
        ax.tick_params(axis="x", labelsize=7)
        ax.tick_params(axis="y", labelsize=7)
        ax.grid(alpha=0.25)


def plot_feature_group_distributions(df: pd.DataFrame) -> None:
    groups: dict[str, list[str]] = {
        "01_smp_self_lags": [
            "smp_t_observed", "smp_lag_1m", "smp_lag_3m",
            "smp_lag_12m", "smp_rolling_12m_mean", "smp_rolling_12m_std",
        ],
        "02_calendar": [
            "month", "quarter", "is_summer", "is_winter", "month_sin", "month_cos",
        ],
        "03_settlement_unit_price": [
            "settlement_unit_price_coal_total_lag_1m",
            "settlement_unit_price_lng_lag_1m",
            "settlement_unit_price_nuclear_lag_1m",
            "settlement_unit_price_oil_lag_1m",
            "settlement_unit_price_renewable_lag_1m",
            "settlement_unit_price_total_lag_1m",
        ],
        "04_capacity_fuel_mw": [
            "capacity_fuel_nuclear_mw_lag_1m",
            "capacity_fuel_lng_mw_lag_1m",
            "capacity_fuel_coal_total_mw_lag_1m",
            "capacity_fuel_renewable_total_mw_lag_1m",
            "capacity_fuel_renewable_solar_mw_lag_1m",
            "capacity_fuel_renewable_wind_mw_lag_1m",
        ],
        "05_capacity_shares": [
            "capacity_fuel_nuclear_share_lag_1m",
            "capacity_fuel_lng_share_lag_1m",
            "capacity_fuel_coal_total_share_lag_1m",
            "capacity_fuel_renewable_total_share_lag_1m",
            "capacity_type_combined_cycle_share_lag_1m",
            "capacity_type_renewable_share_lag_1m",
        ],
        "06_transactions": [
            "transaction_volume_lng_mwh_lag_1m",
            "transaction_volume_coal_total_mwh_lag_1m",
            "transaction_volume_renewable_total_mwh_lag_1m",
            "transaction_amount_lng_krw_lag_1m",
            "transaction_amount_coal_total_krw_lag_1m",
            "transaction_amount_total_krw_lag_1m",
        ],
        "07_lng_jkm": [
            "lng_price_usd_per_mmbtu_lag_1m",
            "lng_price_usd_per_mmbtu_lag_2m",
            "lng_price_chg_1m_lag_1m",
            "jkm_daily_mean_lag_1m",
            "jkm_daily_std_lag_1m",
            "jkm_daily_range_lag_1m",
        ],
    }
    for name, cols in groups.items():
        cols = [c for c in cols if c in df.columns]
        if not cols:
            continue
        n = len(cols)
        ncols = 3
        nrows = int(np.ceil(n / ncols))
        fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 4.2, nrows * 2.8))
        axes = np.atleast_2d(axes).ravel()
        _hist_grid(df, cols, axes)
        for ax in axes[n:]:
            ax.axis("off")
        fig.suptitle(f"Mainland SMP feature distribution — {name}", y=1.0, fontsize=12)
        fig.tight_layout()
        out = OUT / f"smp_dist_{name}.png"
        fig.savefig(out, dpi=130, bbox_inches="tight")
        plt.close(fig)
        print(f"wrote {out}")


def plot_coverage(df: pd.DataFrame) -> None:
    feature_cols = [
        c for c in df.columns
        if c not in {"period_month", "area", "smp_krw_per_kwh", "horizon",
                     "forecast_origin_month", "target_month", "information_cutoff",
                     "target_smp_t_plus_h_months"}
        and df[c].dtype != object
    ]
    counts = df[feature_cols].notna().sum().sort_values()
    fig, ax = plt.subplots(figsize=(10, max(10, len(feature_cols) * 0.18)))
    ax.barh(range(len(counts)), counts.values, color="#4c78a8")
    ax.set_yticks(range(len(counts)))
    ax.set_yticklabels(counts.index, fontsize=7)
    ax.axvline(len(df), color="red", lw=0.8, ls="--", label=f"total rows = {len(df)}")
    ax.set_xlabel("non-null rows")
    ax.set_title("Mainland: feature coverage (non-null counts)")
    ax.legend()
    ax.grid(alpha=0.3, axis="x")
    fig.tight_layout()
    out = OUT / "smp_feature_coverage.png"
    fig.savefig(out, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}")


def plot_overview(areas: dict[str, pd.DataFrame]) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
    plot_target_timeseries(areas, axes[0])
    plot_target_hist(areas, axes[1])
    if "mainland" in areas:
        plot_seasonality_box(areas["mainland"], axes[2])
    fig.tight_layout()
    out = OUT / "smp_target_overview.png"
    fig.savefig(out, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}")


def main() -> None:
    areas = load_areas()
    if not areas:
        raise SystemExit("No processed SMP parquet found in data/processed/")
    plot_overview(areas)
    if "mainland" in areas:
        plot_feature_group_distributions(areas["mainland"])
        plot_coverage(areas["mainland"])
    print("done")


if __name__ == "__main__":
    main()
