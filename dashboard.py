"""KPX SMP forecasting dashboard.

Run:
    streamlit run dashboard.py

Pages:
    1. Overview      — project state, data sources loaded, row counts
    2. Models        — comparison.csv visualised (MAE/RMSE/MAPE × split)
    3. Predictions   — actual vs predicted time series per model
    4. Features      — explore smp_monthly_<area>_h1m.parquet (feature table)
    5. Data Quality  — dedup log, sources side-info

This is a read-only viewer over artefacts the existing pipelines produce.
It does NOT trigger collectors, loaders, or training — those are CLI-driven
under src/pipelines/.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src.models.registry import (
    BASELINE_MODELS,
    DEFAULT_DASHBOARD_MODEL,
    STRONG_MONTHLY_BASELINE,
    TRAINABLE_MODELS,
    classify,
)


# ---------------------------------------------------------------------------
# Paths + cached loaders
# ---------------------------------------------------------------------------

ROOT = Path(__file__).resolve().parent
DATA_PROCESSED = ROOT / "data" / "processed"
DATA_RAW_KPX = ROOT / "data" / "raw" / "kpx"
DROP_ROOT = ROOT / "data" / "raw" / "manual_or_filedata"
OUTPUTS = ROOT / "outputs"
METRICS_CSV = OUTPUTS / "metrics" / "comparison.csv"
MODELS_DIR = OUTPUTS / "models"
DQ_DIR = OUTPUTS / "data_quality"


@st.cache_data(show_spinner=False)
def load_metrics_comparison() -> pd.DataFrame:
    if not METRICS_CSV.exists():
        return pd.DataFrame()
    return pd.read_csv(METRICS_CSV)


@st.cache_data(show_spinner=False)
def list_models() -> list[str]:
    if not MODELS_DIR.exists():
        return []
    return sorted(p.name for p in MODELS_DIR.iterdir() if p.is_dir())


@st.cache_data(show_spinner=False)
def load_predictions(model_name: str, split: str) -> pd.DataFrame:
    path = MODELS_DIR / model_name / f"predictions_{split}.csv"
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_csv(path)
    if "period_month" in df.columns:
        df["period_month"] = pd.to_datetime(df["period_month"])
    return df


@st.cache_data(show_spinner=False)
def load_feature_table(area: str) -> tuple[pd.DataFrame, dict | None]:
    path = DATA_PROCESSED / f"smp_monthly_{area}_h1m.parquet"
    if not path.exists():
        return pd.DataFrame(), None
    df = pd.read_parquet(path)
    if "period_month" in df.columns:
        df["period_month"] = pd.to_datetime(df["period_month"])
    side_path = path.with_suffix(".sideinfo.json")
    side = json.loads(side_path.read_text()) if side_path.exists() else None
    return df, side


@st.cache_data(show_spinner=False)
def list_source_inventory() -> pd.DataFrame:
    """Walk data/raw/kpx/<source>/.../parsed_*.parquet to build a roll-up."""
    rows = []
    if not DATA_RAW_KPX.exists():
        return pd.DataFrame(columns=["source", "snapshots", "rows", "min", "max"])
    for source_dir in sorted(DATA_RAW_KPX.iterdir()):
        if not source_dir.is_dir():
            continue
        parquets = sorted(source_dir.rglob("parsed_*.parquet"))
        if not parquets:
            rows.append({
                "source": source_dir.name, "snapshots": 0, "rows": 0,
                "min": None, "max": None,
            })
            continue
        total = 0
        min_p, max_p = None, None
        for p in parquets:
            try:
                df = pd.read_parquet(p)
            except Exception:
                continue
            total += len(df)
            for col in ("period_month", "trade_date", "interval_end"):
                if col in df.columns and len(df):
                    pmin = pd.to_datetime(df[col]).min()
                    pmax = pd.to_datetime(df[col]).max()
                    if min_p is None or pmin < min_p:
                        min_p = pmin
                    if max_p is None or pmax > max_p:
                        max_p = pmax
                    break
            if "period_year" in df.columns and len(df):
                ymin, ymax = int(df["period_year"].min()), int(df["period_year"].max())
                if min_p is None:
                    min_p = pd.Timestamp(f"{ymin}-01-01")
                    max_p = pd.Timestamp(f"{ymax}-12-31")
        rows.append({
            "source": source_dir.name,
            "snapshots": len(parquets),
            "rows": total,
            "min": str(min_p)[:10] if min_p is not None else "—",
            "max": str(max_p)[:10] if max_p is not None else "—",
        })
    return pd.DataFrame(rows)


@st.cache_data(show_spinner=False)
def list_drop_inbox() -> pd.DataFrame:
    rows = []
    if not DROP_ROOT.exists():
        return pd.DataFrame()
    for sub in sorted(DROP_ROOT.iterdir()):
        if not sub.is_dir() or sub.name in {"incoming_for_quarantine", "quarantine"}:
            continue
        files = [p for p in sub.iterdir() if p.is_file()]
        rows.append({
            "source": sub.name,
            "files_in_inbox": len(files),
            "filenames": ", ".join(f.name for f in files) or "—",
        })
    return pd.DataFrame(rows)


@st.cache_data(show_spinner=False)
def load_latest_dq_report() -> dict | None:
    if not DQ_DIR.exists():
        return None
    reports = sorted(DQ_DIR.glob("report_*.json"))
    if not reports:
        return None
    return json.loads(reports[-1].read_text())


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="KPX SMP Forecast Dashboard",
    page_icon="⚡",
    layout="wide",
)

st.title("⚡ KPX SMP Forecast Dashboard")
st.caption(
    "Pre-approval MVP — file-based pipeline. "
    "Read-only viewer over artefacts that `src/pipelines/*` produced."
)

page = st.sidebar.radio(
    "Page",
    [
        "1. Overview",
        "2. Models",
        "3. Predictions",
        "4. Features",
        "5. Data Quality",
        "6. Walk-forward CV",
    ],
)

# ---------------------------------------------------------------------------
# Page 1: Overview
# ---------------------------------------------------------------------------

if page == "1. Overview":
    st.header("Overview")

    col1, col2, col3, col4 = st.columns(4)
    inventory = list_source_inventory()
    metrics = load_metrics_comparison()
    models = list_models()
    drop = list_drop_inbox()

    col1.metric("Sources loaded", int((inventory["snapshots"] > 0).sum()) if not inventory.empty else 0)
    col2.metric("Total snapshots", int(inventory["snapshots"].sum()) if not inventory.empty else 0)
    col3.metric("Models trained", len(models))
    col4.metric("Drop-inbox sources", len(drop))

    st.subheader("Parsed parquet inventory")
    st.caption("Each source's date-partitioned parsed_*.parquet snapshots.")
    if inventory.empty:
        st.warning("No `data/raw/kpx/<source>/` snapshots yet. Run `src.pipelines.load_files` first.")
    else:
        st.dataframe(inventory, width="stretch", hide_index=True)

    st.subheader("Drop-inbox (`data/raw/manual_or_filedata/<source>/`)")
    st.caption("Files awaiting `load_files` ingestion. Files copied here will get parsed on next run.")
    if drop.empty:
        st.info("Drop-inbox empty.")
    else:
        st.dataframe(drop, width="stretch", hide_index=True)


# ---------------------------------------------------------------------------
# Page 2: Models
# ---------------------------------------------------------------------------

elif page == "2. Models":
    st.header("Model comparison")
    metrics = load_metrics_comparison()
    if metrics.empty:
        st.warning("No `outputs/metrics/comparison.csv` yet. Run `src.pipelines.evaluate`.")
    else:
        splits = sorted(metrics["split"].unique())
        split = st.radio("Split", splits, index=splits.index("test") if "test" in splits else 0, horizontal=True)
        sub = metrics[metrics["split"] == split].sort_values("mae").copy()
        sub["kind"] = sub["model"].map(classify)

        st.subheader(f"Metrics — {split}")
        st.caption(
            f"`{STRONG_MONTHLY_BASELINE}` is the strong baseline; trainable "
            "models must beat it by a meaningful MAE margin to add value."
        )
        st.dataframe(
            sub[["model", "kind", "mae", "rmse", "mape", "smape", "r2",
                 "directional_accuracy", "peak_precision", "peak_recall",
                 "peak_f1", "peak_threshold", "n_observations"]]
            .round(3),
            width="stretch", hide_index=True,
        )

        col1, col2 = st.columns(2)
        with col1:
            fig = px.bar(sub, x="model", y="mae",
                         title=f"MAE by model — {split}",
                         color="model", text="mae")
            fig.update_traces(texttemplate="%{text:.2f}", textposition="outside")
            fig.update_layout(showlegend=False)
            st.plotly_chart(fig, width="stretch")
        with col2:
            fig = px.bar(sub, x="model", y="mape",
                         title=f"MAPE [%] by model — {split}",
                         color="model", text="mape")
            fig.update_traces(texttemplate="%{text:.2f}", textposition="outside")
            fig.update_layout(showlegend=False)
            st.plotly_chart(fig, width="stretch")

        st.subheader("All splits — MAE")
        fig = px.bar(metrics.sort_values(["split", "mae"]),
                     x="model", y="mae", color="split", barmode="group",
                     title="MAE across train/valid/test")
        st.plotly_chart(fig, width="stretch")


# ---------------------------------------------------------------------------
# Page 3: Predictions
# ---------------------------------------------------------------------------

elif page == "3. Predictions":
    st.header("Actual vs predicted")

    st.info(
        "**Persistence baseline note.** "
        "`persistence_monthly` predicts next-month SMP as the current month's "
        "observed SMP, so its line **naturally follows the actual series with "
        "a one-month delay**. That is the floor any trainable model must beat. "
        "True lag reduction beyond persistence requires leading exogenous "
        "variables (e.g. next-month gas-price futures) or higher-frequency "
        "(hourly/day-ahead) API data."
    )

    models = list_models()
    if not models:
        st.warning("No models trained yet. Run `src.pipelines.train`.")
    else:
        # Default selection: the registry's recommended trainable model, so
        # the dashboard never opens on a baseline by accident.
        default_idx = (
            models.index(DEFAULT_DASHBOARD_MODEL)
            if DEFAULT_DASHBOARD_MODEL in models
            else 0
        )
        col1, col2, col3, col4 = st.columns(4)
        sel = col1.multiselect(
            "Trainable model(s) to plot",
            [m for m in models if classify(m) != "baseline"],
            default=[models[default_idx]] if classify(models[default_idx]) == "trainable"
                    else [DEFAULT_DASHBOARD_MODEL]
                    if DEFAULT_DASHBOARD_MODEL in models else [],
        )
        split = col2.radio("Split", ["test", "valid", "train"], horizontal=True, index=0)
        show_persistence = col3.checkbox(
            "Show persistence baseline",
            value=True,
            help=f"Overlay `{STRONG_MONTHLY_BASELINE}` as a strong-baseline "
                 "reference line. A trainable model that does not visibly "
                 "outperform this is not adding value.",
        )
        show_actual = col4.checkbox("Show actual", value=True)

        if not sel and not show_persistence:
            st.info("Pick at least one model or enable the baseline overlay.")
        else:
            fig = go.Figure()
            actual_added = False

            # Helper: build a hovertemplate that surfaces forecast origin
            # and target month so users always see which month a prediction
            # refers to and what info it was based on.
            def _hover_template():
                return (
                    "<b>%{customdata[0]}</b><br>"
                    "forecast_for_month: %{customdata[1]}<br>"
                    "information_cutoff_month: %{customdata[2]}<br>"
                    "value: %{y:.2f} KRW/kWh<extra></extra>"
                )

            def _custom_data(df, model_label):
                # Fall back to period_month derived columns if the meta
                # columns weren't included in the prediction CSV (older runs).
                tgt = df["target_month"] if "target_month" in df.columns else (
                    pd.to_datetime(df["period_month"]) + pd.DateOffset(months=1)
                )
                cutoff = df["forecast_origin_month"] if "forecast_origin_month" in df.columns else df["period_month"]
                return list(zip(
                    [model_label] * len(df),
                    [str(t)[:10] for t in pd.to_datetime(tgt)],
                    [str(c)[:10] for c in pd.to_datetime(cutoff)],
                ))

            def _add_trace(df, name, dash, color=None, width=2.0):
                fig.add_trace(go.Scatter(
                    x=df["period_month"], y=df["y_pred"],
                    name=name, mode="lines+markers",
                    line=dict(dash=dash, color=color, width=width),
                    customdata=_custom_data(df, name),
                    hovertemplate=_hover_template(),
                ))

            # Optional persistence overlay first (so it draws beneath models)
            if show_persistence and STRONG_MONTHLY_BASELINE in models:
                df_b = load_predictions(STRONG_MONTHLY_BASELINE, split)
                if not df_b.empty and "period_month" in df_b.columns:
                    df_b = df_b.sort_values("period_month")
                    _add_trace(df_b,
                               f"baseline: {STRONG_MONTHLY_BASELINE}",
                               dash="dashdot", color="#9aa0a6", width=1.5)

            for m in sel:
                df = load_predictions(m, split)
                if df.empty or "period_month" not in df.columns:
                    st.warning(f"{m}/{split}: no predictions on disk.")
                    continue
                df = df.sort_values("period_month")
                if show_actual and not actual_added:
                    # Plot actual at the TARGET month if metadata is present,
                    # otherwise at period_month (legacy behaviour). This
                    # makes it clear what the model is being judged against.
                    x_actual = (
                        df["target_month"] if "target_month" in df.columns
                        else df["period_month"]
                    )
                    actual_custom = list(zip(
                        ["Actual"] * len(df),
                        [str(t)[:10] for t in pd.to_datetime(x_actual)],
                        ["—"] * len(df),
                    ))
                    fig.add_trace(go.Scatter(
                        x=df["period_month"], y=df["y_true"],
                        name="Actual SMP", mode="lines+markers",
                        line=dict(color="black", width=2.5),
                        customdata=actual_custom,
                        hovertemplate=_hover_template(),
                    ))
                    actual_added = True
                _add_trace(df, f"{m} (pred)", dash="dot")

            fig.update_layout(
                title=f"SMP forecast vs actual ({split}, KRW/kWh)",
                xaxis_title="period_month (= forecast_origin_month)",
                yaxis_title="SMP [KRW/kWh]",
                hovermode="x unified",
            )
            st.plotly_chart(fig, width="stretch")

            # Residual + delta-direction table
            st.subheader("Residual + delta summary")
            rows = []
            extra_models = list(sel) + (
                [STRONG_MONTHLY_BASELINE]
                if show_persistence and STRONG_MONTHLY_BASELINE in models else []
            )
            for m in extra_models:
                df = load_predictions(m, split)
                if df.empty:
                    continue
                resid = df["y_pred"] - df["y_true"]
                entry = {
                    "model": m,
                    "kind": classify(m),
                    "n": len(df),
                    "mean_residual": float(resid.mean()),
                    "mean_abs_residual": float(resid.abs().mean()),
                    "n_underpred": int((resid < 0).sum()),
                    "n_overpred": int((resid > 0).sum()),
                }
                if "true_delta_1m" in df.columns and "predicted_delta_1m" in df.columns:
                    td = df["true_delta_1m"].to_numpy(dtype=float)
                    pd_ = df["predicted_delta_1m"].to_numpy(dtype=float)
                    entry["delta_mae"] = float(abs(td - pd_).mean())
                    mask = td != 0
                    entry["delta_direction_accuracy"] = (
                        float(((td > 0) == (pd_ > 0))[mask].mean())
                        if mask.any() else float("nan")
                    )
                rows.append(entry)
            if rows:
                st.dataframe(pd.DataFrame(rows).round(3),
                             width="stretch", hide_index=True)


# ---------------------------------------------------------------------------
# Page 4: Features
# ---------------------------------------------------------------------------

elif page == "4. Features":
    st.header("Monthly feature table")
    area = st.radio("Area", ["mainland", "jeju", "integrated"], horizontal=True)
    df, side = load_feature_table(area)
    if df.empty:
        st.warning(
            f"No `smp_monthly_{area}_h1m.parquet` yet. "
            "Run `src.pipelines.build_monthly_features --area " + area + " --horizon-months 1`."
        )
    else:
        c1, c2, c3 = st.columns(3)
        c1.metric("Rows", len(df))
        c2.metric("Columns", len(df.columns))
        c3.metric("Period range",
                  f"{df['period_month'].min():%Y-%m} — {df['period_month'].max():%Y-%m}")

        # Group columns by prefix for the picker
        prefix_groups = {
            "Baseline (SMP lag / rolling / calendar)": [
                c for c in df.columns
                if c.startswith(("smp_lag_", "smp_rolling_", "month_", "is_", "year", "quarter"))
                and c != "smp_krw_per_kwh"
            ],
            "Settlement (lag_1m)": [c for c in df.columns if c.startswith("settlement_")],
            "Capacity — fuel (lag_1m)": [c for c in df.columns if c.startswith("capacity_fuel_")],
            "Capacity — generation type (lag_1m)": [c for c in df.columns if c.startswith("capacity_type_")],
            "Capacity — yearly broadcast (lag_1y)": [c for c in df.columns if c.startswith("capacity_yearly_")],
            "Transaction — volume (lag_1m)": [c for c in df.columns if c.startswith("transaction_volume_")],
            "Transaction — amount (lag_1m)": [c for c in df.columns if c.startswith("transaction_amount_")],
            "Transaction — derived price (lag_1m)": [c for c in df.columns if c.startswith("market_trade_price_")],
        }

        st.subheader("Coverage by feature group")
        cov_rows = []
        for grp, cols in prefix_groups.items():
            if not cols:
                continue
            cov_rows.append({
                "feature_group": grp,
                "n_cols": len(cols),
                "avg_non_null_pct": float(
                    (df[cols].notna().sum().sum() / (len(df) * len(cols))) * 100
                ),
            })
        st.dataframe(pd.DataFrame(cov_rows).round(1),
                     width="stretch", hide_index=True)

        st.subheader("SMP target + lag features over time")
        plot_cols = ["smp_krw_per_kwh"]
        if "target_smp_t_plus_h_months" in df.columns:
            plot_cols.append("target_smp_t_plus_h_months")
        if "smp_lag_1m" in df.columns:
            plot_cols.append("smp_lag_1m")
        if "smp_rolling_12m_mean" in df.columns:
            plot_cols.append("smp_rolling_12m_mean")
        fig = px.line(df, x="period_month", y=plot_cols,
                      title=f"SMP ({area}) — KRW/kWh")
        st.plotly_chart(fig, width="stretch")

        st.subheader("Pick an optional-feature group to plot")
        opts = [k for k, v in prefix_groups.items() if v and "Baseline" not in k]
        chosen = st.selectbox("Group", opts) if opts else None
        if chosen:
            cols = prefix_groups[chosen]
            picked = st.multiselect("Features to plot",
                                    cols, default=cols[: min(5, len(cols))])
            if picked:
                fig = px.line(df, x="period_month", y=picked,
                              title=f"{chosen} — {area}")
                st.plotly_chart(fig, width="stretch")

        with st.expander("Raw feature table (last 30 rows)"):
            st.dataframe(df.tail(30), width="stretch")

        if side:
            with st.expander("Side-info (build_monthly_features output)"):
                # Hide the bulky dedup log unless explicitly requested
                shown = {k: v for k, v in side.items() if k != "smp_priority_dedup_log"}
                st.json(shown, expanded=False)


# ---------------------------------------------------------------------------
# Page 5: Data Quality
# ---------------------------------------------------------------------------

elif page == "5. Data Quality":
    st.header("Data quality + revision tracking")
    report = load_latest_dq_report()
    if report is None:
        st.warning(
            "No DQ report yet. Run `python -m src.pipelines.dq_report` "
            "(reports land under `outputs/data_quality/`)."
        )
    else:
        st.caption(f"Latest DQ report: {report.get('generated_at', '—')}")

        sources = report.get("sources", {})
        if sources:
            st.subheader("Per-source DQ summary")
            sources_df = pd.DataFrame([
                {
                    "source": s,
                    "rows": info.get("rows", 0),
                    "runs": info.get("runs", 0),
                    "unit": info.get("unit"),
                    "frequency": info.get("frequency"),
                    "latest_collected_at": info.get("latest_collected_at"),
                    "limitations": len(info.get("limitations", [])),
                }
                for s, info in sources.items()
            ])
            st.dataframe(sources_df, width="stretch", hide_index=True)

        if report.get("quarantine"):
            st.subheader("Quarantined files (filename-content mismatch)")
            qdf = pd.DataFrame(report["quarantine"])
            st.dataframe(qdf[["source_id", "reason", "original_path", "sha256"]],
                         width="stretch", hide_index=True)

        st.subheader("Pick an area to inspect dedup log")
        area = st.radio("Area", ["mainland", "jeju", "integrated"], horizontal=True,
                         key="dq_area")
        _, side = load_feature_table(area)
        log = side.get("smp_priority_dedup_log", []) if side else []
        if not log:
            st.info(f"No dedup conflicts logged for {area}.")
        else:
            log_df = pd.DataFrame(log)
            st.write(f"Total conflict rows: {len(log_df)} "
                     f"({(log_df['role'] == 'selected').sum()} selected, "
                     f"{(log_df['role'] == 'dropped').sum()} dropped)")
            if "reason" in log_df.columns:
                st.write("Drop reason counts:")
                st.write(
                    log_df[log_df["role"] == "dropped"]["reason"].value_counts()
                    .to_frame("count")
                )
            with st.expander("Full dedup log (first 200 rows)"):
                st.dataframe(log_df.head(200), width="stretch")


# ---------------------------------------------------------------------------
# Page 6: Walk-forward CV
# ---------------------------------------------------------------------------

elif page == "6. Walk-forward CV":
    st.header("Walk-forward CV")
    st.caption(
        "Each step refits on all-data-before(t) and predicts row t. "
        "More realistic than the single-split test because it mirrors "
        "production deployment under a non-stationary regime."
    )
    wf_dir = OUTPUTS / "walk_forward"
    if not wf_dir.exists():
        st.warning(
            "No walk-forward results yet. Run "
            "`python -m src.pipelines.walk_forward main --features-path "
            "data/processed/smp_monthly_mainland_h1m.parquet --model <m>` "
            "for each model, then `walk_forward compare`."
        )
    else:
        comp_csv = wf_dir / "comparison.csv"
        if comp_csv.exists():
            cmp = pd.read_csv(comp_csv)
            st.subheader("Aggregate metrics (all walk-forward steps)")
            st.dataframe(
                cmp.round(3),
                width="stretch", hide_index=True,
            )
            fig = px.bar(cmp.sort_values("mae"), x="model", y="mae",
                         color="model", text="mae",
                         title="Walk-forward MAE by model")
            fig.update_traces(texttemplate="%{text:.2f}", textposition="outside")
            fig.update_layout(showlegend=False)
            st.plotly_chart(fig, width="stretch")
        else:
            st.info("No comparison.csv yet — run `walk_forward compare`.")

        st.subheader("Per-model walk-forward predictions")
        wf_models = sorted(
            p.name for p in wf_dir.iterdir()
            if p.is_dir() and (p / "predictions.csv").exists()
        )
        if wf_models:
            sel = st.multiselect("Models to overlay", wf_models,
                                  default=wf_models[: min(3, len(wf_models))])
            fig = go.Figure()
            actual_drawn = False
            for m in sel:
                df = pd.read_csv(wf_dir / m / "predictions.csv")
                df["period_month"] = pd.to_datetime(df["period_month"])
                df = df.sort_values("period_month")
                if not actual_drawn:
                    fig.add_trace(go.Scatter(
                        x=df["period_month"], y=df["y_true"],
                        name="Actual", mode="lines+markers",
                        line=dict(color="black", width=2.5),
                    ))
                    actual_drawn = True
                fig.add_trace(go.Scatter(
                    x=df["period_month"], y=df["y_pred"],
                    name=f"{m} (pred)", mode="lines+markers",
                    line=dict(dash="dot"),
                ))
            fig.update_layout(
                title="Walk-forward: actual vs predicted SMP",
                xaxis_title="period_month", yaxis_title="SMP [KRW/kWh]",
                hovermode="x unified",
            )
            st.plotly_chart(fig, width="stretch")


# ---------------------------------------------------------------------------
# Footer
# ---------------------------------------------------------------------------

st.sidebar.markdown("---")
st.sidebar.caption(
    "Read-only viewer. Pipelines (`load_files`, `build_monthly_features`, "
    "`train`, `evaluate`, `dq_report`) are CLI-driven; this dashboard never "
    "writes to the data tree."
)
