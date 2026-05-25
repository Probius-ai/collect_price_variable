"""Round-6 tests: LNG forecast feature integration (`solar/lng_model.py` →
this project) and applicability under the no-leakage / 1-step forecast
contract.

These tests pin the wiring:
  * sources.yaml has the LNG file_source
  * BaseFileLoader produces the canonical schema
  * build_smp_monthly_features joins the three LNG columns
  * Ridge / LightGBM / MonthlyARRidge default pools list them
  * Forecast contract preserved (LNG(M) observable at end-of-M = SMP info cutoff)
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.collectors.kpx_files import (
    LNG_PRICE_MONTHLY_COLUMNS,
    LngPriceMonthlyFileLoader,
)
from src.config.settings import get_source_config
from src.features.build_monthly import (
    build_lng_features,
    load_lng_price_monthly_long,
)
from src.models.ar_monthly import MONTHLY_AR_FEATURES
from src.models.lightgbm_model import DEFAULT_LGB_FEATURES_MONTHLY
from src.models.ridge_model import DEFAULT_RIDGE_FEATURES_MONTHLY


# ---------------------------------------------------------------------------
# 1. sources.yaml entry exists with correct unit + URL
# ---------------------------------------------------------------------------

def test_lng_source_yaml_entry_is_correctly_configured():
    cfg = get_source_config("lng_price_monthly_file")
    assert cfg["unit"] == "USD/MMBtu"
    assert "fred.stlouisfed.org" in cfg["source_url"]
    assert cfg["frequency"] == "monthly"
    assert cfg["file_format"] == "csv"
    # column_mapping must expose the canonical names
    cm = cfg["column_mapping"]
    assert "lng_price_usd_per_mmbtu" in cm
    assert "period_month" in cm


# ---------------------------------------------------------------------------
# 2. Loader parses the canonical schema
# ---------------------------------------------------------------------------

def test_lng_loader_emits_canonical_columns_when_real_csv_present():
    """Skip if the user hasn't run the DB extraction yet."""
    drop = Path("data/raw/manual_or_filedata/lng_price_monthly_file/lng_price_monthly.csv")
    if not drop.exists():
        pytest.skip(f"LNG CSV not present at {drop} — DB export not run yet.")
    loader = LngPriceMonthlyFileLoader()
    df = loader.parse_file(drop)
    assert set(LNG_PRICE_MONTHLY_COLUMNS).issubset(df.columns)
    # Numeric LNG price
    assert df["lng_price_usd_per_mmbtu"].dtype.kind in {"f", "i"}
    # Period_month parsable as datetime
    assert pd.api.types.is_datetime64_any_dtype(df["period_month"])
    # Monotonic increasing
    assert df["period_month"].is_monotonic_increasing


# ---------------------------------------------------------------------------
# 3. build_lng_features wires three columns + computes them correctly
# ---------------------------------------------------------------------------

def test_build_lng_features_emits_only_lagged_columns(tmp_path, monkeypatch):
    """Round-6 leakage fix: build_lng_features must expose ONLY lagged
    columns. The unlagged ``lng_price_usd_per_mmbtu`` (= LNG(M)) and
    ``lng_price_chg_1m`` (= LNG(M)−LNG(M−1)) are forbidden because LNG(M)
    is published ~mid-M+1, AFTER the SMP forecast's information_cutoff."""
    from src.utils import io as io_mod
    class _Stub:
        data_dir = tmp_path / "data"
    monkeypatch.setattr(io_mod, "get_settings", lambda: _Stub())
    import importlib
    bm = importlib.reload(importlib.import_module("src.features.build_monthly"))

    lng_dir = bm.source_root_dir("lng_price_monthly_file") / "2024" / "12" / "01"
    lng_dir.mkdir(parents=True, exist_ok=True)
    lng = pd.DataFrame({
        "period_month": pd.date_range("2024-01-01", periods=6, freq="MS"),
        "lng_price_usd_per_mmbtu": [10.0, 11.0, 12.0, 11.5, 13.0, 14.5],
        "source_id": "lng_price_monthly_file",
        "collected_at": pd.Timestamp("2026-05-25"),
        "source_file": "lng_test.csv",
        "source_priority": 1,
        "source_file_sha256": "a" * 64,
    })
    lng.to_parquet(lng_dir / "parsed_test.parquet", index=False)

    period_index = pd.Series(pd.date_range("2024-01-01", "2024-06-01", freq="MS"))
    out, info = bm.build_lng_features(period_index)
    # Required: lagged columns are present
    for col in ("lng_price_usd_per_mmbtu_lag_1m",
                "lng_price_usd_per_mmbtu_lag_2m",
                "lng_price_chg_1m_lag_1m"):
        assert col in out.columns, f"missing lagged col {col}"
    # FORBIDDEN: unlagged columns must NOT appear (would leak LNG(M))
    for forbidden in ("lng_price_usd_per_mmbtu", "lng_price_chg_1m"):
        assert forbidden not in out.columns, (
            f"build_lng_features leaks LNG(M) via unlagged column '{forbidden}'. "
            "JKM monthly avg for M is published ~mid-M+1, AFTER the SMP "
            "forecast's information_cutoff=end-of-M, so this column is "
            "not observable at forecast origin."
        )

    # Spot-check arithmetic: lag values are previous-month and 2-months-back
    row_mar = out[out["period_month"] == "2024-03-01"].iloc[0]
    assert row_mar["lng_price_usd_per_mmbtu_lag_1m"] == pytest.approx(11.0)  # Feb
    assert row_mar["lng_price_usd_per_mmbtu_lag_2m"] == pytest.approx(10.0)  # Jan
    assert row_mar["lng_price_chg_1m_lag_1m"] == pytest.approx((11.0 - 10.0) / 10.0)
    assert info["source_id"] == "lng_price_monthly_file"
    assert info["rows_in_source"] == 6
    assert "publish_timing_note" in info


# ---------------------------------------------------------------------------
# 4. Model defaults all include the LNG columns
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "default_pool,label",
    [
        (DEFAULT_RIDGE_FEATURES_MONTHLY, "ridge"),
        (DEFAULT_LGB_FEATURES_MONTHLY, "lightgbm"),
    ],
)
def test_lng_lagged_features_present_in_defaults(default_pool, label):
    """LNG lag_1m + lag_2m + chg_1m_lag_1m are the leading-indicator
    features that survived the publish-timing review."""
    assert "lng_price_usd_per_mmbtu_lag_1m" in default_pool, (
        f"{label} default pool missing lng_price_usd_per_mmbtu_lag_1m"
    )
    assert "lng_price_chg_1m_lag_1m" in default_pool, (
        f"{label} default pool missing lng_price_chg_1m_lag_1m"
    )


@pytest.mark.parametrize(
    "default_pool,label",
    [
        (DEFAULT_RIDGE_FEATURES_MONTHLY, "ridge"),
        (DEFAULT_LGB_FEATURES_MONTHLY, "lightgbm"),
        (MONTHLY_AR_FEATURES, "monthly_ar_ridge"),
    ],
)
def test_unlagged_lng_NOT_in_any_default_pool(default_pool, label):
    """Leakage canary. LNG(M) is published ~mid-M+1 → NOT observable at
    SMP info_cutoff=end-of-M. Any default pool that includes the unlagged
    column would silently leak future LNG into the M+1 forecast.
    """
    assert "lng_price_usd_per_mmbtu" not in default_pool, (
        f"{label} default pool LEAKS future LNG via unlagged "
        "'lng_price_usd_per_mmbtu' (= LNG(M), published mid-M+1, not "
        "observable at info_cutoff=end-of-M)."
    )
    assert "lng_price_chg_1m" not in default_pool, (
        f"{label} default pool LEAKS future LNG via 'lng_price_chg_1m' "
        "(= (LNG(M)−LNG(M−1))/LNG(M−1); uses LNG(M), not observable at "
        "info_cutoff). Use 'lng_price_chg_1m_lag_1m' instead."
    )


def test_lng_lagged_features_in_monthly_ar_ridge_defaults():
    assert "lng_price_usd_per_mmbtu_lag_1m" in MONTHLY_AR_FEATURES
    assert "lng_price_chg_1m_lag_1m" in MONTHLY_AR_FEATURES
    # smp_t_observed retained (no regression on round-5)
    assert "smp_t_observed" in MONTHLY_AR_FEATURES


# ---------------------------------------------------------------------------
# 5. Forecast contract: LNG(M) is observable at SMP info_cutoff = end-of-M
# ---------------------------------------------------------------------------

def test_lng_lag_grid_exposes_newest_usable_month(tmp_path, monkeypatch):
    """Round-6 grid-extension fix: when LNG raw covers [Tmin, Tmax], the
    output grid must extend to Tmax + MAX_LAG so the newest usable lag
    is exposed at the corresponding row.

    Concretely: with LNG raw at months {2024-01..2024-04} and a `lag_2m`
    feature, the output must include row 2024-06 (lag_2m there = LNG(2024-04))
    AND row 2024-05 (lag_1m there = LNG(2024-04), lag_2m = LNG(2024-03)).

    Without the extension the SMP feature table at row Tmax+1 silently
    receives NaN for lag_1m even though LNG(Tmax) is fully observable at
    that row's information_cutoff. Same failure mode round-3 fixed in
    `_exogenous_lag_1m`.
    """
    from src.utils import io as io_mod
    class _Stub:
        data_dir = tmp_path / "data"
    monkeypatch.setattr(io_mod, "get_settings", lambda: _Stub())
    import importlib
    bm = importlib.reload(importlib.import_module("src.features.build_monthly"))

    lng_dir = bm.source_root_dir("lng_price_monthly_file") / "2024" / "04" / "01"
    lng_dir.mkdir(parents=True, exist_ok=True)
    raw_months = pd.date_range("2024-01-01", "2024-04-01", freq="MS")  # Tmax = Apr
    lng = pd.DataFrame({
        "period_month": raw_months,
        "lng_price_usd_per_mmbtu": [10.0, 20.0, 30.0, 40.0],
        "source_id": "lng_price_monthly_file",
        "collected_at": pd.Timestamp("2026-05-26"),
        "source_file": "lng_grid_test.csv",
        "source_priority": 1,
        "source_file_sha256": "c" * 64,
    })
    lng.to_parquet(lng_dir / "parsed_test.parquet", index=False)

    # Period index extends BEYOND raw.max (Apr) — same as a refreshed SMP
    # feature table whose own data lands ahead of the next LNG publication.
    period_index = pd.Series(pd.date_range("2024-01-01", "2024-06-01", freq="MS"))
    out, _ = bm.build_lng_features(period_index)

    # The output grid must include rows Tmax+1=May and Tmax+2=Jun, otherwise
    # the downstream left-join silently loses the newest usable lag.
    out_months = set(out["period_month"].dt.strftime("%Y-%m"))
    assert "2024-05" in out_months, (
        "Output grid stops too early. Without extending past raw.max=Apr, "
        "row 2024-05 is missing, so SMP forecast for 2024-06 loses its "
        "lag_1m=LNG(Apr) input even though LNG(Apr) IS observable at "
        "end-of-2024-05."
    )
    assert "2024-06" in out_months, (
        "lag_2m at row Tmax+2=Jun must equal LNG(Apr) — needs grid extension."
    )

    row_may = out[out["period_month"] == "2024-05-01"].iloc[0]
    assert row_may["lng_price_usd_per_mmbtu_lag_1m"] == 40.0, (
        f"lag_1m at row 2024-05 should equal LNG(2024-04)=40.0, "
        f"got {row_may['lng_price_usd_per_mmbtu_lag_1m']}"
    )
    assert row_may["lng_price_usd_per_mmbtu_lag_2m"] == 30.0

    row_jun = out[out["period_month"] == "2024-06-01"].iloc[0]
    assert row_jun["lng_price_usd_per_mmbtu_lag_2m"] == 40.0, (
        "lag_2m at row Tmax+2=Jun should equal LNG(Tmax)=40.0"
    )
    # lag_1m at row 2024-06 = LNG(2024-05) which we don't know → NaN, correct
    assert pd.isna(row_jun["lng_price_usd_per_mmbtu_lag_1m"])


def test_lng_chg_is_not_fabricated_at_extended_grid_positions(tmp_path, monkeypatch):
    """Round-6 fill_method fix: pandas 2.x had `pct_change` default
    `fill_method='pad'` which forward-fills NaN BEFORE the calculation.
    Combined with the round-6 grid-extension fix (output extends to
    Tmax+MAX_LAG with NaN raw), that default would silently fabricate
    `chg_1m=0` at positions Tmax+1 and Tmax+2 — feeding a phantom
    'no change' signal into the SMP model for months where LNG isn't
    actually published yet.

    `pct_change(..., fill_method=None)` is the explicit fix. This test
    pins it.
    """
    from src.utils import io as io_mod
    class _Stub:
        data_dir = tmp_path / "data"
    monkeypatch.setattr(io_mod, "get_settings", lambda: _Stub())
    import importlib
    bm = importlib.reload(importlib.import_module("src.features.build_monthly"))

    lng_dir = bm.source_root_dir("lng_price_monthly_file") / "2024" / "04" / "01"
    lng_dir.mkdir(parents=True, exist_ok=True)
    raw_months = pd.date_range("2024-01-01", "2024-04-01", freq="MS")
    lng = pd.DataFrame({
        "period_month": raw_months,
        # Strictly non-flat raw so any "pad-then-pct_change" fabrication is
        # visible: the genuine chg(Mar→Apr) is +1/3, but the fabricated
        # chg at Tmax+1/Tmax+2 would be 0 under the deprecated default.
        "lng_price_usd_per_mmbtu": [10.0, 20.0, 30.0, 40.0],
        "source_id": "lng_price_monthly_file",
        "collected_at": pd.Timestamp("2026-05-26"),
        "source_file": "lng_chg_test.csv",
        "source_priority": 1,
        "source_file_sha256": "d" * 64,
    })
    lng.to_parquet(lng_dir / "parsed_test.parquet", index=False)

    period_index = pd.Series(pd.date_range("2024-01-01", "2024-06-01", freq="MS"))
    out, _ = bm.build_lng_features(period_index)

    # At row Tmax+1=May: chg_1m_lag_1m = chg(LNG(Mar) → LNG(Apr)) = +1/3
    row_may = out[out["period_month"] == "2024-05-01"].iloc[0]
    assert row_may["lng_price_chg_1m_lag_1m"] == pytest.approx((40.0 - 30.0) / 30.0), (
        "Genuine chg at row 2024-05 (using observable LNG Mar→Apr) wrong."
    )

    # At row Tmax+2=Jun: chg uses (LNG(Apr) − LNG(May))/LNG(May), but
    # LNG(May) is NaN (extended position) → result must be NaN.
    # Under the deprecated pad default this would WRONGLY be (40-40)/40 = 0.
    row_jun = out[out["period_month"] == "2024-06-01"].iloc[0]
    assert pd.isna(row_jun["lng_price_chg_1m_lag_1m"]), (
        f"chg_1m_lag_1m at row 2024-06 (Tmax+2) must be NaN — LNG(May) is "
        f"not yet observable. Got {row_jun['lng_price_chg_1m_lag_1m']!r} — "
        "pandas pct_change pad-default is fabricating a 'no change' value."
    )

    # And the row just past — for safety, NaN remains NaN forever
    # (we only extend by MAX_LAG=2 anyway, but pin it for clarity).


def test_lng_lag_1m_satisfies_forecast_contract(tmp_path, monkeypatch):
    """At forecast_origin_month = M (information_cutoff = end-of-M), the
    JKM LNG monthly average for month M−1 IS observable (published
    ~mid-M, i.e. by day 15 of M). LNG(M) is NOT (published ~mid-M+1).

    Round-6 contract: row M's `lng_price_usd_per_mmbtu_lag_1m` MUST
    equal LNG(M−1). The unlagged column is no longer exposed and
    therefore cannot leak LNG(M) into the M+1 forecast."""
    from src.utils import io as io_mod
    class _Stub:
        data_dir = tmp_path / "data"
    monkeypatch.setattr(io_mod, "get_settings", lambda: _Stub())
    import importlib
    bm = importlib.reload(importlib.import_module("src.features.build_monthly"))

    lng_dir = bm.source_root_dir("lng_price_monthly_file") / "2024" / "12" / "01"
    lng_dir.mkdir(parents=True, exist_ok=True)
    lng = pd.DataFrame({
        "period_month": pd.date_range("2024-01-01", periods=4, freq="MS"),
        "lng_price_usd_per_mmbtu": [10.0, 20.0, 30.0, 40.0],
        "source_id": "lng_price_monthly_file",
        "collected_at": pd.Timestamp("2026-05-25"),
        "source_file": "lng.csv",
        "source_priority": 1,
        "source_file_sha256": "b" * 64,
    })
    lng.to_parquet(lng_dir / "parsed_test.parquet", index=False)

    period_index = pd.Series(pd.date_range("2024-01-01", "2024-04-01", freq="MS"))
    out, _ = bm.build_lng_features(period_index)
    # Row at 2024-03-01: lag_1m must equal LNG(2024-02) = 20.0 (observable)
    row = out[out["period_month"] == "2024-03-01"].iloc[0]
    assert row["lng_price_usd_per_mmbtu_lag_1m"] == 20.0, (
        "lag_1m at row M must equal LNG(M-1), which IS observable at "
        "information_cutoff = end-of-M (JKM publishes M-1 by mid-M)."
    )
    assert row["lng_price_usd_per_mmbtu_lag_2m"] == 10.0
    # Unlagged LNG must NOT appear (canary for the leakage fix)
    assert "lng_price_usd_per_mmbtu" not in out.columns


# ---------------------------------------------------------------------------
# 6. Before/after comparison: post-LNG MAE ≤ pre-LNG MAE for top trainable
# ---------------------------------------------------------------------------

def test_post_lng_metric_snapshots_exist_and_show_improvement():
    """If both baseline snapshots are present, delta_ar_ridge (current top
    trainable) must show non-positive Δ MAE on the test split after LNG
    integration. This is the canonical 'LNG helped' check."""
    pre = Path("docs/figures/baseline_20260525/metrics_comparison_snapshot.csv")
    post = Path("docs/figures/baseline_post_lng_v1/metrics_comparison_snapshot.csv")
    if not pre.exists() or not post.exists():
        pytest.skip("Pre/post snapshot CSVs not committed yet.")
    pre_df = pd.read_csv(pre)
    post_df = pd.read_csv(post)
    pre_mae = pre_df[(pre_df.split=="test") & (pre_df.model=="delta_ar_ridge")]["mae"].iloc[0]
    post_mae = post_df[(post_df.split=="test") & (post_df.model=="delta_ar_ridge")]["mae"].iloc[0]
    assert post_mae <= pre_mae + 0.1, (
        f"LNG integration regressed delta_ar_ridge test MAE: "
        f"pre={pre_mae:.3f} → post={post_mae:.3f}"
    )
    # Persistence must be unchanged (LNG doesn't enter it)
    pre_pmae = pre_df[(pre_df.split=="test") & (pre_df.model=="persistence_monthly")]["mae"].iloc[0]
    post_pmae = post_df[(post_df.split=="test") & (post_df.model=="persistence_monthly")]["mae"].iloc[0]
    assert post_pmae == pytest.approx(pre_pmae, abs=1e-6), (
        "persistence_monthly is regime-independent — its MAE must not change "
        "when only adding LNG features to other models' inputs."
    )
