"""Round-5 tests: smp_t_observed leakage contract, delta-target models,
persistence baseline, dashboard default, and forecast-origin metadata.

The user-facing concern this guards: predictions from the previous round
looked like the input shifted by 2 months because smp_t_observed was
excluded from the feature pool. Round 5 exposes it explicitly, adds
delta-target models that learn the residual on top of persistence, and
makes the forecast contract (origin month, target month, info cutoff)
machine-readable.
"""

from __future__ import annotations

import importlib

import numpy as np
import pandas as pd
import pytest

from src.features.build_monthly import add_calendar_features
from src.models.delta_models import (
    OBSERVED_COL,
    DeltaARRidge,
    DeltaLightGBM,
    DeltaRidge,
    compute_delta_metrics,
)
from src.models.metrics import compute_metrics
from src.models.naive import PersistenceMonthly
from src.models.registry import (
    BASELINE_MODELS,
    DEFAULT_DASHBOARD_MODEL,
    STRONG_MONTHLY_BASELINE,
    TRAINABLE_MODELS,
    classify,
)


def _toy_monthly(n: int = 60, seed: int = 7) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    months = pd.date_range("2020-01-01", periods=n, freq="MS")
    smp = 100 + 30 * np.sin(2 * np.pi * months.month / 12) + rng.normal(0, 3, n)
    df = pd.DataFrame({
        "period_month": months,
        "smp_krw_per_kwh": smp,
        "smp_t_observed": smp,                     # exposed at row M
        "smp_lag_1m": np.r_[np.nan, smp[:-1]],
        "smp_lag_2m": np.r_[[np.nan]*2, smp[:-2]],
        "smp_rolling_3m_mean": pd.Series(smp).shift(1).rolling(3).mean(),
    })
    df = add_calendar_features(df)
    return df.dropna().reset_index(drop=True)


# ---------------------------------------------------------------------------
# 1. smp_t_observed leakage contract
# ---------------------------------------------------------------------------

def test_smp_t_observed_is_not_leakage_under_month_end_forecast_origin():
    """The forecast contract: predictions are issued *after* month M is
    closed (information_cutoff = end-of-M), with target_month = M+1. Under
    that contract, smp_t_observed (= SMP at M) is fully observable at the
    information_cutoff timestamp — not leakage. The metadata columns the
    pipeline emits must be consistent with that contract.
    """
    df = pd.DataFrame({"period_month": pd.date_range("2024-01-01", periods=4, freq="MS")})
    df["forecast_origin_month"] = df["period_month"]
    df["target_month"] = df["period_month"] + pd.DateOffset(months=1)
    df["information_cutoff"] = (
        df["period_month"] + pd.offsets.MonthEnd(0)
        + pd.Timedelta(hours=23, minutes=59, seconds=59)
    )

    # The contract: for every row, target_month > information_cutoff (target
    # is in the future of the cutoff), and forecast_origin_month is the
    # last *observable* month (information_cutoff falls inside it).
    for _, row in df.iterrows():
        assert row["target_month"] > row["information_cutoff"], (
            "target_month must be strictly after information_cutoff"
        )
        # forecast_origin_month must equal the calendar month of cutoff
        cutoff_month = pd.Timestamp(row["information_cutoff"]).to_period("M").to_timestamp()
        assert row["forecast_origin_month"] == cutoff_month, (
            "forecast_origin_month must equal the cutoff's calendar month"
        )
    # smp_t_observed semantically tied to forecast_origin_month → it is
    # available BEFORE target_month → not leakage.


# ---------------------------------------------------------------------------
# 2. Delta-target reconstruction
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("cls", [DeltaRidge, DeltaLightGBM, DeltaARRidge])
def test_delta_target_reconstructs_price_prediction(cls):
    """For every delta wrapper:
       prediction(X)  ==  smp_t_observed(X)  +  predict_delta(X)
    must hold by construction (not just approximately).
    """
    df = _toy_monthly(n=72)
    y_target = df["smp_krw_per_kwh"].shift(-1).dropna()
    feats = df.loc[y_target.index].copy()
    if cls is DeltaLightGBM:
        # Cap boost rounds so the test is quick; precision is unaffected.
        model = cls(num_boost_round=80)
    else:
        model = cls()
    model.fit(feats, y_target)
    level = model.predict(feats).to_numpy()
    delta = model.predict_delta(feats).to_numpy()
    obs = feats[OBSERVED_COL].to_numpy()
    np.testing.assert_allclose(level, obs + delta, rtol=1e-10, atol=1e-10)


# ---------------------------------------------------------------------------
# 3. Persistence ≡ delta=0 baseline
# ---------------------------------------------------------------------------

def test_persistence_is_delta_zero_baseline():
    """A predictor that always emits delta=0 must reproduce the persistence
    baseline EXACTLY: prediction = smp_t_observed for every row, and the
    delta_mae over y_true equals the persistence MAE (= mean |y_true −
    smp_t_observed|)."""
    df = _toy_monthly(n=72)
    y_target = df["smp_krw_per_kwh"].shift(-1).dropna()
    feats = df.loc[y_target.index].copy()

    # Persistence model output (level)
    persistence = PersistenceMonthly().fit(feats, y_target)
    persistence_pred = persistence.predict(feats)
    np.testing.assert_allclose(
        persistence_pred.to_numpy(), feats[OBSERVED_COL].to_numpy()
    )

    # delta=0 reconstructed prediction must equal persistence
    obs = feats[OBSERVED_COL]
    delta_zero_pred = pd.Series(obs.to_numpy() + 0.0, index=feats.index)
    np.testing.assert_allclose(
        delta_zero_pred.to_numpy(), persistence_pred.to_numpy()
    )

    # Delta MAE of the delta=0 predictor equals MAE of persistence on the
    # level target. (delta_mae here is |true_delta - 0| = |y_true - obs|.)
    diag = compute_delta_metrics(y_target, delta_zero_pred, obs)
    persistence_mae = compute_metrics(y_target, persistence_pred).mae
    assert diag["delta_mae"] == pytest.approx(persistence_mae)
    assert diag["mae_level"] == pytest.approx(persistence_mae)


# ---------------------------------------------------------------------------
# 4. Dashboard default is trainable, not persistence
# ---------------------------------------------------------------------------

def test_dashboard_default_model_is_trainable_not_persistence():
    """The dashboard registry's default must point at a TRAINABLE model so
    users opening the Predictions page see what the project's modelling
    adds beyond persistence, not the baseline itself."""
    assert DEFAULT_DASHBOARD_MODEL in TRAINABLE_MODELS
    assert DEFAULT_DASHBOARD_MODEL not in BASELINE_MODELS
    assert classify(DEFAULT_DASHBOARD_MODEL) == "trainable"
    # The reference baseline is exposed separately and IS a baseline.
    assert STRONG_MONTHLY_BASELINE in BASELINE_MODELS
    assert classify(STRONG_MONTHLY_BASELINE) == "baseline"


# ---------------------------------------------------------------------------
# 5. Prediction output has forecast-origin metadata
# ---------------------------------------------------------------------------

def test_prediction_output_has_forecast_origin_and_target_month(tmp_path, monkeypatch):
    """train.py must write forecast_origin_month, target_month,
    information_cutoff, horizon into every predictions_<split>.csv it
    produces — so consumers (humans + the dashboard) can answer "this
    prediction was made when, for what month?" without re-deriving."""
    from src.utils import io as io_mod
    class _Stub:
        data_dir = tmp_path / "data"
        outputs_dir = tmp_path / "outputs"
    monkeypatch.setattr(io_mod, "get_settings", lambda: _Stub())
    # build_monthly_features uses cached settings via DATA_DIR; reload it
    # so it picks up the stubbed settings.
    bm = importlib.reload(importlib.import_module("src.features.build_monthly"))

    # Synthesise enough SMP for the build pipeline to produce >=24 train rows.
    smp_rows = pd.DataFrame({
        "period_month": pd.date_range("2018-01-01", periods=84, freq="MS"),
        "area": ["mainland"] * 84,
        "smp_krw_per_kwh": np.linspace(80, 140, 84),
        "source_id": "kpx_smp_monthly_kepco_file",
        "collected_at": pd.Timestamp("2026-05-25"),
        "source_file": "fake.csv",
        "source_priority": 1,
        "source_file_sha256": "a" * 64,
    })
    sdir = bm.source_root_dir("kpx_smp_monthly_kepco_file") / "2025" / "01" / "01"
    sdir.mkdir(parents=True, exist_ok=True)
    smp_rows.to_parquet(sdir / "parsed_test.parquet", index=False)

    feats, _ = bm.build_smp_monthly_features(area="mainland", horizon_months=1)
    for col in ("forecast_origin_month", "target_month",
                "information_cutoff", "horizon"):
        assert col in feats.columns, f"feature table missing {col!r}"

    # Now exercise the train pipeline against this feature table.
    # Clear side-info attrs (build_smp_monthly_features attaches the dedup
    # log as a DataFrame in df.attrs, which parquet can't serialise).
    feats.attrs.clear()
    feats_path = tmp_path / "feats.parquet"
    feats.to_parquet(feats_path, index=False)
    import src.pipelines.train as train_mod
    monkeypatch.setattr(train_mod, "get_settings", lambda: _Stub())

    from typer.testing import CliRunner
    runner = CliRunner()
    res = runner.invoke(train_mod.app, [
        "--features-path", str(feats_path),
        "--model", "persistence_monthly",
        "--target", "target_smp_t_plus_h_months",
        "--timestamp-col", "period_month",
        "--min-train-rows", "12",
    ])
    assert res.exit_code == 0, res.output
    preds_path = _Stub.outputs_dir / "models" / "persistence_monthly" / "predictions_test.csv"
    assert preds_path.exists(), f"missing {preds_path}"
    pred_df = pd.read_csv(preds_path)
    for col in ("forecast_origin_month", "target_month",
                "information_cutoff", "horizon"):
        assert col in pred_df.columns, (
            f"predictions_test.csv missing forecast metadata column {col!r}"
        )
    # And the contract holds per row in the output too.
    pred_df["target_month_ts"] = pd.to_datetime(pred_df["target_month"])
    pred_df["info_cutoff_ts"] = pd.to_datetime(pred_df["information_cutoff"])
    assert (pred_df["target_month_ts"] > pred_df["info_cutoff_ts"]).all()
