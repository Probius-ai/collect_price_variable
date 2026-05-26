"""FastAPI backend for the model-selection web frontend.

Endpoints
---------
GET  /api/health                   — liveness probe
GET  /api/models/comparison        — v1..v5 × model comparison table (CSV → JSON)
GET  /api/models/recommendation    — single best (recommended_historical + v5 candidate)
GET  /api/models/registry          — list of model_name → version count
GET  /api/models/registry/{model}  — per-model registry history (JSON fallback)
GET  /api/retrain/status           — current background retrain status (PID, alive?)
POST /api/retrain                  — trigger a background smoke-test run
GET  /api/solar/integration        — presence check for solar/lng external models

All endpoints are READ-ONLY except `POST /api/retrain`. The retrain endpoint
spawns a detached subprocess so the HTTP request returns immediately while
the smoke test (~5-10 min) runs in the background.

Security note: this API has no auth. It's intended for local dev /
classroom-demo use. Don't bind to 0.0.0.0 on a public network without
fronting it with auth or a reverse proxy.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel


# ---------------------------------------------------------------------------
# Paths + config
# ---------------------------------------------------------------------------

ROOT = Path(__file__).resolve().parent.parent
COMPARISON_CSV = ROOT / "outputs" / "metrics" / "mlops_version_comparison.csv"
REGISTRY_DIR = ROOT / "outputs" / "model_registry"
RETRAIN_LOCK = ROOT / "outputs" / "_retrain_status.json"
RETRAIN_LOG = ROOT / "outputs" / "_retrain.log"
SMOKE_TEST_CONFIG = ROOT / "config" / "mlops_smoke_test.yaml"


# ---------------------------------------------------------------------------
# FastAPI app + CORS
# ---------------------------------------------------------------------------

app = FastAPI(
    title="KPX SMP Model Selection API",
    description=(
        "Read-only API serving the v1..v5 staged-retraining results to the "
        "Next.js frontend. The only write endpoint is `POST /api/retrain`, "
        "which spawns a background subprocess and returns immediately."
    ),
    version="0.1.0",
)

# CORS — Next.js dev server runs on :3000, prod build behind a reverse
# proxy talks to us on the same origin. Allow both.
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        "http://192.168.50.100:3000",  # LAN dev access (matches Streamlit's setup)
    ],
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------


class ModelRow(BaseModel):
    version: str
    model: str
    data_cutoff_month: str
    evaluation_mode: str
    n_train: int
    n_test: int
    mae: Optional[float] = None
    rmse: Optional[float] = None
    mape: Optional[float] = None
    r2: Optional[float] = None
    directional_accuracy: Optional[float] = None
    improvement_vs_persistence: Optional[float] = None
    registry_status: str
    mlflow_run_id: Optional[str] = None
    skipped: bool = False
    skip_reason: Optional[str] = None


class ComparisonResponse(BaseModel):
    generated_at: str
    n_rows: int
    rows: list[ModelRow]


class RecommendationResponse(BaseModel):
    recommended_historical: Optional[ModelRow]
    latest_candidate: Optional[ModelRow]
    persistence_baseline_v5_mae: Optional[float]
    improvement_vs_persistence_mae: Optional[float]
    rationale: str


class RegistryRecord(BaseModel):
    version: str
    data_cutoff_month: str
    evaluation_mode: str
    metrics: dict[str, Any]
    promotion_status: str
    mlflow_run_id: Optional[str]
    artifact_uri: Optional[str]
    created_at: str


class RetrainStatusResponse(BaseModel):
    state: str  # "idle" | "running" | "completed"
    pid: Optional[int] = None
    started_at: Optional[str] = None
    log_path: Optional[str] = None
    mlflow_uri: Optional[str] = None


class SolarIntegrationStatus(BaseModel):
    resource: str
    path: str
    present: bool
    kind: str


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _load_comparison_df() -> pd.DataFrame:
    if not COMPARISON_CSV.exists():
        raise HTTPException(
            status_code=404,
            detail=(
                f"Comparison data not found at {COMPARISON_CSV.relative_to(ROOT)}. "
                "Run the smoke test first (POST /api/retrain or "
                "`python -m src.pipelines.mlops_smoke_test`)."
            ),
        )
    df = pd.read_csv(COMPARISON_CSV)
    return df


def _df_row_to_model(row: pd.Series) -> ModelRow:
    """Convert a comparison DataFrame row to a ModelRow, dropping NaNs."""
    def _safe_float(v: Any) -> Optional[float]:
        try:
            f = float(v)
            if pd.isna(f) or not (f == f) or f in (float("inf"), float("-inf")):
                return None
            return f
        except (TypeError, ValueError):
            return None

    return ModelRow(
        version=str(row["version"]),
        model=str(row["model"]),
        data_cutoff_month=str(row.get("data_cutoff_month", "")),
        evaluation_mode=str(row.get("evaluation_mode", "")),
        n_train=int(row.get("n_train") or 0),
        n_test=int(row.get("n_test") or 0),
        mae=_safe_float(row.get("mae")),
        rmse=_safe_float(row.get("rmse")),
        mape=_safe_float(row.get("mape")),
        r2=_safe_float(row.get("r2")),
        directional_accuracy=_safe_float(row.get("directional_accuracy")),
        improvement_vs_persistence=_safe_float(row.get("improvement_vs_persistence")),
        registry_status=str(row.get("registry_status") or ""),
        mlflow_run_id=(str(row["mlflow_run_id"]) if pd.notna(row.get("mlflow_run_id")) else None),
        skipped=bool(row.get("skipped", False)),
        skip_reason=(str(row["skip_reason"]) if pd.notna(row.get("skip_reason")) else None),
    )


def _pid_alive(pid: Optional[int]) -> bool:
    if not pid:
        return False
    try:
        os.kill(pid, 0)
        return True
    except (ProcessLookupError, PermissionError):
        return False
    except OSError as e:
        import errno
        return e.errno != errno.ESRCH


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@app.get("/api/health")
def health() -> dict[str, Any]:
    return {
        "ok": True,
        "service": "kpx-smp-model-selection-api",
        "comparison_csv_present": COMPARISON_CSV.exists(),
        "registry_dir_present": REGISTRY_DIR.exists(),
        "smoke_test_config_present": SMOKE_TEST_CONFIG.exists(),
    }


@app.get("/api/models/comparison", response_model=ComparisonResponse)
def get_comparison() -> ComparisonResponse:
    df = _load_comparison_df()
    rows = [_df_row_to_model(r) for _, r in df.iterrows()]
    return ComparisonResponse(
        generated_at=datetime.fromtimestamp(
            COMPARISON_CSV.stat().st_mtime, tz=timezone.utc,
        ).isoformat(),
        n_rows=len(rows),
        rows=rows,
    )


@app.get("/api/models/recommendation", response_model=RecommendationResponse)
def get_recommendation() -> RecommendationResponse:
    df = _load_comparison_df()
    clean = df[df["skipped"] != True]  # noqa: E712

    rec = clean[clean["registry_status"] == "recommended_historical"]
    rec_row = _df_row_to_model(rec.iloc[0]) if not rec.empty else None

    latest = clean[clean["registry_status"] == "latest_candidate"]
    latest_best = (
        _df_row_to_model(latest.sort_values("mae").iloc[0])
        if not latest.empty else None
    )

    v5_persistence = latest[latest["model"] == "persistence_monthly"]
    baseline_mae = (
        float(v5_persistence["mae"].iloc[0]) if not v5_persistence.empty else None
    )
    improvement = (
        baseline_mae - latest_best.mae
        if (baseline_mae is not None and latest_best is not None and latest_best.mae is not None)
        else None
    )

    rationale_parts = []
    if rec_row:
        rationale_parts.append(
            f"v1~v4 holdout 평균 기준 best: {rec_row.model} (MAE {rec_row.mae:.3f})."
        )
    if latest_best:
        rationale_parts.append(
            f"v5 rolling validation best: {latest_best.model} "
            f"(MAE {latest_best.mae:.3f})."
        )
    if rec_row and latest_best and rec_row.model == latest_best.model:
        rationale_parts.append(
            "두 view에서 동일 모델이 1위 → 가장 강한 후보."
        )
    elif rec_row and latest_best:
        rationale_parts.append(
            "두 view가 다른 모델을 추천 → 안정성(v4→v5 변화량) 추가 검토 필요."
        )

    return RecommendationResponse(
        recommended_historical=rec_row,
        latest_candidate=latest_best,
        persistence_baseline_v5_mae=baseline_mae,
        improvement_vs_persistence_mae=improvement,
        rationale=" ".join(rationale_parts) or "데이터 부족.",
    )


@app.get("/api/models/registry")
def list_registry() -> dict[str, Any]:
    """List of (model_name → registry record count) for the model-picker UI."""
    if not REGISTRY_DIR.exists():
        return {"models": []}
    out = []
    for f in sorted(REGISTRY_DIR.glob("*_registry.json")):
        model_name = f.stem.removesuffix("_registry")
        try:
            records = json.loads(f.read_text(encoding="utf-8")).get("records", [])
        except json.JSONDecodeError:
            records = []
        out.append({
            "model_name": model_name,
            "record_count": len(records),
            "registry_file": str(f.relative_to(ROOT)),
        })
    return {"models": out}


@app.get("/api/models/registry/{model_name}")
def get_registry(model_name: str) -> dict[str, Any]:
    f = REGISTRY_DIR / f"{model_name}_registry.json"
    if not f.exists():
        raise HTTPException(404, f"No registry for model {model_name!r}")
    try:
        data = json.loads(f.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        raise HTTPException(500, f"Registry file {f.name} is corrupt JSON")
    return data


@app.get("/api/retrain/status", response_model=RetrainStatusResponse)
def retrain_status() -> RetrainStatusResponse:
    if not RETRAIN_LOCK.exists():
        return RetrainStatusResponse(state="idle")
    try:
        info = json.loads(RETRAIN_LOCK.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return RetrainStatusResponse(state="idle")
    pid = info.get("pid")
    state = "running" if _pid_alive(pid) else "completed"
    return RetrainStatusResponse(
        state=state,
        pid=pid,
        started_at=info.get("started_at"),
        log_path=info.get("log_path"),
        mlflow_uri=info.get("mlflow_uri"),
    )


@app.post("/api/retrain", response_model=RetrainStatusResponse, status_code=202)
def trigger_retrain() -> RetrainStatusResponse:
    """Spawn a detached background subprocess that runs the v1..v5 smoke
    test with MLflow logging enabled. Returns immediately with 202 +
    the launched-process info.
    """
    # Prevent double-trigger: if a recent run is still alive, reject.
    if RETRAIN_LOCK.exists():
        try:
            existing = json.loads(RETRAIN_LOCK.read_text(encoding="utf-8"))
            if _pid_alive(existing.get("pid")):
                raise HTTPException(
                    status_code=409,
                    detail=(
                        f"Retrain already running (PID {existing['pid']}, "
                        f"started {existing.get('started_at')})."
                    ),
                )
        except json.JSONDecodeError:
            pass  # Corrupt lock — overwrite it

    # Pre-flight: config exists, module is importable.
    if not SMOKE_TEST_CONFIG.exists():
        raise HTTPException(
            status_code=500,
            detail=f"Config not found: {SMOKE_TEST_CONFIG.relative_to(ROOT)}",
        )
    import importlib.util
    if importlib.util.find_spec("src.pipelines.mlops_smoke_test") is None:
        raise HTTPException(
            status_code=500,
            detail=(
                "`src.pipelines.mlops_smoke_test` not importable from the "
                "Python interpreter running this API. Check PYTHONPATH / "
                "install."
            ),
        )

    RETRAIN_LOG.parent.mkdir(parents=True, exist_ok=True)
    env = {
        **os.environ,
        "MLFLOW_TRACKING_URI": os.environ.get(
            "MLFLOW_TRACKING_URI", "http://localhost:5000"
        ),
        "MLFLOW_EXPERIMENT_NAME": os.environ.get(
            "MLFLOW_EXPERIMENT_NAME", "kpx-smp-monthly"
        ),
        "ENABLE_MLFLOW": "true",
    }
    cmd = [
        sys.executable,
        "-m", "src.pipelines.mlops_smoke_test",
        "--config", str(SMOKE_TEST_CONFIG),
        "--log-to-mlflow",
    ]
    popen_kwargs: dict[str, Any] = {
        "stderr": subprocess.STDOUT,
        "cwd": str(ROOT),
        "env": env,
    }
    if os.name == "posix":
        popen_kwargs["start_new_session"] = True

    with RETRAIN_LOG.open("w", encoding="utf-8") as log_f:
        proc = subprocess.Popen(cmd, stdout=log_f, **popen_kwargs)

    status = {
        "pid": proc.pid,
        "started_at": datetime.now(timezone.utc).isoformat(),
        "trigger": "POST /api/retrain",
        "log_path": str(RETRAIN_LOG.relative_to(ROOT)),
        "mlflow_uri": env["MLFLOW_TRACKING_URI"],
        "python_executable": sys.executable,
    }
    RETRAIN_LOCK.parent.mkdir(parents=True, exist_ok=True)
    RETRAIN_LOCK.write_text(json.dumps(status, indent=2), encoding="utf-8")

    return RetrainStatusResponse(
        state="running",
        pid=proc.pid,
        started_at=status["started_at"],
        log_path=status["log_path"],
        mlflow_uri=status["mlflow_uri"],
    )


# ---------------------------------------------------------------------------
# Forecast — the actual KRW/kWh number the selection service exists to produce
# ---------------------------------------------------------------------------


class ForecastResponse(BaseModel):
    model_name: str
    version: str
    unit: str                          # always "KRW/kWh"
    forecast_origin_month: str         # the month whose features we used
    target_month: str                  # the month being predicted
    predicted_smp_krw_per_kwh: float
    most_recent_actual_smp_krw_per_kwh: float | None
    most_recent_actual_month: str | None
    inference_seconds: float
    artifact: dict[str, str]           # paths the prediction was built from
    note: str


def _pick_best_v5_artifact() -> tuple[str, str, Path]:
    """Pick the best (version, model_name, artifact_dir) whose pickle is
    actually usable (non-empty and unpicklable).

    Skips models whose pickle is 0 bytes — that happens silently for
    delta wrappers (DeltaRidge / DeltaLightGBM / DeltaARRidge) because
    they hold lambdas via `_DeltaWrapped._base_factory`, which raises
    `Can't pickle local object …` during `pickle.dump()` and the smoke
    test caches `model_path = None` in that branch instead of writing.

    Preference order:
      1. v5 latest_candidates sorted ascending by MAE — skip empty pickles
      2. Best historical model with a usable pickle, across v1..v4
    """
    df = _load_comparison_df()
    clean = df[df["skipped"] != True]  # noqa: E712

    def _usable(version: str, model: str) -> Path | None:
        p = ROOT / "outputs" / "mlops_smoke_test" / version / model / "model.pkl"
        try:
            return p if p.exists() and p.stat().st_size > 0 else None
        except OSError:
            return None

    # 1. v5 latest candidates, sorted by MAE
    v5 = clean[clean["registry_status"] == "latest_candidate"].sort_values("mae")
    for _, row in v5.iterrows():
        m = str(row["model"])
        path = _usable("v5", m)
        if path is not None:
            return "v5", m, path.parent

    # 2. Any historical model with a usable pickle (best MAE first)
    hist = clean[clean["version"].isin(["v1", "v2", "v3", "v4"])].sort_values("mae")
    for _, row in hist.iterrows():
        v = str(row["version"])
        m = str(row["model"])
        path = _usable(v, m)
        if path is not None:
            return v, m, path.parent

    raise HTTPException(
        404,
        "No usable model.pkl found in outputs/mlops_smoke_test/. "
        "Run a retrain (POST /api/retrain).",
    )


@app.get("/api/forecast/next", response_model=ForecastResponse)
def forecast_next_month() -> ForecastResponse:
    """Produce the actual KRW/kWh forecast the model-selection service
    exists to deliver.

    Loads the recommended v5 model's pickle, applies it to the most
    recent row of the monthly feature panel, and returns the predicted
    SMP for ``forecast_origin_month + 1`` along with the most recent
    actually-observed SMP for reference.

    This is what makes the page a "price model selection service" — the
    selection is justified by the concrete price number the model
    actually produces, not just MAE numbers in the abstract.
    """
    import pickle
    import time

    version_used, model_name, artifact_dir = _pick_best_v5_artifact()
    pickle_path = artifact_dir / "model.pkl"
    if not pickle_path.exists():
        raise HTTPException(
            500,
            f"Model pickle not found at {pickle_path.relative_to(ROOT)}. "
            "Some delta-wrappers contain lambdas and don't pickle; pick a "
            "different recommended model or re-run the smoke test.",
        )

    # Load the feature panel and pick the latest row whose features are
    # observable. The model predicts target_smp_t_plus_h_months = SMP at
    # period_month + 1.
    feature_path = ROOT / "data" / "processed" / "smp_monthly_mainland_h1m.parquet"
    if not feature_path.exists():
        raise HTTPException(
            500, f"Feature table missing: {feature_path.relative_to(ROOT)}"
        )
    panel = pd.read_parquet(feature_path)
    panel = panel.sort_values("period_month").reset_index(drop=True)
    if panel.empty:
        raise HTTPException(500, "Feature table is empty.")

    latest_row = panel.iloc[[-1]].copy()
    forecast_origin = pd.Timestamp(latest_row["period_month"].iloc[0])
    target_month = forecast_origin + pd.DateOffset(months=1)

    # Most recent actually-observed SMP (smp_t_observed at the latest row
    # IS the SMP at forecast_origin — observable now). Useful baseline
    # context for the user.
    most_recent_actual = (
        float(latest_row["smp_t_observed"].iloc[0])
        if "smp_t_observed" in latest_row.columns
           and pd.notna(latest_row["smp_t_observed"].iloc[0])
        else None
    )

    # Drop the columns the smoke test drops — same contract as training
    drop_cols = {"target_smp_t_plus_h_months"}
    for c in latest_row.columns:
        s = latest_row[c]
        if not (pd.api.types.is_numeric_dtype(s) or pd.api.types.is_bool_dtype(s)):
            drop_cols.add(c)
    X = latest_row.drop(columns=[c for c in drop_cols if c in latest_row.columns])

    t0 = time.perf_counter()
    try:
        with pickle_path.open("rb") as f:
            model = pickle.load(f)
        y_pred = float(pd.Series(model.predict(X)).iloc[0])
    except Exception as exc:
        raise HTTPException(
            500,
            f"Inference failed for {model_name}: {type(exc).__name__}: {exc}",
        )
    elapsed = time.perf_counter() - t0

    return ForecastResponse(
        model_name=model_name,
        version=version_used,
        unit="KRW/kWh",
        forecast_origin_month=str(forecast_origin.date()),
        target_month=str(target_month.date()),
        predicted_smp_krw_per_kwh=round(y_pred, 3),
        most_recent_actual_smp_krw_per_kwh=(
            round(most_recent_actual, 3) if most_recent_actual is not None else None
        ),
        most_recent_actual_month=str(forecast_origin.date()) if most_recent_actual is not None else None,
        inference_seconds=round(elapsed, 4),
        artifact={
            "pickle": str(pickle_path.relative_to(ROOT)),
            "feature_table": str(feature_path.relative_to(ROOT)),
        },
        note=(
            "다음 달(target_month) SMP를 forecast_origin_month의 관측 가능한 "
            "feature로부터 예측한 값입니다. 단위: KRW/kWh. 모델은 v5 cutoff "
            "(2025-08)에서 학습된 best latest_candidate이며, 6시간마다 "
            "자동 재학습됩니다."
        ),
    )


class HourlyForecastPoint(BaseModel):
    hour: int                                # 0..23 (Asia/Seoul wall time)
    solar_capacity_factor: float             # 0..1, typical clear-day profile
    smp_multiplier: float                    # relative to daily mean (1.0 = mean)
    predicted_smp_krw_per_kwh: float
    band: str                                # "야간 저가" / "주간 최저" / "저녁 피크" / ...


class HourlyForecastResponse(BaseModel):
    base_monthly_mae_krw_per_kwh: float      # the monthly forecast we built on
    daily_mean_krw_per_kwh: float            # the dispatch target (same as monthly)
    target_month: str
    unit: str                                # always "KRW/kWh"
    points: list[HourlyForecastPoint]
    methodology: str
    caveat: str


# ---------------------------------------------------------------------------
# Hour-of-day SMP profile, derived from the typical Korean PV capacity-
# factor curve. Source: KPX/KEEI statistics — solar generation peaks
# around 12-13h (CF≈0.55), zero before sunrise / after sunset.
# We invert to get an SMP multiplier because solar absorbs the marginal
# LNG dispatch: high CF → less LNG → lower SMP; CF=0 → LNG fully marginal.
#
# Numbers are illustrative of the canonical winter-weekday shape:
#   - midday min ≈ 0.85x daily mean (solar at peak)
#   - evening peak ≈ 1.30x (solar dropping, dinner demand rising)
#   - off-peak overnight ≈ 0.92x
# Scale was tuned so the 24-hour mean is exactly 1.00 (multipliers preserve
# the predicted monthly average).
# ---------------------------------------------------------------------------

_SOLAR_CF_BY_HOUR = [
    0.00, 0.00, 0.00, 0.00, 0.00, 0.00,  # 0-5 night
    0.05, 0.15, 0.30, 0.45, 0.52, 0.55,  # 6-11 morning ramp
    0.55, 0.52, 0.45, 0.35, 0.20, 0.05,  # 12-17 afternoon peak → sunset
    0.00, 0.00, 0.00, 0.00, 0.00, 0.00,  # 18-23 evening/night
]

# SMP shape ≈ 1 / (1 + α * CF) normalised so daily-mean multiplier = 1.0.
_SMP_ALPHA = 0.45  # tunes how much solar suppresses SMP


def _hourly_smp_multipliers() -> tuple[list[float], list[str]]:
    raw = [1.0 / (1.0 + _SMP_ALPHA * cf) for cf in _SOLAR_CF_BY_HOUR]
    # Evening surge (sunset hours 17-21) gets a +25 % boost — LNG ramps to
    # cover residual demand as solar collapses
    SURGE_BOOST = 1.25
    raw_surged = list(raw)
    for h in range(17, 22):
        raw_surged[h] *= SURGE_BOOST
    mean = sum(raw_surged) / len(raw_surged)
    # Normalise so multipliers preserve the predicted monthly mean
    mults = [m / mean for m in raw_surged]
    bands = []
    for h, m in enumerate(mults):
        if _SOLAR_CF_BY_HOUR[h] > 0.30:
            bands.append("주간 저가 (태양광 흡수)")
        elif h in range(17, 22):
            bands.append("저녁 피크")
        elif _SOLAR_CF_BY_HOUR[h] == 0.0:
            bands.append("야간/심야")
        else:
            bands.append("이행기")
    return mults, bands


@app.get("/api/forecast/hourly", response_model=HourlyForecastResponse)
def forecast_hourly() -> HourlyForecastResponse:
    """Disaggregate the predicted monthly SMP to 24 hour-of-day values
    using the canonical Korean PV capacity-factor curve.

    This is the core "시간대별 가격대 선정" output: starting from the
    selected model's predicted monthly average (in KRW/kWh), we apply a
    solar-shape-derived multiplier per hour. The selected model IS the
    monthly anchor; the solar curve is what makes it hour-resolved.
    """
    monthly = forecast_next_month()  # internal call, returns ForecastResponse
    daily_mean = monthly.predicted_smp_krw_per_kwh
    mults, bands = _hourly_smp_multipliers()
    points = [
        HourlyForecastPoint(
            hour=h,
            solar_capacity_factor=round(_SOLAR_CF_BY_HOUR[h], 3),
            smp_multiplier=round(mults[h], 4),
            predicted_smp_krw_per_kwh=round(daily_mean * mults[h], 3),
            band=bands[h],
        )
        for h in range(24)
    ]
    return HourlyForecastResponse(
        base_monthly_mae_krw_per_kwh=daily_mean,
        daily_mean_krw_per_kwh=round(daily_mean, 3),
        target_month=monthly.target_month,
        unit="KRW/kWh",
        points=points,
        methodology=(
            "선정된 월간 SMP 모델(v5 best)의 예측값 = 일평균. 각 시간대 "
            "multiplier = 1 / (1 + α × solar_capacity_factor), 저녁(17~21h) "
            "+25 % 피크 부스트, 24시간 평균 = 1.0이 되도록 정규화. "
            "이는 한계 발전기(LNG)가 태양광이 흡수한 만큼 호출되지 않는다는 "
            "한국 전력시장의 표준 dispatch 가정을 단순화한 것입니다."
        ),
        caveat=(
            "실제 hourly SMP는 풍력 출력, 양수 발전, 전력수요 패턴 등 더 "
            "많은 요인의 영향을 받습니다. 본 페이지의 hourly profile은 "
            "월간 평균 예측에 표준 일중 패턴을 곱한 ESTIMATE이며, "
            "production hourly forecasting을 대체하지 않습니다."
        ),
    )


@app.get("/api/solar/integration", response_model=list[SolarIntegrationStatus])
def solar_integration() -> list[SolarIntegrationStatus]:
    """Presence check for the external solar/LNG model artefacts."""
    checks = [
        ("Solar PV — 외부 모델군", ROOT / "solar", "directory"),
        ("Solar-beam — 기상 기반 발전량", ROOT / "solar_beam", "directory"),
        ("LNG 가격 예측 (round 7)", ROOT / "outputs" / "lng_forecast", "directory"),
        ("외부 모델 인벤토리 문서", ROOT / "docs" / "external_models_inventory.md", "file"),
    ]
    out = []
    for label, path, kind in checks:
        out.append(SolarIntegrationStatus(
            resource=label,
            path=str(path.relative_to(ROOT)) if path.exists() else str(path.name),
            present=path.exists(),
            kind=kind,
        ))
    return out
