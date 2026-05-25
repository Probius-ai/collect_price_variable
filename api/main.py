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
