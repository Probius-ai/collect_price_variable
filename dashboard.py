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
import os
import re
from pathlib import Path
from urllib.parse import quote, unquote

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
from src.utils.io import source_root_dir


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


# ---------------------------------------------------------------------------
# Credential redaction (defense-in-depth at display time)
# ---------------------------------------------------------------------------
#
# Raw API envelopes on disk are SUPPOSED to be redacted by each collector
# before persistence (see `redact_service_key_in_url` for KMA, the EIA
# `request_log` masking, etc.). But the dashboard preview is a separate
# trust boundary: a collector added later, an upstream error envelope that
# echoes the key in its message body (data.go.kr SERVICE_KEY responses do
# this), or a stale on-disk file written before the redaction fix would all
# leak straight into the rendered page. We therefore scrub at READ time too.
#
# Patterns covered:
#   * Query/JSON keys: serviceKey, api_key, apiKey, api-key, x-api-key,
#     subscription-key, access_token, accessToken, Bearer/Authorization
#   * Verbatim values pulled from the live process env vars (KMA_PUBLIC_API_KEY,
#     KMA_PUBLIC_API_KEY_ENCODED, EIA_API_KEY) — only if they're set AND non-trivial
#
# Keep this fast: it runs on every preview render.

# Any key/header name whose body word is one of: key, token, secret, auth,
# password/passwd/pwd, credential, cookie, session, sso. The leading
# `[A-Za-z_-]*` lets us catch `apiKey`, `service_key`, `x-api-key`,
# `JSESSIONID`, `aws_access_key_id`, `client_secret`, etc. without
# enumerating every name. Quoting + separator handle JSON, querystring,
# YAML-ish, and HTTP-header forms.
_CREDENTIAL_KEY_RE = re.compile(
    r"""
    (                                       # 1: leading key + separator
        "?[A-Za-z0-9_-]*                    # optional opening quote + prefix chars
        (?:key|token|secret|auth|password|passwd|pwd|credential|cookie|session|sso)
        [A-Za-z0-9_-]*                      # suffix chars
        "?                                  # optional closing quote
        \s*
        (?:[=:]\s*"?)                       # = or :, then optional opening quote
    )
    (                                       # 2: the value to redact
        [^"'&\s,;}\]<]+                     # stop at quote, &, whitespace, , ; } ] <
                                            # (`<` not `>`: an XML close-tag is
                                            # `</element>`, so `<` correctly
                                            # bounds the value, and `<REDACTED>`
                                            # placeholders survive a second pass
                                            # → redaction is idempotent.)
    )
    """,
    re.IGNORECASE | re.VERBOSE,
)

# Authorization header — captures the ENTIRE header value (everything
# after the scheme word), in two forms:
#
#   * JSON-quoted: ``"Authorization":"<scheme> <rest>"`` — value extends to
#     the closing JSON quote (handles ``\"`` escapes inside).
#   * HTTP raw:    ``Authorization: <scheme> <rest>\r\n`` — value extends to
#     end-of-line, but stops at ``"`` so we don't consume the surrounding
#     JSON structure when ``Authorization`` happens to appear inside a
#     string value.
#
# Why capture the whole value instead of just one token: parameterised
# headers (HMAC, AWS SigV4, OAuth1, Digest) pack multiple credential
# parameters into a single header — ``signature=…``, ``response=…``,
# ``oauth_signature=…`` — many of which don't match the generic key
# keyword set. Wholesale masking avoids per-parameter coverage gaps.
#
# Both MUST run before ``_CREDENTIAL_KEY_RE``, otherwise the latter matches
# ``Authorization:``, captures the scheme word as the value, and the actual
# credential after the space slips through unmasked.
_AUTH_HEADER_JSON_RE = re.compile(
    r"""
    (
        ["']Authorization["']\s*:\s*["']   # "Authorization":" or 'Authorization':'
    )
    (
        (?:[^"'\\]|\\["'\\])+              # value (handles \" / \' / \\ escapes)
    )
    (["'])                                 # closing quote
    """,
    re.IGNORECASE | re.VERBOSE,
)
_AUTH_HEADER_HTTP_RE = re.compile(
    r"""
    (
        Authorization
        \s*:\s*
        [A-Za-z0-9_+-]+                    # scheme: Bearer / Basic / HMAC / AWS4-HMAC-SHA256 / OAuth / Digest / Token / ...
        \s+
    )
    (
        [^\r\n]+                           # entire value to end-of-line, INCLUDING internal " characters
                                           # (necessary for OAuth1's quoted parameters; the JSON-form regex
                                           # runs first and consumes well-formed JSON Authorization fields,
                                           # so the only remaining `Authorization:` matches are raw HTTP-style)
    )
    """,
    re.IGNORECASE | re.VERBOSE,
)

# Naked JWT anywhere in the body — three base64url segments joined by `.`.
# Catches tokens echoed in error envelopes without a `Bearer ` prefix.
_JWT_RE = re.compile(
    r"\beyJ[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,}\b"
)

# Credential-shaped env-var names: any var whose name contains one of these
# tokens is treated as sensitive and substring-masked from the preview.
_CRED_ENV_NAME_RE = re.compile(
    r"(key|token|secret|auth|password|passwd|pwd|credential|cookie|session|sso)",
    re.IGNORECASE,
)

# Minimum length before we substring-mask an env value. Below this we risk
# wiping legitimate substrings (e.g. an env var set to "dev").
_ENV_MIN_MASK_LEN = 16


def _redact_preview(text: str) -> str:
    """Mask anything that looks like a credential before display.

    Defense in depth — collectors are supposed to redact at write time,
    but a future collector or an upstream error envelope that echoes a key
    in a free-text field would slip past. This is intentionally aggressive:
    a false positive (masking a value that wasn't sensitive) is cheap
    because the user can still open the file directly; a false negative
    (leaking a real key on the dashboard) is the bug we're defending
    against.
    """
    if not text:
        return text
    # Authorization headers MUST run before the generic key regex (see
    # ordering note on `_AUTH_HEADER_JSON_RE`/`_AUTH_HEADER_HTTP_RE`).
    # JSON form first so the structure-preserving regex masks the value
    # cleanly without the HTTP regex's greedier end-of-line capture
    # destroying the surrounding JSON.
    out = _AUTH_HEADER_JSON_RE.sub(
        lambda m: f"{m.group(1)}<REDACTED>{m.group(3)}", text
    )
    out = _AUTH_HEADER_HTTP_RE.sub(lambda m: f"{m.group(1)}<REDACTED>", out)
    out = _CREDENTIAL_KEY_RE.sub(lambda m: f"{m.group(1)}<REDACTED>", out)
    out = _JWT_RE.sub("<REDACTED>", out)

    # Auto-discover credential-shaped env vars rather than hardcoding a list,
    # so a future collector that adds (e.g.) WEATHER_API_KEY gets masked
    # automatically. Mask BOTH the raw value AND its URL-encoded form, since
    # data.go.kr issues the same key in two forms (raw with `/+=`, encoded
    # with `%2F%2B%3D`) and the on-disk envelope may carry whichever form
    # was actually transmitted.
    for env_name, val in os.environ.items():
        if not val or len(val) < _ENV_MIN_MASK_LEN:
            continue
        if not _CRED_ENV_NAME_RE.search(env_name):
            continue
        out = out.replace(val, "<REDACTED>")
        try:
            encoded = quote(val, safe="")
        except Exception:
            encoded = val
        if encoded != val:
            out = out.replace(encoded, "<REDACTED>")
        try:
            decoded = unquote(val)
        except Exception:
            decoded = val
        if decoded != val and len(decoded) >= _ENV_MIN_MASK_LEN:
            out = out.replace(decoded, "<REDACTED>")
    return out


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
        "0. 모델 선정 (Solar → SMP)",
        "1. Overview",
        "2. Models",
        "3. Predictions",
        "4. Features",
        "5. Data Quality",
        "6. Walk-forward CV",
        "7. LNG before/after",
        "8. LNG forecast (PDF-guided)",
        "9. Round 8-9 collectors",
        "10. MLOps smoke test (v1-v5)",
    ],
)

# ---------------------------------------------------------------------------
# Force-retrain helper (used by Page 0's bottom-right button)
# ---------------------------------------------------------------------------
#
# The smoke-test pipeline takes ~5-10 min end-to-end (LightGBM + rolling
# v5 are the slow parts). We can't block the Streamlit script for that
# long, so we fire-and-forget a subprocess and stash a status marker on
# disk that the page re-reads on every rerun.

RETRAIN_LOCK = ROOT / "outputs" / "_retrain_status.json"


def _retrain_status() -> dict | None:
    if not RETRAIN_LOCK.exists():
        return None
    try:
        return json.loads(RETRAIN_LOCK.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None


def _retrain_pid_alive(pid: int | None) -> bool:
    """True iff the recorded subprocess PID is still in the process table."""
    if not pid:
        return False
    try:
        import os, errno
        os.kill(pid, 0)
        return True
    except (ProcessLookupError, PermissionError):
        return False
    except OSError as e:
        import errno
        return e.errno != errno.ESRCH


class RetrainLaunchError(RuntimeError):
    """Raised when the force-retrain pre-flight check fails — surfaces a
    human-readable reason in the UI instead of a silent subprocess crash."""


def _force_retrain_preflight() -> tuple[Path, Path]:
    """Validate everything the force-retrain subprocess needs BEFORE we
    fork. On failure, raises RetrainLaunchError with a UI-friendly
    message that names the exact missing piece.

    Returns ``(config_path, log_path)`` when all checks pass.
    """
    config_path = ROOT / "config" / "mlops_smoke_test.yaml"
    if not config_path.exists():
        raise RetrainLaunchError(
            f"설정 파일 누락: {config_path.relative_to(ROOT)}. "
            "리포지토리가 부분적으로만 클론된 상태일 수 있습니다."
        )
    # Make sure the smoke-test module itself is importable from the
    # interpreter we're about to launch — otherwise the subprocess
    # would die with a ModuleNotFoundError that only shows up in the log.
    import importlib.util
    if importlib.util.find_spec("src.pipelines.mlops_smoke_test") is None:
        raise RetrainLaunchError(
            "`src.pipelines.mlops_smoke_test` 모듈을 import할 수 없습니다. "
            "현재 dashboard가 실행 중인 Python 환경에 프로젝트가 "
            "설치되지 않았거나 PYTHONPATH가 잘못되었습니다."
        )

    log_path = ROOT / "outputs" / "_retrain.log"
    try:
        log_path.parent.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise RetrainLaunchError(
            f"`outputs/` 디렉토리에 쓸 수 없습니다 ({exc}). 권한 문제일 가능성이 큽니다."
        ) from exc
    # Make sure the RETRAIN_LOCK parent (same `outputs/`) is also good.
    if not log_path.parent.is_dir():
        raise RetrainLaunchError(
            f"`{log_path.parent.relative_to(ROOT)}`가 디렉토리가 아닙니다."
        )
    return config_path, log_path


def _trigger_force_retrain() -> dict:
    """Launch the MLOps smoke test in the background and persist the
    status marker. Returns the new status dict.

    Environment robustness:
      * Uses ``sys.executable`` (the interpreter currently running
        Streamlit) instead of a hardcoded ``.venv/bin/python`` path —
        works in Docker, on fresh clones, and on any platform.
      * Pre-flight (`_force_retrain_preflight`) validates the config
        file + module import + writable `outputs/` BEFORE we fork, so
        failures surface as a clear UI message instead of a silent
        subprocess crash logged only to disk.
      * Closes the parent's reference to the log file handle right
        after Popen — the subprocess inherits the fd, but the parent
        (Streamlit's process) doesn't leak one per click.
      * Detaches the subprocess so it survives the dashboard process
        restarting. Uses `start_new_session=True` on POSIX (Linux /
        macOS / WSL), no-op on Windows (`creationflags` would be the
        Windows equivalent but isn't needed for this project's WSL
        target).
    """
    import os, subprocess, sys
    from datetime import datetime, timezone

    config_path, log_path = _force_retrain_preflight()

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
        sys.executable,  # whichever Python is running Streamlit right now
        "-m", "src.pipelines.mlops_smoke_test",
        "--config", str(config_path),
        "--log-to-mlflow",
    ]
    popen_kwargs: dict = {
        "stderr": subprocess.STDOUT,
        "cwd": str(ROOT),
        "env": env,
    }
    # Detach the subprocess so it survives a Streamlit restart. POSIX-only;
    # on Windows we'd use creationflags=CREATE_NEW_PROCESS_GROUP, but this
    # project's deployment target is WSL/Linux.
    if os.name == "posix":
        popen_kwargs["start_new_session"] = True

    # Open the log file as a context-managed handle so the PARENT's fd
    # is closed right after Popen completes. The subprocess gets its own
    # inherited fd via the OS, so it can keep writing without us holding
    # a reference. Without this, every button click leaks one fd in the
    # Streamlit process.
    with log_path.open("w", encoding="utf-8") as log_f:
        proc = subprocess.Popen(cmd, stdout=log_f, **popen_kwargs)

    status = {
        "pid": proc.pid,
        "started_at": datetime.now(timezone.utc).isoformat(),
        "trigger": "force_retrain_button",
        "log_path": str(log_path.relative_to(ROOT)),
        "mlflow_uri": env["MLFLOW_TRACKING_URI"],
        "python_executable": sys.executable,
    }
    RETRAIN_LOCK.parent.mkdir(parents=True, exist_ok=True)
    RETRAIN_LOCK.write_text(json.dumps(status, indent=2), encoding="utf-8")
    return status


# ---------------------------------------------------------------------------
# Page 0: 모델 선정 (Solar → SMP) — landing / decision page
# ---------------------------------------------------------------------------

if page == "0. 모델 선정 (Solar → SMP)":
    st.header("⚡ 모델 선정 — 단기 태양광 발전량 예측을 통한 전력 가격 모델")
    st.caption(
        "이 페이지는 v1~v5 staged-retraining 결과를 한 화면으로 모아서 "
        "**production에 올릴 모델 1개를 고르는 의사결정**을 돕는 뷰입니다. "
        "각 섹션은 깊은 분석이 가능한 사이드바의 다른 페이지로 연결됩니다."
    )

    COMPARISON_CSV = ROOT / "outputs/metrics/mlops_version_comparison.csv"
    REGISTRY_DIR = ROOT / "outputs/model_registry"

    if not COMPARISON_CSV.exists():
        st.warning(
            "비교 데이터가 없습니다. 사이드바의 우하단 **강제 재학습** 버튼을 "
            "누르거나, 터미널에서 "
            "`python -m src.pipelines.mlops_smoke_test "
            "--config config/mlops_smoke_test.yaml --log-to-mlflow` 를 "
            "실행해 주세요."
        )
    else:
        cmp = pd.read_csv(COMPARISON_CSV)
        cmp_clean = cmp[cmp["skipped"] != True].copy()  # noqa: E712

        # -- Decision panel ---------------------------------------------
        rec_row = cmp_clean[cmp_clean["registry_status"] == "recommended_historical"]
        latest_rows = cmp_clean[cmp_clean["registry_status"] == "latest_candidate"]
        # Best latest_candidate = lowest MAE among v5 entries
        best_latest = (
            latest_rows.sort_values("mae").iloc[0]
            if not latest_rows.empty else None
        )
        persistence_v5 = latest_rows[latest_rows["model"] == "persistence_monthly"]
        baseline_mae = (
            float(persistence_v5["mae"].iloc[0])
            if not persistence_v5.empty else None
        )

        st.subheader("의사결정 패널")
        st.caption(
            "MAE 기준. 추천 모델은 (1) v1~v4 holdout 평균 우수 + (2) v5 rolling "
            "검증에서도 안정적인 것을 우선합니다."
        )

        c1, c2, c3, c4 = st.columns(4)
        if not rec_row.empty:
            r = rec_row.iloc[0]
            c1.metric(
                "추천 모델 (v1~v4 best)",
                str(r["model"]),
                help=f"version={r['version']}, evaluation_mode={r['evaluation_mode']}",
            )
            c2.metric("Test MAE", f"{float(r['mae']):.3f}")
        if best_latest is not None:
            c3.metric(
                "v5 latest candidate",
                str(best_latest["model"]),
                f"{float(best_latest['mae']):.3f} MAE",
                help="2025-08 cutoff, 12-month rolling validation",
            )
        if baseline_mae is not None and best_latest is not None:
            improvement = baseline_mae - float(best_latest["mae"])
            c4.metric(
                "vs Persistence (v5)",
                f"{improvement:+.3f} MAE",
                f"{(improvement / baseline_mae * 100):+.1f} %",
                help=f"Persistence baseline MAE at v5 = {baseline_mae:.3f}",
            )

        # -- Stability + ranking table ----------------------------------
        st.subheader("후보 모델 안정성 (v4 → v5 변화)")
        st.caption(
            "v4는 fixed-holdout (forward labels 있음), v5는 rolling validation "
            "(future labels 없음). 두 점수 차이가 작을수록 **새 데이터가 들어와도 "
            "성능이 흔들리지 않는** 모델."
        )
        v4 = cmp_clean[cmp_clean["version"] == "v4"].set_index("model")[["mae"]]
        v5 = cmp_clean[cmp_clean["version"] == "v5"].set_index("model")[["mae"]]
        v4.columns = ["v4_holdout_mae"]
        v5.columns = ["v5_rolling_mae"]
        stab = v4.join(v5, how="inner").reset_index()
        stab["delta_v5_minus_v4"] = (stab["v5_rolling_mae"] - stab["v4_holdout_mae"]).round(3)
        stab["mean_v4_v5_mae"] = ((stab["v4_holdout_mae"] + stab["v5_rolling_mae"]) / 2).round(3)
        stab = stab.sort_values("mean_v4_v5_mae").reset_index(drop=True)

        def _highlight_row(row):
            colour = ""
            if row["model"] == (rec_row.iloc[0]["model"] if not rec_row.empty else None):
                colour = "background-color: #d4edda;"
            elif best_latest is not None and row["model"] == best_latest["model"]:
                colour = "background-color: #d1ecf1;"
            return [colour] * len(row)

        st.dataframe(
            stab.round(3).style.apply(_highlight_row, axis=1),
            width="stretch", hide_index=True,
        )
        st.caption(
            "🟢 = recommended_historical (v1~v4 best) · "
            "🔵 = v5 latest_candidate (best). 두 모델이 일치하면 그 모델이 "
            "가장 강한 후보입니다."
        )

        # -- Why Solar matters ------------------------------------------
        with st.expander(
            "🌞  왜 단기 태양광 예측이 SMP 모델 선정에 중요한가",
            expanded=False,
        ):
            st.markdown("""
한국 전력시장의 SMP(System Marginal Price)는 마지막에 호출된
**한계 발전기의 변동비**가 결정합니다. 일반적으로 한계 발전기는 LNG 화력이고,
LNG는 가격 변동이 큰 연료입니다. 따라서:

```
↑ 단기 태양광 발전량 예측
        ↓
↓ LNG 화력 호출량 (재생에너지가 우선 송전)
        ↓
↓ LNG 수요·도입가 충격 가능성
        ↓
↓ SMP (특히 일간/시간대별)
```

**모델 선정 관점에서**:
- 단기 (수 시간) 태양광 예측이 정확할수록 LNG 콜이 줄어드는 시간대를 미리 알 수 있고,
  그만큼 SMP 예측의 분산이 줄어듭니다.
- 본 워크스페이스는 월간 SMP 모델을 후보로 갖고 있고, 태양광은 외부
  `solar/` 모델 (별도 PV 모델군)이 처리합니다 — Round 6/7에서 **LNG 가격
  예측을 SMP feature로 통합**한 결과(`docs/figures/baseline_post_lng_v2_leakfix/`)를
  비교한 것이 그 첫걸음입니다.
- 따라서 추천 기준은 단순 MAE 최저가 아니라
  **(a) 안정성 (v5 rolling 검증 강건) + (b) 외부 LNG·날씨 신호에 반응하는 구조**
  입니다.
            """)

        # -- External solar/LNG model inventory -------------------------
        st.subheader("외부 모델 인벤토리 (Solar / LNG)")
        ext = []
        for label, path, kind in [
            ("Solar PV — 외부 모델군", ROOT / "solar", "directory"),
            ("Solar-beam — 기상 기반 발전량", ROOT / "solar_beam", "directory"),
            ("LNG 가격 예측 (round 7)", ROOT / "outputs/lng_forecast", "directory"),
            ("외부 모델 인벤토리 문서", ROOT / "docs/external_models_inventory.md", "file"),
        ]:
            present = path.exists()
            ext.append({
                "리소스": label,
                "경로": str(path.relative_to(ROOT)) if present else "—",
                "상태": "✅ 있음" if present else "❌ 없음",
                "유형": kind,
            })
        st.dataframe(pd.DataFrame(ext), width="stretch", hide_index=True)

        # -- Deep-dive jump links ---------------------------------------
        st.subheader("더 깊이 들어가려면")
        c1, c2, c3 = st.columns(3)
        c1.info(
            "**모델 성능 표** — Page 2 Models\n\n"
            "comparison.csv 기반 MAE/RMSE/MAPE/R²/directional_accuracy 전체."
        )
        c2.info(
            "**예측 vs 실측 시계열** — Page 3 Predictions\n\n"
            "선택한 모델의 forecast_origin / target_month 호버 포함."
        )
        c3.info(
            "**Walk-forward CV** — Page 6\n\n"
            "test split이 아닌 매월 refit하며 본 production-like 성능."
        )
        c1, c2, c3 = st.columns(3)
        c1.info(
            "**LNG 통합 효과** — Page 7\n\n"
            "Round 6 leakage fix 전·후, Round 8 JKM 추가 3-way 비교."
        )
        c2.info(
            "**LNG 단독 예측 모델** — Page 8\n\n"
            "PDF 가이드 기반 LightGBM/MLP/naive 비교."
        )
        c3.info(
            "**MLOps v1~v5 상세** — Page 10\n\n"
            "각 cutoff별 모델 등록 + JSON registry."
        )

        # -- Bottom row: force-retrain button + 6-hour auto-retrain note
        st.markdown("---")
        st.markdown("")  # spacer

        note_col, btn_col = st.columns([3, 1])
        with note_col:
            st.caption(
                "ℹ️ **모델은 6시간마다 단기 예보 정보를 통해 자동 재학습됩니다.** "
                "수동으로 즉시 새 학습을 돌리려면 오른쪽 버튼을 누르세요. "
                "백그라운드에서 v1~v5 smoke test가 실행되며 MLflow에 자동 기록됩니다 "
                "(보통 5~10분 소요)."
            )
        with btn_col:
            if st.button("🔄 강제 재학습", type="primary", width="stretch"):
                current = _retrain_status()
                if current and _retrain_pid_alive(current.get("pid")):
                    st.warning(
                        f"이미 재학습이 진행 중입니다 (PID {current['pid']}, "
                        f"시작 {current['started_at'][:19]})."
                    )
                else:
                    try:
                        new_status = _trigger_force_retrain()
                        st.success(
                            f"재학습 시작 (PID {new_status['pid']}). "
                            "페이지를 새로고침하면 진행 상태가 업데이트됩니다."
                        )
                    except RetrainLaunchError as e:
                        st.error(f"재학습 시작 실패: {e}")

        # Status display (always visible if a marker exists)
        status = _retrain_status()
        if status:
            alive = _retrain_pid_alive(status.get("pid"))
            ts = status.get("started_at", "?")[:19]
            if alive:
                st.info(
                    f"🟡 재학습 진행 중 — PID {status['pid']}, "
                    f"started {ts} (UTC). "
                    f"진행 로그: `{status.get('log_path', '-')}`. "
                    f"완료 시 MLflow ({status.get('mlflow_uri', '-')}) 의 "
                    "`kpx-smp-monthly` experiment에 새 run 50개가 추가됩니다."
                )
            else:
                st.success(
                    f"🟢 최근 재학습 완료 — started {ts} (UTC). "
                    "Page 10 또는 MLflow UI에서 새 결과를 확인하세요."
                )


# ---------------------------------------------------------------------------
# Page 1: Overview
# ---------------------------------------------------------------------------

elif page == "1. Overview":
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
# Page 7: LNG before/after comparison
# ---------------------------------------------------------------------------

elif page == "7. LNG before/after":
    st.header("LNG integration: before vs after")
    st.caption(
        "Compares three snapshots: pre-LNG (baseline_20260525), LNG with leakage "
        "(baseline_post_lng_v1 — REVIEWED OUT), and LNG with publish-timing "
        "discipline (baseline_post_lng_v2_leakfix — current honest result)."
    )

    PRE = Path("docs/figures/baseline_20260525")
    LEAKY = Path("docs/figures/baseline_post_lng_v1")
    SAFE = Path("docs/figures/baseline_post_lng_v2_leakfix")

    # Numeric table
    def _test_mae(snap_dir):
        p = snap_dir / "metrics_comparison_snapshot.csv"
        if not p.exists():
            return None
        m = pd.read_csv(p)
        return m[m["split"] == "test"].set_index("model")["mae"]

    pre_t = _test_mae(PRE)
    leak_t = _test_mae(LEAKY)
    safe_t = _test_mae(SAFE)
    if pre_t is not None and safe_t is not None:
        common = pre_t.index.intersection(safe_t.index)
        df = pd.DataFrame({"pre-LNG": pre_t.loc[common], "post-LNG (leak-fixed)": safe_t.loc[common]})
        if leak_t is not None:
            df.insert(1, "LNG (leaky, rejected)", leak_t.loc[common])
        df["Δ vs pre"] = (df["post-LNG (leak-fixed)"] - df["pre-LNG"]).round(3)
        df["Δ %"] = (df["Δ vs pre"] / df["pre-LNG"] * 100).round(2)
        df = df.sort_values("post-LNG (leak-fixed)").round(3)
        df["kind"] = df.index.map(classify)
        st.subheader("Test split MAE — three-way comparison")
        st.dataframe(df, width="stretch")

    # Walk-forward
    def _wf_mae(snap_dir):
        p = snap_dir / "walk_forward_comparison_snapshot.csv"
        if not p.exists():
            return None
        return pd.read_csv(p).set_index("model")["mae"]

    pre_w = _wf_mae(PRE)
    safe_w = _wf_mae(SAFE)
    if pre_w is not None and safe_w is not None:
        common = pre_w.index.intersection(safe_w.index)
        dfw = pd.DataFrame({"pre-LNG": pre_w.loc[common], "post-LNG": safe_w.loc[common]})
        dfw["Δ"] = (dfw["post-LNG"] - dfw["pre-LNG"]).round(3)
        dfw["Δ %"] = (dfw["Δ"] / dfw["pre-LNG"] * 100).round(2)
        dfw = dfw.sort_values("post-LNG").round(3)
        st.subheader("Walk-forward MAE — first time trainable beats persistence")
        st.dataframe(dfw, width="stretch")

    # Static PNGs from save_baseline_plots --docs-copy
    st.subheader("3-way comparison plots (PNG snapshots)")
    plot_files = [
        ("plot_07_test_mae_3way_pre_leaky_safe.png",
         "Test split — pre vs leaky vs leak-fixed"),
        ("plot_08_walkforward_mae_3way_pre_leaky_safe.png",
         "Walk-forward CV — leak-fixed trainable beats persistence floor"),
    ]
    for fname, caption in plot_files:
        fp = SAFE / fname
        if fp.exists():
            st.image(str(fp), caption=caption, use_container_width=True)
        else:
            st.warning(f"Missing: {fp}")

    with st.expander("Per-snapshot model overlays (Predictions plot from each tag)"):
        for label, snap_dir in [
            ("Pre-LNG", PRE),
            ("LNG leaky (rejected)", LEAKY),
            ("LNG leak-fixed (current)", SAFE),
        ]:
            p = snap_dir / "plot_01_predictions_test_all_models.png"
            st.markdown(f"**{label}** — `{snap_dir.name}`")
            if p.exists():
                st.image(str(p), use_container_width=True)
            else:
                st.warning(f"Missing: {p}")

    with st.expander("Persistence-lag zoom (delta plot for the LNG fix)"):
        for label, snap_dir in [
            ("Pre-LNG", PRE),
            ("LNG leak-fixed", SAFE),
        ]:
            p = snap_dir / "plot_02_persistence_lag_zoom.png"
            st.markdown(f"**{label}** — `{snap_dir.name}`")
            if p.exists():
                st.image(str(p), use_container_width=True)

    # Round-8 snapshot — adds JKM/EIA/KMA paths
    R8 = Path("docs/figures/baseline_round8_jkm_eia_kma")
    if R8.exists():
        st.subheader("Round-8 snapshot — JKM forward + EIA STEO + KMA wired in")
        for fname, caption in [
            ("plot_04_test_mae_r2_comparison.png", "Test MAE & R² (round-8 snapshot)"),
            ("plot_05_walk_forward_long_range.png", "Walk-forward long range"),
            ("plot_06_feature_group_ablation_heatmap.png",
             "Feature-group ablation (round-8 snapshot)"),
        ]:
            fp = R8 / fname
            if fp.exists():
                st.image(str(fp), caption=caption, use_container_width=True)


# ---------------------------------------------------------------------------
# Page 8: LNG forecast (PDF-guided, round 7)
# ---------------------------------------------------------------------------

elif page == "8. LNG forecast (PDF-guided)":
    st.header("LNG forecast — PDF-guided model (round 7)")
    st.caption(
        "Standalone LNG price forecaster built per the two PDF guides "
        "(에너지가격예측 파이프라인 / energy-price-forecast-pipeline-guide). "
        "TimeSeriesSplit + embargo, LightGBM with inner-validation early stopping "
        "(round-7 leakage fix), MLP with per-fold scaler isolation, naive_lag1 "
        "baseline. Metrics are mean ± std across folds."
    )

    LNG_ROOT = OUTPUTS / "lng_forecast"
    if not LNG_ROOT.exists():
        st.warning(
            "No `outputs/lng_forecast/` yet. Run "
            "`python -m src.pipelines.train_lng_forecast --horizon-months 1 "
            "--n-splits 5 --embargo 1`."
        )
    else:
        runs = sorted(p.name for p in LNG_ROOT.iterdir() if p.is_dir())
        if not runs:
            st.info("No runs under `outputs/lng_forecast/`.")
        else:
            run = st.selectbox("Run", runs, index=len(runs) - 1)
            run_dir = LNG_ROOT / run

            # Panel info — sample size, feature inventory
            panel_path = run_dir / "panel_info.json"
            if panel_path.exists():
                panel = json.loads(panel_path.read_text())
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Rows (post NaN-drop)", panel.get("rows", "—"))
                c2.metric("Features", panel.get("n_features", "—"))
                c3.metric("Horizon (months)", panel.get("horizon_months", "—"))
                c4.metric(
                    "Period",
                    f"{str(panel.get('period_min', '—'))[:7]} → "
                    f"{str(panel.get('period_max', '—'))[:7]}",
                )

            # Cross-model comparison
            comp_csv = run_dir / "comparison.csv"
            if comp_csv.exists():
                st.subheader("Cross-validation comparison")
                cmp = pd.read_csv(comp_csv).round(3)
                st.dataframe(cmp, width="stretch", hide_index=True)
                fig = px.bar(
                    cmp.sort_values("mae_mean"),
                    x="model", y="mae_mean", error_y="mae_std",
                    text="mae_mean", color="model",
                    title="LNG forecast MAE (mean ± std across folds)",
                )
                fig.update_traces(texttemplate="%{text:.2f}", textposition="outside")
                fig.update_layout(showlegend=False)
                st.plotly_chart(fig, width="stretch")

                fig2 = px.bar(
                    cmp.sort_values("rmse_mean"),
                    x="model", y="rmse_mean", error_y="rmse_std",
                    text="rmse_mean", color="model",
                    title="LNG forecast RMSE (mean ± std across folds)",
                )
                fig2.update_traces(texttemplate="%{text:.2f}", textposition="outside")
                fig2.update_layout(showlegend=False)
                st.plotly_chart(fig2, width="stretch")

                st.info(
                    "**Finding:** the `naive_lag1` baseline wins on this CV slice. "
                    "Trainable models (LightGBM, MLP) do not yet beat naive lag-1 on "
                    "monthly JKM levels — consistent with the PDF guide's note that "
                    "monthly-frequency LNG is dominated by persistence."
                )

            # Per-model predictions
            st.subheader("Per-model fold predictions")
            model_dirs = sorted(
                p.name for p in run_dir.iterdir()
                if p.is_dir() and (p / "predictions.csv").exists()
            )
            sel = st.multiselect(
                "Models to overlay", model_dirs,
                default=model_dirs[: min(3, len(model_dirs))],
            )
            if sel:
                fig = go.Figure()
                actual_drawn = False
                for m in sel:
                    df = pd.read_csv(run_dir / m / "predictions.csv")
                    if "period_month" in df.columns:
                        df["period_month"] = pd.to_datetime(df["period_month"])
                        df = df.sort_values("period_month")
                    if not actual_drawn and "y_true" in df.columns:
                        fig.add_trace(go.Scatter(
                            x=df["period_month"], y=df["y_true"],
                            name="Actual JKM (USD/MMBtu)", mode="lines+markers",
                            line=dict(color="black", width=2.5),
                        ))
                        actual_drawn = True
                    if "y_pred" in df.columns:
                        fig.add_trace(go.Scatter(
                            x=df["period_month"], y=df["y_pred"],
                            name=f"{m} (pred)", mode="lines+markers",
                            line=dict(dash="dot"),
                        ))
                fig.update_layout(
                    title="LNG forecast: actual vs predicted (out-of-fold)",
                    xaxis_title="period_month", yaxis_title="JKM [USD/MMBtu]",
                    hovermode="x unified",
                )
                st.plotly_chart(fig, width="stretch")

            # Feature inventory expander
            if panel_path.exists():
                with st.expander("Feature columns used (panel_info.json)"):
                    st.json(panel, expanded=False)


# ---------------------------------------------------------------------------
# Page 9: Round 8-9 collectors (EIA STEO + JKM + KMA + raw weather DB)
# ---------------------------------------------------------------------------

elif page == "9. Round 8-9 collectors":
    st.header("Round 8-9 collectors")
    st.caption(
        "API-driven sources added in the last two rounds: EIA STEO "
        "(world oil/gas outlook), JKM daily history + forward curve, KMA "
        "mid-term temperature + village (단기예보), plus a one-shot "
        "raw_weather monthly aggregation pulled from solar_beam's DB."
    )

    DATA_RAW = ROOT / "data" / "raw"

    @st.cache_data(show_spinner=False)
    def _scan_collector_root(root_str: str) -> dict:
        """Roll up response_*.json + parsed_*.parquet under an absolute root.

        Cached on the stringified path because Streamlit's cache hashes the
        argument; `Path` is hashable but we keep the API string-typed so the
        cache key is stable across imports.
        """
        root = Path(root_str)
        if not root.exists():
            return {"present": False}
        raws = sorted(root.rglob("response_*.json"))
        parsed = sorted(root.rglob("parsed_*.parquet"))
        latest_raw = raws[-1] if raws else None
        latest_rows = 0
        if parsed:
            try:
                latest_rows = len(pd.read_parquet(parsed[-1]))
            except Exception:
                latest_rows = 0
        return {
            "present": True,
            "n_raw": len(raws),
            "n_parsed": len(parsed),
            "latest_raw": str(latest_raw.relative_to(ROOT)) if latest_raw else None,
            "latest_parsed_rows": latest_rows,
            "latest_parsed_path": (
                str(parsed[-1].relative_to(ROOT)) if parsed else None
            ),
        }

    # source_id → human label. Path is derived via `source_root_dir(source_id)`
    # which is the same helper `BaseFileLoader.load_one()` uses to write
    # `parsed_*.parquet` snapshots — so inventory counts always match what's
    # actually on disk, regardless of how the source-root scheme evolves.
    collectors = {
        "eia_steo":                       "EIA STEO (BREPUUS / WTIPUUS / NGHHMCF)",
        "kma_village_fcst":               "KMA village forecast (단기예보)",
        "kma_mid_temperature":            "KMA mid-term temperature",
        "jkm_lng_daily_history_file":     "JKM LNG daily history (file)",
        "jkm_lng_futures_curve_file":     "JKM LNG futures curve (file)",
        "raw_weather_monthly_db_file":    "Raw weather monthly (from solar_beam DB)",
        "kma_grid_xy_mapping_file":       "KMA grid x/y mapping (xlsx)",
        "oil_price_monthly_file":         "Oil price monthly (file)",
        "fx_usd_krw_monthly_file":        "FX USD/KRW monthly (file)",
        "lng_price_monthly_file":         "LNG price monthly (file)",
    }
    rows = []
    for source_id, label in collectors.items():
        out_root = source_root_dir(source_id)
        info = _scan_collector_root(str(out_root))
        try:
            rel = str(out_root.relative_to(ROOT))
        except ValueError:
            rel = str(out_root)
        rows.append({
            "source": label,
            "path": rel,
            "raw payloads": info.get("n_raw", 0) if info["present"] else "—",
            "parsed snapshots": info.get("n_parsed", 0) if info["present"] else "—",
            "latest rows": info.get("latest_parsed_rows", 0) if info["present"] else "—",
            "present": "✓" if info["present"] else "—",
        })
    st.subheader("Collector inventory")
    st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)

    # Recent raw envelopes — verbatim audit trail
    st.subheader("Recent KMA / EIA raw envelopes (verbatim audit trail)")
    st.caption(
        "Round-9 fix: data.go.kr error responses are persisted verbatim "
        "BEFORE the collector raises, with microsecond + sha256 disambiguator "
        "so same-second retries don't overwrite each other. Each line below "
        "is a saved `response_*.json` with the server's untouched body and "
        "`serviceKey` redacted to `<REDACTED>`."
    )
    envelope_paths: list[Path] = []
    for sub in ("kma/village_fcst", "kma/mid_temperature", "eia/steo"):
        root = DATA_RAW / sub
        if root.exists():
            envelope_paths.extend(sorted(root.rglob("response_*.json"))[-3:])
    if not envelope_paths:
        st.info("No raw envelopes on disk yet.")
    else:
        env_rows = []
        for p in envelope_paths:
            try:
                stat = p.stat()
                size_kb = round(stat.st_size / 1024, 1)
            except OSError:
                size_kb = None
            env_rows.append({
                "file": str(p.relative_to(ROOT)),
                "size (KB)": size_kb,
            })
        st.dataframe(pd.DataFrame(env_rows), width="stretch", hide_index=True)

        sel_env = st.selectbox(
            "Inspect an envelope (first 4 KB)",
            [r["file"] for r in env_rows],
            index=len(env_rows) - 1,
        )
        st.caption(
            "Credentials (`serviceKey`, `api_key`, `Authorization: Bearer …`, "
            "and verbatim env-var values) are scrubbed at display time as "
            "defense-in-depth — collector-level redaction is the primary line "
            "of defense, this preview is the second."
        )
        try:
            head = (ROOT / sel_env).read_text(encoding="utf-8")[:4096]
            st.code(_redact_preview(head), language="json")
        except OSError as exc:
            st.error(f"Could not read {sel_env}: {exc}")

    # Raw weather monthly preview (extracted from solar_beam DB)
    st.subheader("Raw weather monthly (extracted from solar_beam DB)")
    wp = DATA_RAW / "manual_or_filedata" / "raw_weather_monthly_db_file" / "weather_monthly_from_db.csv"
    if wp.exists():
        wdf = pd.read_csv(wp)
        if "period_month" in wdf.columns:
            wdf["period_month"] = pd.to_datetime(wdf["period_month"])
        # CSV column is `region_name` (Korean: 강원 / 경기 / …). The earlier
        # `region` reference here matched nothing, so region count + per-region
        # plot were silently empty.
        REGION_COL = "region_name"
        c1, c2, c3 = st.columns(3)
        c1.metric("Monthly rows", len(wdf))
        if REGION_COL in wdf.columns:
            c2.metric("Regions", wdf[REGION_COL].nunique())
        if "period_month" in wdf.columns and len(wdf):
            c3.metric(
                "Period",
                f"{wdf['period_month'].min():%Y-%m} → {wdf['period_month'].max():%Y-%m}",
            )

        numeric_cols = [c for c in wdf.columns if c not in {"period_month", REGION_COL}
                        and pd.api.types.is_numeric_dtype(wdf[c])]
        if numeric_cols and "period_month" in wdf.columns:
            metric = st.selectbox("Metric to plot", numeric_cols, index=0)
            region_opts = (
                sorted(wdf[REGION_COL].dropna().unique())
                if REGION_COL in wdf.columns else []
            )
            chosen = st.multiselect(
                "Regions", region_opts,
                default=region_opts[: min(3, len(region_opts))],
            ) if region_opts else None
            plot_df = wdf if not chosen else wdf[wdf[REGION_COL].isin(chosen)]
            fig = px.line(
                plot_df.sort_values([REGION_COL, "period_month"])
                if REGION_COL in plot_df.columns
                else plot_df.sort_values("period_month"),
                x="period_month", y=metric,
                color=REGION_COL if REGION_COL in plot_df.columns else None,
                title=f"Monthly weather — {metric}",
            )
            st.plotly_chart(fig, width="stretch")
        with st.expander("Raw weather table (first 100 rows)"):
            st.dataframe(wdf.head(100), width="stretch")
    else:
        st.info(
            "weather_monthly_from_db.csv not found — "
            "run the solar_beam → monthly aggregation script."
        )


# ---------------------------------------------------------------------------
# Page 10: MLOps smoke test (v1..v5)
# ---------------------------------------------------------------------------

elif page == "10. MLOps smoke test (v1-v5)":
    st.header("MLOps smoke test — v1..v5 staged retraining")
    st.caption(
        "Read-only view of the latest run of "
        "`python -m src.pipelines.mlops_smoke_test`. Simulates retraining "
        "as new data accumulates by truncating the feature panel to "
        "successively-later `data_cutoff_month` values. v1..v4 score "
        "against a forward holdout; v5 has no future labels and uses "
        "rolling validation only — promotion to production is intentionally "
        "left to a human operator."
    )

    MLOPS_REPORT_JSON = ROOT / "outputs/reports/mlops_smoke_test_v1_v5_report.json"
    MLOPS_REPORT_MD = ROOT / "outputs/reports/mlops_smoke_test_v1_v5_report.md"
    MLOPS_COMPARISON_CSV = ROOT / "outputs/metrics/mlops_version_comparison.csv"
    MLOPS_REGISTRY_DIR = ROOT / "outputs/model_registry"

    if not MLOPS_COMPARISON_CSV.exists():
        st.warning(
            "No MLOps smoke-test results yet. Run "
            "`python -m src.pipelines.mlops_smoke_test "
            "--config config/mlops_smoke_test.yaml` (add `--log-to-mlflow` "
            "after `docker compose up -d` to also log to MLflow)."
        )
    else:
        df = pd.read_csv(MLOPS_COMPARISON_CSV)
        if "data_cutoff_month" in df.columns:
            df["data_cutoff_month"] = pd.to_datetime(
                df["data_cutoff_month"], errors="coerce"
            )

        # Topline metrics
        n_versions = df["version"].nunique() if "version" in df.columns else 0
        n_runs = len(df)
        n_skipped = int(df["skipped"].sum()) if "skipped" in df.columns else 0
        n_mlflow = (
            int(df["mlflow_run_id"].notna().sum())
            if "mlflow_run_id" in df.columns else 0
        )
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Versions", n_versions)
        c2.metric("Total (version, model) runs", n_runs)
        c3.metric("Controlled-skips", n_skipped)
        c4.metric("MLflow-logged runs", n_mlflow)

        # ---- Version comparison table ------------------------------------
        st.subheader("Comparison table — v1..v5 × models")
        st.caption(
            "`registry_status`: `historical_backtest` (v1..v4 fixed holdout), "
            "`latest_candidate` (v5 rolling validation only), "
            "`recommended_historical` (best v1..v4 by primary metric), "
            "`skipped` (controlled-skip — see `skip_reason`). "
            "No row is ever auto-promoted to production."
        )
        show_cols = [
            c for c in [
                "version", "model", "data_cutoff_month", "evaluation_mode",
                "n_train", "n_test", "mae", "rmse", "mape", "r2",
                "improvement_vs_persistence", "registry_status",
                "mlflow_run_id", "skip_reason",
            ] if c in df.columns
        ]
        st.dataframe(
            df[show_cols].round(3),
            width="stretch", hide_index=True,
        )

        # ---- Metric trend per version ------------------------------------
        st.subheader("Metric trend across versions")
        numeric_metrics = [
            c for c in ["mae", "rmse", "mape", "r2", "directional_accuracy"]
            if c in df.columns
        ]
        if numeric_metrics:
            metric = st.selectbox(
                "Metric", numeric_metrics, index=0,
                help="Plot this metric per model across v1..v5."
            )
            # Mark v5 visually so the "no future holdout" caveat is obvious
            chart_df = df[["version", "model", metric, "evaluation_mode"]].copy()
            chart_df = chart_df.dropna(subset=[metric])
            fig = px.line(
                chart_df.sort_values(["model", "version"]),
                x="version", y=metric, color="model",
                line_dash="evaluation_mode",
                markers=True,
                title=f"{metric} by version (dashed = rolling-validation only)",
            )
            st.plotly_chart(fig, width="stretch")

        # ---- Registry status snapshot ------------------------------------
        st.subheader("Model registry status")
        if "registry_status" in df.columns:
            status_counts = (
                df.groupby(["version", "registry_status"])
                .size().reset_index(name="count")
            )
            fig = px.bar(
                status_counts,
                x="version", y="count", color="registry_status",
                barmode="stack",
                title="Promotion status per version",
                category_orders={
                    "registry_status": [
                        "recommended_historical",
                        "historical_backtest",
                        "latest_candidate",
                        "skipped",
                    ],
                },
            )
            st.plotly_chart(fig, width="stretch")

            # Surface the recommendation explicitly
            rec = df[df["registry_status"] == "recommended_historical"]
            if not rec.empty:
                r = rec.iloc[0]
                st.success(
                    f"**Recommended historical model**: `{r['model']}` "
                    f"from `{r['version']}` (cutoff "
                    f"{pd.Timestamp(r['data_cutoff_month']):%Y-%m}) — "
                    f"MAE {r['mae']:.3f}. Promotion to production is manual."
                )
            latest = df[df["registry_status"] == "latest_candidate"]
            if not latest.empty:
                st.info(
                    f"**Latest candidate**: {len(latest)} model(s) trained at "
                    f"cutoff `{pd.Timestamp(latest.iloc[0]['data_cutoff_month']):%Y-%m}`. "
                    "v5 has **no future labels** — scores come from rolling "
                    "validation against the last 12 months of its own training "
                    "window, NOT a forward holdout."
                )

        # ---- Per-model registry history (JSON fallback) ------------------
        if MLOPS_REGISTRY_DIR.exists():
            registry_files = sorted(MLOPS_REGISTRY_DIR.glob("*_registry.json"))
            if registry_files:
                st.subheader("JSON registry (per-model history)")
                st.caption(
                    "The smoke test always writes a JSON registry under "
                    "`outputs/model_registry/<model>_registry.json` in "
                    "addition to MLflow, so version history survives even "
                    "if the MLflow backend is down."
                )
                sel_model = st.selectbox(
                    "Model registry to inspect",
                    [p.stem.removesuffix("_registry") for p in registry_files],
                )
                sel_path = MLOPS_REGISTRY_DIR / f"{sel_model}_registry.json"
                if sel_path.exists():
                    records = json.loads(sel_path.read_text(encoding="utf-8")).get(
                        "records", [])
                    if records:
                        rdf = pd.DataFrame([
                            {
                                "version": r["version"],
                                "cutoff": r["data_cutoff_month"],
                                "mode": r["evaluation_mode"],
                                "promotion": r["promotion_status"],
                                "mae": (r.get("metrics") or {}).get("mae"),
                                "rmse": (r.get("metrics") or {}).get("rmse"),
                                "mlflow_run_id": r.get("mlflow_run_id"),
                                "n_train": r.get("n_train"),
                                "n_test": r.get("n_test"),
                                "created_at": r.get("created_at"),
                            }
                            for r in records
                        ])
                        st.dataframe(
                            rdf.round(3),
                            width="stretch", hide_index=True,
                        )

        # ---- Raw markdown report (collapsed) -----------------------------
        if MLOPS_REPORT_MD.exists():
            with st.expander("Full markdown report"):
                st.markdown(MLOPS_REPORT_MD.read_text(encoding="utf-8"))


# ---------------------------------------------------------------------------
# Footer
# ---------------------------------------------------------------------------

st.sidebar.markdown("---")
st.sidebar.caption(
    "Read-only viewer. Pipelines (`load_files`, `build_monthly_features`, "
    "`train`, `evaluate`, `dq_report`) are CLI-driven; this dashboard never "
    "writes to the data tree."
)
