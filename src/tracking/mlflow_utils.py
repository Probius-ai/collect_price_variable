"""MLflow tracking helpers.

Design goals
------------
1.  MLflow logging is **optional**. Training pipelines must succeed whether
    or not a tracking server is reachable. Logging is enabled by either:
      * the ``ENABLE_MLFLOW=true`` env var, OR
      * the explicit ``--log-to-mlflow`` CLI flag (which wraps to
        ``enable=True`` in :func:`maybe_mlflow_run`).
    When enabled and the server is *not* reachable, the helper raises a
    clear error rather than silently dropping the run — partial logging
    is more confusing than no logging.

2.  All run metadata lands under a single experiment (default
    ``kpx-smp-monthly``), with a consistent param/metric naming scheme so
    the dashboard MLOps page can do simple SQL-style queries.

3.  A JSON registry fallback at
    ``outputs/model_registry/<model_name>_registry.json`` records the
    same run information even when MLflow's own model registry is
    unavailable (older MLflow versions, broken connection mid-run, etc.).
    The smoke-test pipeline uses this as the source of truth for the
    version-comparison report.

Usage
-----
    from src.tracking.mlflow_utils import maybe_mlflow_run

    with maybe_mlflow_run(enable=args.log_to_mlflow,
                           run_name="ridge_mainland_h1m",
                           tags={"version": "v3"}) as run:
        run.log_params({"model": "ridge", ...})
        run.log_metrics({"mae": 5.4, "rmse": 7.1})
        run.log_artifact(predictions_csv_path)

Outside an enabled run, the context manager yields a NO-OP object with
the same surface, so callers don't need ``if enabled:`` branches.
"""

from __future__ import annotations

import json
import os
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator


# ---------------------------------------------------------------------------
# Decision: enable / don't enable MLflow logging
# ---------------------------------------------------------------------------

_TRUE_VALUES = {"true", "1", "yes", "on"}


def mlflow_enabled_via_env() -> bool:
    """Read ``ENABLE_MLFLOW`` from the environment.

    Returns True when the env var is one of ``true|1|yes|on`` (case-
    insensitive). Anything else (missing, empty, ``false``, ``0``, …)
    returns False.
    """
    return os.environ.get("ENABLE_MLFLOW", "").strip().lower() in _TRUE_VALUES


def resolve_should_log(cli_flag: bool | None = None) -> bool:
    """Combine the CLI flag and the env var into one decision.

    * ``cli_flag=True``   → always log (and fail loud if server is down).
    * ``cli_flag=False``  → never log (explicit opt-out, even if env is on).
    * ``cli_flag=None``   → fall back to ``ENABLE_MLFLOW`` env var.

    Pipelines that don't have a CLI flag pass ``None`` and rely on env.
    """
    if cli_flag is True:
        return True
    if cli_flag is False:
        return False
    return mlflow_enabled_via_env()


# ---------------------------------------------------------------------------
# Tracking URI + experiment resolution
# ---------------------------------------------------------------------------


def tracking_uri() -> str:
    """Resolve the MLflow tracking URI from the environment.

    Order:
        1. ``MLFLOW_TRACKING_URI``
        2. Default ``http://localhost:5000`` (matches docker-compose)
    """
    return os.environ.get("MLFLOW_TRACKING_URI", "http://localhost:5000")


def experiment_name() -> str:
    return os.environ.get("MLFLOW_EXPERIMENT_NAME", "kpx-smp-monthly")


# ---------------------------------------------------------------------------
# JSON registry fallback
# ---------------------------------------------------------------------------
#
# Always written, even when MLflow logging is also enabled — so a corrupt
# Postgres / lost MLflow container doesn't take the version history with
# it. The smoke-test report writer reads from this file, not from MLflow,
# so the report works even with MLflow off.

REGISTRY_DIR = Path("outputs/model_registry")


@dataclass
class RegistryRecord:
    model_name: str
    version: str
    mlflow_run_id: str | None
    data_cutoff_month: str
    evaluation_mode: str
    metrics: dict[str, float]
    artifact_uri: str | None
    created_at: str
    promotion_status: str
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "model_name": self.model_name,
            "version": self.version,
            "mlflow_run_id": self.mlflow_run_id,
            "data_cutoff_month": self.data_cutoff_month,
            "evaluation_mode": self.evaluation_mode,
            "metrics": self.metrics,
            "artifact_uri": self.artifact_uri,
            "created_at": self.created_at,
            "promotion_status": self.promotion_status,
            **self.extra,
        }


def _registry_path(model_name: str) -> Path:
    REGISTRY_DIR.mkdir(parents=True, exist_ok=True)
    return REGISTRY_DIR / f"{model_name}_registry.json"


def append_registry_record(record: RegistryRecord) -> Path:
    """Append a record to the JSON registry for `record.model_name`.

    Records are stored as a JSON object: ``{"records": [...]}``. New
    records go at the end so the file reads chronologically.
    """
    path = _registry_path(record.model_name)
    existing: dict[str, Any]
    if path.exists():
        try:
            existing = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            # Corrupt file — preserve as `.broken` and start fresh.
            path.rename(path.with_suffix(".broken.json"))
            existing = {"records": []}
    else:
        existing = {"records": []}
    existing.setdefault("records", []).append(record.to_dict())
    path.write_text(
        json.dumps(existing, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )
    return path


def load_registry(model_name: str) -> list[dict[str, Any]]:
    """Return all records for the given model, or [] if the registry file
    doesn't exist."""
    path = _registry_path(model_name)
    if not path.exists():
        return []
    try:
        return json.loads(path.read_text(encoding="utf-8")).get("records", [])
    except json.JSONDecodeError:
        return []


# ---------------------------------------------------------------------------
# Run context manager
# ---------------------------------------------------------------------------
#
# A NO-OP run object that satisfies the same surface as the live one, so
# callers can `with maybe_mlflow_run(...) as run: run.log_params(...)`
# unconditionally.


class _NoopRun:
    """Stand-in returned when MLflow logging is disabled.

    Every method swallows its args silently — the goal is for pipeline
    code to be free of `if logging_enabled:` branches.
    """

    run_id: str | None = None

    def log_param(self, *_args: Any, **_kw: Any) -> None: ...
    def log_params(self, *_args: Any, **_kw: Any) -> None: ...
    def log_metric(self, *_args: Any, **_kw: Any) -> None: ...
    def log_metrics(self, *_args: Any, **_kw: Any) -> None: ...
    def log_artifact(self, *_args: Any, **_kw: Any) -> None: ...
    def log_artifacts(self, *_args: Any, **_kw: Any) -> None: ...
    def set_tag(self, *_args: Any, **_kw: Any) -> None: ...
    def set_tags(self, *_args: Any, **_kw: Any) -> None: ...


class _LiveRun:
    """Thin wrapper over the MLflow client surface, scoped to one run."""

    def __init__(self, mlflow_module: Any, run_id: str) -> None:
        self._mlflow = mlflow_module
        self.run_id = run_id

    def log_param(self, key: str, value: Any) -> None:
        self._mlflow.log_param(key, value)

    def log_params(self, params: dict[str, Any]) -> None:
        # MLflow rejects non-string params silently in old versions; coerce.
        self._mlflow.log_params({k: _stringify(v) for k, v in params.items()})

    def log_metric(self, key: str, value: float, step: int | None = None) -> None:
        self._mlflow.log_metric(key, float(value), step=step)

    def log_metrics(self, metrics: dict[str, float], step: int | None = None) -> None:
        clean = {k: float(v) for k, v in metrics.items() if v is not None and _is_finite(v)}
        if clean:
            self._mlflow.log_metrics(clean, step=step)

    def log_artifact(self, path: str | Path, artifact_path: str | None = None) -> None:
        self._mlflow.log_artifact(str(path), artifact_path=artifact_path)

    def log_artifacts(self, dir_path: str | Path, artifact_path: str | None = None) -> None:
        self._mlflow.log_artifacts(str(dir_path), artifact_path=artifact_path)

    def set_tag(self, key: str, value: Any) -> None:
        self._mlflow.set_tag(key, _stringify(value))

    def set_tags(self, tags: dict[str, Any]) -> None:
        self._mlflow.set_tags({k: _stringify(v) for k, v in tags.items()})


def _stringify(v: Any) -> str:
    if isinstance(v, (str, int, float, bool)) or v is None:
        return str(v)
    return json.dumps(v, ensure_ascii=False, default=str)


def _is_finite(v: float) -> bool:
    try:
        return float(v) == float(v) and float(v) not in (float("inf"), float("-inf"))
    except (TypeError, ValueError):
        return False


@contextmanager
def maybe_mlflow_run(
    *,
    enable: bool | None = None,
    run_name: str | None = None,
    tags: dict[str, Any] | None = None,
    nested: bool = False,
) -> Iterator[_LiveRun | _NoopRun]:
    """Context manager that conditionally opens an MLflow run.

    When ``enable`` resolves to True (see :func:`resolve_should_log`), a
    real MLflow run is started against the configured tracking URI and
    yielded as a :class:`_LiveRun`. The MLflow import + connection
    happens INSIDE the context — that way callers who don't enable
    logging don't pay the import cost and don't need `mlflow` installed.

    When disabled, yields a :class:`_NoopRun` so the caller's code path
    is identical.
    """
    if not resolve_should_log(enable):
        yield _NoopRun()
        return

    try:
        import mlflow  # type: ignore[import-not-found]
    except ImportError as e:
        raise RuntimeError(
            "MLflow logging was requested (ENABLE_MLFLOW or --log-to-mlflow) "
            "but the `mlflow` package is not installed. Run "
            "`pip install -r requirements.txt` or use the project Docker image."
        ) from e

    uri = tracking_uri()
    mlflow.set_tracking_uri(uri)
    exp_name = experiment_name()
    # `mlflow.set_experiment` creates the experiment if it doesn't exist;
    # on a bad tracking URI it raises here (which we let propagate — that's
    # the "fail loud" contract).
    try:
        mlflow.set_experiment(exp_name)
    except Exception as e:
        raise RuntimeError(
            f"MLflow logging was requested but the tracking server at {uri} "
            f"is not reachable (set_experiment('{exp_name}') failed: {e}). "
            "Bring it up with `docker compose up -d` or unset ENABLE_MLFLOW."
        ) from e

    with mlflow.start_run(run_name=run_name, nested=nested) as run:
        if tags:
            mlflow.set_tags({k: _stringify(v) for k, v in tags.items()})
        yield _LiveRun(mlflow, run.info.run_id)


# ---------------------------------------------------------------------------
# Tiny convenience for callers that want to write a run-summary JSON
# alongside the artifact bundle — useful when MLflow is off and we want
# at least one human-readable record.
# ---------------------------------------------------------------------------


def write_run_summary(
    out_dir: Path,
    *,
    params: dict[str, Any],
    metrics: dict[str, float],
    tags: dict[str, Any] | None = None,
    mlflow_run_id: str | None = None,
) -> Path:
    """Write `out_dir/run_summary.json` capturing this run end-to-end.

    Mirrors the MLflow run's params + metrics + tags so a developer who
    only has the artifact directory (no MLflow UI access) can still
    reconstruct what happened.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "run_id": mlflow_run_id,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "params": params,
        "metrics": metrics,
        "tags": tags or {},
    }
    path = out_dir / "run_summary.json"
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )
    return path
