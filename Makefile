# One-command helpers for the web stack (FastAPI + Next.js) and the
# existing Streamlit + MLflow stack.
#
# All targets are .PHONY because none of them produce a single file
# that make can hash to skip work — they're orchestration commands.

.PHONY: help api web web-build web-install streamlit mlflow stop-web check-web

PY := .venv/bin/python
UVICORN := .venv/bin/uvicorn

help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-15s\033[0m %s\n", $$1, $$2}'

# ---------------------------------------------------------------------------
# Web stack (Next.js + FastAPI)
# ---------------------------------------------------------------------------

api: ## Start FastAPI on :8000 (Ctrl+C to stop)
	$(UVICORN) api.main:app --reload --host 127.0.0.1 --port 8000

web-install: ## npm install for the Next.js frontend
	cd web && npm install

web: ## Start Next.js dev server on :3000 (Ctrl+C to stop; requires `make api` running)
	cd web && npm run dev

web-build: ## Production build of the Next.js frontend
	cd web && npm run build

stop-web: ## Kill anything listening on :3000 and :8000 (Next.js + FastAPI)
	-@lsof -ti :3000 | xargs -r kill 2>/dev/null
	-@lsof -ti :8000 | xargs -r kill 2>/dev/null
	@echo "web stack stopped"

check-web: ## Verify both services respond
	@echo "FastAPI health:"
	@curl -sS http://127.0.0.1:8000/api/health || echo "  → not running"
	@echo
	@echo "Next.js home:"
	@curl -sS -o /dev/null -w "  HTTP %{http_code}\n" http://127.0.0.1:3000/ || echo "  → not running"

# ---------------------------------------------------------------------------
# Existing Streamlit + MLflow stack (separate from the web frontend)
# ---------------------------------------------------------------------------

streamlit: ## Start the legacy Streamlit dashboard on :8501
	$(PY) -m streamlit run dashboard.py

mlflow: ## Start MLflow tracking server on :5000 (SQLite backend)
	.venv/bin/mlflow server \
	    --backend-store-uri sqlite:///mlflow.db \
	    --default-artifact-root ./artifacts/mlflow \
	    --host 0.0.0.0 --port 5000 --serve-artifacts
