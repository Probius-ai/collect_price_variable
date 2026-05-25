# Project Docker image — used by docker-compose.yml for the MLflow tracking
# server AND as the runtime for training/smoke-test commands.
#
# Build: `docker compose build` (or `docker build -t kpx-mlops .`)
# Run:   `docker compose up -d` (boots Postgres + MLflow)
# Train: `docker compose run --rm app python -m src.pipelines.train ...`
#
# Single-image design: the MLflow tracking server doesn't have any
# project-code dependencies — it just needs `mlflow` + `psycopg2-binary`
# — but baking them into the same image as the training code keeps the
# compose file simple and means there's only ONE thing to rebuild when
# `requirements.txt` changes.

FROM python:3.11-slim

# Postgres client lib + build tools needed by psycopg2-binary on some
# architectures, plus `libgomp1` for LightGBM's OpenMP runtime.
RUN apt-get update \
 && apt-get install -y --no-install-recommends \
        build-essential \
        libpq-dev \
        libgomp1 \
        curl \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install dependencies first so a code-only change doesn't bust the
# pip layer cache.
COPY requirements.txt ./
RUN pip install --no-cache-dir --upgrade pip \
 && pip install --no-cache-dir -r requirements.txt

# Copy the project last. The compose file mounts the working tree over
# this during dev, so live edits don't require a rebuild.
COPY . .

# Default to a shell. The compose file overrides this per-service
# (mlflow service runs `mlflow server …`, the app service runs whatever
# CLI you invoke via `docker compose run …`).
CMD ["/bin/bash"]
