# kpx-price-forecast

Korean electricity-market price-variable forecasting (SMP → 정산단가 → REC).
The repo has two stacks running side-by-side:

* **Pre-approval MVP** (file-based). Drop KPX/EPSIS downloads under
  `data/raw/manual_or_filedata/<source>/`, run the file loaders, build the
  **monthly** SMP feature table, train naive/Ridge/LightGBM baselines.
* **Post-approval hourly pipeline** (API-based). Once
  `KPX_PUBLIC_API_KEY` is active, the hourly SMP + demand-forecast collector
  takes over and emits parsed Parquet files in the same on-disk layout as the
  file loaders, so the feature builder doesn't change.

See `Plan.md` for the long-form design.

## Status

| Stage | Status |
| --- | --- |
| Project skeleton, config, utils | ✅ |
| BaseCollector + raw/snapshot/metadata persistence (API path) | ✅ |
| KPX SMP + demand-forecast API collector | ✅ (schema fields TBD until first live response) |
| BaseFileLoader + raw/parsed/metadata persistence (pre-approval path) | ✅ |
| KPX monthly file loaders (SMP mainland/Jeju/integrated, settlement, REC monthly, REC weekly) | ✅ |
| KPX yearly SMP file loader (EDA only) | ✅ |
| Hourly + monthly feature builders, no-leakage contract | ✅ |
| Naive / Seasonal-naive / Ridge / LightGBM (hourly + monthly) | ✅ |
| Tests | ✅ (35 tests) |
| KPX SMP-decision / generation / fuel-cost API collectors | ⏳ skeleton only |
| KMA weather / ECOS FX collectors | ⏳ skeleton only |

## Quick start — pre-approval MVP (file-based)

While the API key is still pending, work entirely from KPX/EPSIS downloads.
**Do not call any API collector** — every `src/pipelines/discover_schema`
or `collect_all` invocation needs `KPX_PUBLIC_API_KEY` and will hard-fail
without one.

```bash
# 1. Create env + install
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# 2. Drop downloaded files (CSV/XLSX from KPX/EPSIS) into the inbox.
#    Use one folder per file source, named after the sources.yaml key.
mkdir -p data/raw/manual_or_filedata/kpx_smp_monthly_mainland
cp ~/Downloads/SMP_육지_월별.csv data/raw/manual_or_filedata/kpx_smp_monthly_mainland/

# 3. Inspect the raw headers so you can fill in the column mapping.
python -m src.pipelines.discover_file \
    --source kpx_smp_monthly_mainland \
    --path data/raw/manual_or_filedata/kpx_smp_monthly_mainland/SMP_육지_월별.csv
#    → prints column names + sample rows + a checklist.
#    Edit src/config/sources.yaml::file_sources.kpx_smp_monthly_mainland:
#      - source_url        (KPX/EPSIS landing page)
#      - file_format       (csv | xlsx)
#      - encoding          (utf-8 | cp949)
#      - year_month_format ("%Y-%m" | "%Y%m" | "%Y년 %m월" …)
#      - column_mapping    (replace TBD_AFTER_FIRST_DOWNLOAD)
#      - schema_version    (bump)
#      - last_verified_at  (today's date)

# 4. Parse + persist. Writes:
#    - raw copy:   data/raw/kpx/smp_monthly_mainland/<YYYY>/<MM>/<DD>/raw_*.csv
#    - parsed:     data/raw/kpx/smp_monthly_mainland/.../parsed_*.parquet
#    - metadata:   data/raw/kpx/smp_monthly_mainland/.../metadata_*.json
#                  (source_url, frequency, unit, limitations, sha256, etc.)
python -m src.pipelines.load_files --source kpx_smp_monthly_mainland

# 5. Repeat 2–4 for the other file sources you want included:
#    - kpx_smp_monthly_jeju          (separate model — never concat with mainland)
#    - kpx_smp_monthly_integrated    (integrated target; only post-2024-02)
#    - kpx_smp_yearly                (EDA only; refuses monthly modelling)
#    - kpx_settlement_monthly_file   (revisable — track collected_at)
#    - kpx_rec_monthly_file          (units: REC / million KRW; confirm header)
#    - kpx_rec_weekly_file           (verify week-start vs week-end)

# 6. Build the monthly SMP feature table (mainland, T+1 month horizon).
python -m src.pipelines.build_monthly_features \
    --area mainland --horizon-months 1

# 7. Train monthly baselines + Ridge + LightGBM.
F=data/processed/smp_monthly_mainland_h1m.parquet
TS=year_month
Y=target_smp_t_plus_h_months
python -m src.pipelines.train --features-path $F --model naive_lag_1m            --target $Y --timestamp-col $TS --min-train-rows 24
python -m src.pipelines.train --features-path $F --model seasonal_naive_lag_12m  --target $Y --timestamp-col $TS --min-train-rows 24
python -m src.pipelines.train --features-path $F --model ridge                   --target $Y --timestamp-col $TS --min-train-rows 24
python -m src.pipelines.train --features-path $F --model lightgbm                --target $Y --timestamp-col $TS --min-train-rows 24
python -m src.pipelines.evaluate
```

## Quick start — post-approval (hourly SMP API)

Run only after `KPX_PUBLIC_API_KEY` is active.

```bash
# 1. Configure
cp .env.example .env
# Edit .env and set KPX_PUBLIC_API_KEY (data.go.kr decoded key).

# 2. Run tests
pytest -q

# 3. Discover the real SMP API schema (one-shot)
python -m src.pipelines.discover_schema --source kpx_smp_day_ahead \
    --param base_date=20240301
# → writes data/raw/kpx/smp_day_ahead/<date>/response_*.json + metadata
# → prints the JSON key structure so you can fill column_mapping.
# Update src/config/sources.yaml::sources.kpx_smp_day_ahead with:
#   - base_url, operation
#   - column_mapping (replace TBD_AFTER_FIRST_RESPONSE entries)
#   - units (demand_forecast: MW vs MWh — verify!)
#   - schema_version (bump it)

# 4. Backfill SMP
python -m src.pipelines.collect_all --source kpx_smp \
    --start 2024-01-01 --end 2024-12-31

# 5. Build features (T+24h target)
python -m src.pipelines.build_features --target smp_hourly --area mainland --horizon 24

# 6. Train baselines + LightGBM
python -m src.pipelines.train --features-path data/processed/smp_hourly_mainland_h24.parquet --model naive
python -m src.pipelines.train --features-path data/processed/smp_hourly_mainland_h24.parquet --model seasonal_naive
python -m src.pipelines.train --features-path data/processed/smp_hourly_mainland_h24.parquet --model ridge
python -m src.pipelines.train --features-path data/processed/smp_hourly_mainland_h24.parquet --model lightgbm

# 7. Compare
python -m src.pipelines.evaluate
```

## Forecast contract + persistence baseline

The monthly model emits **1-month-ahead forecasts** under this contract:

* **forecast_origin_month** = the month *after* which the forecast is made
  (i.e. all information up to and including this month is observable).
* **information_cutoff** = `end-of-day(last day of forecast_origin_month)`.
* **target_month** = `forecast_origin_month + 1 month`.
* **horizon** = `"1M"`.

Every row in `data/processed/smp_monthly_<area>_h1m.parquet` and every row
in `outputs/models/<m>/predictions_<split>.csv` now carries these four
columns. Under this contract, `smp_t_observed` (SMP at the forecast
origin month) is fully observable before the target month and is **not
leakage**.

### Persistence is the strong baseline

`persistence_monthly` predicts `SMP(M+1) = SMP(M)` — i.e. it always
returns the last observed value. On the dashboard, its prediction line
**naturally follows the actual series with a one-month delay**; that is
the floor every trainable model must beat. We separate baselines from
trainable models in `src/models/registry.py` and the dashboard
Predictions page defaults to the strongest trainable model
(`monthly_ar_ridge`), overlaying persistence as a reference line.

### Limits of monthly-only data

**Reducing the visible one-month lag is not possible with the current
monthly file-based pipeline alone.** Three things would change that:

1. **Leading exogenous variables** — e.g. next-month natural-gas-price
   futures, planned nuclear outage schedules, ANRE quota changes. These
   are signals about month M+1 that are known at the M cutoff.
2. **Hourly / day-ahead API data** — the KPX SMP day-ahead API (pending
   approval) provides intra-month variability whose intra-month dynamics
   the monthly aggregate erases.
3. **Multi-horizon forecasts** — direct-multi-step or recursive
   forecasting can give 2- and 3-month-ahead estimates whose error has a
   different decomposition than the 1-step task.

The current code is conservative on these fronts: it documents the
persistence floor, and reports both level metrics and delta-vs-baseline
diagnostics so honest improvement is measurable.

## Baseline snapshots (before/after comparison)

`src/pipelines/save_baseline_plots.py` freezes the current model results
into a tagged folder under `outputs/figures/<tag>/`. The intended flow:

1. Take a snapshot of today's pipeline (no LNG-price forecasting yet):

   ```bash
   python -m src.pipelines.save_baseline_plots
   # → outputs/figures/baseline_pre_lng_forecast_<YYYYMMDD>/
   ```

2. Later, after integrating LNG-price forecasts (or any other
   leading-indicator feature), take a second snapshot under a different
   tag:

   ```bash
   python -m src.pipelines.save_baseline_plots --tag baseline_post_lng_v1
   ```

   Pass `--docs-copy` if a committed markdown report (e.g. the project
   report in `docs/project_report_kr.md`) references these images — that
   mirrors the PNGs into `docs/figures/<tag>/` so the report survives a
   fresh clone (the original `outputs/figures/**` path is gitignored):

   ```bash
   python -m src.pipelines.save_baseline_plots \
       --tag baseline_post_lng_v1 --docs-copy
   ```

3. Diff the two folders to show the lag-reduction visually.

Each snapshot contains 6 PNG plots (predictions overlay, persistence-lag
zoom, residuals, MAE/R² bar chart, walk-forward over 150 months,
feature-group ablation heatmap), per-model raw predictions CSV, and a
`summary.md` recording the metric values + the forecast contract in
effect.

## Dashboard (read-only viewer)

A Streamlit dashboard at [`dashboard.py`](dashboard.py) reads the artefacts
produced by the CLI pipelines (parsed parquets, processed feature tables,
`outputs/metrics/comparison.csv`, model predictions, side-info JSONs, DQ
reports) and presents them through 5 pages: Overview, Models, Predictions,
Features, Data Quality. It never writes to the data tree.

```bash
pip install -r requirements.txt        # includes streamlit + plotly
streamlit run dashboard.py             # opens http://localhost:8501
```

What each page shows:

| Page | Source files |
| --- | --- |
| 1. Overview | `data/raw/kpx/*/parsed_*.parquet` inventory + drop-inbox state |
| 2. Models | `outputs/metrics/comparison.csv` (MAE/RMSE/MAPE × split) |
| 3. Predictions | `outputs/models/<model>/predictions_<split>.csv` overlay |
| 4. Features | `data/processed/smp_monthly_<area>_h1m.parquet` + sideinfo, grouped by feature family (baseline / settlement / capacity / transaction) |
| 5. Data Quality | latest `outputs/data_quality/report_*.json` + per-area `_priority_dedup_log` |

## Repository layout

```
src/
  config/        settings + sources.yaml (API `sources:` + file `file_sources:`)
  collectors/    BaseCollector (API), BaseFileLoader (manual download),
                 KpxSmpCollector (hourly API), kpx_files.py (monthly/weekly/yearly file loaders)
  features/      time_features, lag_features, build (hourly), build_monthly (monthly)
  models/        naive (hourly + monthly variants), seasonal_naive, ridge, lightgbm + metrics
  pipelines/     discover_schema, collect_all          ← API path
                 discover_file, load_files             ← file path
                 build_features, build_monthly_features, train, evaluate
  utils/         io (raw/snapshot/metadata writers), time, logging, storage (DuckDB)
  validation/    leakage_checks (duplicates, gaps, future-leakage)
tests/          35 tests covering time, duplicates, leakage, parser, models,
                file loader, monthly features, collector↔builder path contract
data/
  raw/manual_or_filedata/<source>/      ← user drops downloads here
  raw/<namespace>/<rest>/YYYY/MM/DD/    ← canonical layout used by both
                                          file loaders and API collectors
  processed/                            ← feature tables (hourly + monthly)
outputs/        models/ metrics/ figures/ predictions/  (gitignored artefacts)
```

## No-future-leakage contract

The feature builder and the leakage check together enforce one rule:

> A feature stamped at `interval_end = t` may only be a function of values
> whose `interval_end` is strictly < t.

* `add_lag_features` rejects `periods <= 0` so positive-shift is the only path.
* `add_rolling_features` shifts by 1 hour **before** rolling, so a 24-hour
  window at time t covers `[t-24h, t-1h]`, never t itself.
* `build_smp_hourly_features` reindexes to a contiguous hourly grid per area
  before shifting, so a missing hour becomes NaN instead of silently pulling
  a later value into the gap.
* `assert_no_future_leakage` raises if any feature column ends up exactly
  equal to the target column (canary against accidental future joins).

## Schema-discovery / "no invented fields" rule

`sources.yaml` is the **only** place where vendor-to-canonical field mappings
live. Every entry that is not yet verified against a real response is
explicitly marked `TBD_AFTER_FIRST_RESPONSE`. The KPX SMP parser refuses to
materialise rows while any of its mapping entries are still TBD — preventing
silent invention of field names.

Workflow:

1. Run `discover_schema` for the source.
2. Open the saved JSON/XML under `data/raw/<source>/.../response_*.json`.
3. Update `sources.yaml::column_mapping` with the verified vendor keys.
4. Bump `schema_version` and set `last_verified_at`.
5. Re-run the collector.

## Modular source adapters (file ↔ API interchange)

Both `BaseCollector` (API) and `BaseFileLoader` (file) write to the SAME path
scheme: `data/raw/<namespace>/<rest>/YYYY/MM/DD/parsed_*.parquet`. The
feature builders resolve their input directories via
`src.utils.io.source_root_dir(<source_name>)` — they never hard-code paths.
That means swapping a file source for the equivalent API collector is a
two-step change:

1. Add an API entry under `sources:` in `sources.yaml` with the canonical
   internal column names (the same names the file loader emits).
2. Write a `BaseCollector` subclass that produces a Parquet with those
   canonical columns.

No feature, model, or pipeline code needs to change.

## Known TBDs (must be confirmed against live responses or downloaded files)

| Source | TBD | Why |
| --- | --- | --- |
| `kpx_smp_day_ahead.base_url` / `.operation` | data.go.kr operation path | Documentation page is the authoritative source |
| `kpx_smp_day_ahead.column_mapping.*` | vendor field names | Do not guess `smpPrice` etc; discover and confirm |
| `kpx_smp_day_ahead.request_params.base_date_param` | request date param name | Pages differ (`baseDate`, `trDay`, `date`); confirm |
| `kpx_smp_day_ahead.units.demand_forecast` | MW vs MWh | The portal description does not specify clearly |
| `kpx_smp_decision_by_fuel.endpoint*` | pre/post 2024-08-20 endpoints | Schema split announced by data.go.kr notice |
| `kpx_generation_5min.column_mapping.*` | vendor field names | 5-min API has known scale caveats; confirm units |
| `kpx_fuel_cost_monthly.units.fuel_cost` | KRW/Gcal vs KRW/kg vs … | Portal page lists multiple keywords |
| `kpx_settlement_monthly.column_mapping.*` | fuel/member breakdown | Subject to revision; track snapshot dates |
| `kpx_rec_spot.units.trade_amount` | KRW vs 백만원 | Both used elsewhere; verify |
| `kma_asos_hourly.column_mapping.*` | observation field names | Multiple variants between API endpoints |
| `kpx_smp_monthly_*.column_mapping.*` | vendor headers | Discover from downloaded CSV/XLSX; mainland/Jeju/integrated kept separate |
| `kpx_smp_monthly_*.year_month_format` | "%Y-%m" vs "%Y%m" vs "%Y년 %m월" | KPX exports vary; confirm before parsing |
| `kpx_smp_yearly.region` | mainland \| jeju \| integrated | Each yearly file is a single region; tag in YAML |
| `kpx_settlement_monthly_file.column_mapping.*` | 연료원/회원사 layout | EPSIS shape changes between revisions |
| `kpx_rec_monthly_file.units` | 원 vs 백만원 | Vendor uses both — confirm in file header |
| `kpx_rec_weekly_file.week_start_format` | week-start vs week-end | Verify which date the row refers to |

Anything still TBD when running collectors will surface as a `CollectorError`
that names the source and the specific YAML key to fix — by design.

## Models

Hourly (post-approval):

* **Naive (lag_24h)** — `target = smp_lag_24h`. (Plan.md §9.1)
* **Seasonal naive (lag_168h)** — same-hour-last-week.
* **Ridge** — scaled linear baseline (auto-detects hourly columns).
* **LightGBM** — main tabular model; reports gain/split importance.

Monthly (pre-approval MVP):

* **naive_lag_1m** — `target = smp_lag_1m`.
* **seasonal_naive_lag_12m** — same-month-last-year.
* **Ridge** — auto-switches to monthly feature defaults
  (`smp_lag_1..12m`, `smp_rolling_3/6/12m_*`, `month_sin/cos`, settlement/REC lag_1m).
* **LightGBM** — note: needs enough rows to beat the default
  `min_data_in_leaf=50`; pass `--min-train-rows 24` plus a custom
  `lightgbm_model.LightGBMModel(params={"min_data_in_leaf": 5})` for very
  small monthly datasets.

Splits are strictly chronological (`train → valid → test`). Random splits
are not supported. Walk-forward evaluation is intentionally not wired into
the CLI yet — Plan.md §11.1 lists it as a follow-up after we have ≥ 2 years
of data loaded.

## Tests

```bash
pytest -q
```

The suite includes:

* `test_time_alignment.py` — KPX 1..24 trade-hour to interval-end mapping.
* `test_duplicates_and_missing.py` — duplicate-key and hourly-gap detection.
* `test_no_future_leakage.py` — hourly lag/rolling correctness + explicit leakage canary.
* `test_kpx_smp_parser.py` — API parser refuses TBD mappings; renames vendor keys correctly.
* `test_models.py` — naive, ridge, lightgbm round-trip.
* `test_collector_paths.py` — collector and feature builder agree on disk layout.
* `test_file_loaders.py` — pre-approval loader refuses TBD mappings; persists raw/parsed/metadata with source_url + unit + limitations.
* `test_monthly_features_and_models.py` — monthly lag/rolling correctness, no-leakage canary, seasonal_naive beats lag_1m on seasonal data, end-to-end build.

All tests run offline; network code is gated behind API collectors that need
a valid `KPX_PUBLIC_API_KEY`.

## What's intentionally NOT in this commit

Per Plan.md §9 and the user brief:

* Dedicated REC / settlement *model* pipelines (file loaders exist; modelling
  joins them as monthly exogenous lags but no standalone REC/settlement target
  predictor — that lands after the SMP pipeline is stable).
* Generation / fuel-cost / weather API collectors (skeletons in
  `sources.yaml::sources` only; feature joins land after their loaders ship).
* SARIMAX / XGBoost / TFT (post-MVP per Plan.md §9.7).
* Walk-forward CV — current splits are single chronological train/valid/test.

## MLOps — MLflow tracking + v1..v5 retraining smoke test

The repo ships a Docker-Compose MLOps stack (MLflow Tracking Server +
Postgres backend) plus a staged-retraining smoke test that simulates
five rounds of "new data arrived → retrain → record". Designed for the
MLOps course demo: every model run lands in the MLflow UI AND in a JSON
registry fallback under `outputs/model_registry/`.

### Stack overview

* `docker-compose.yml` — Postgres 16 (host port `5433`) + MLflow Tracking
  Server (host port `5000`, `--serve-artifacts`, artifacts persisted on
  host under `./artifacts/mlflow`). Built from the project `Dockerfile`
  so the MLflow container has all project deps too.
* `src/tracking/mlflow_utils.py` — `maybe_mlflow_run()` context
  manager (no-op when disabled, fails loud when enabled but unreachable),
  plus the JSON registry fallback (`RegistryRecord` + `append_registry_record`).
* `src/pipelines/mlops_smoke_test.py` — v1..v5 orchestrator.
* `config/mlops_smoke_test.yaml` — version + model definitions.

### Start the stack

```bash
# 1. Boot Postgres + MLflow (port 5000 = MLflow UI, 5433 = Postgres on host).
docker compose up -d

# 2. Open the MLflow UI
xdg-open http://localhost:5000  # or just visit it in a browser
```

### Run the smoke test

The smoke test simulates retraining at five cutoff months
(2021-12 / 2022-12 / 2023-12 / 2024-12 / 2025-08). v1..v4 score against
a forward holdout; v5 has no future labels and falls back to rolling
validation against the last 12 months of its own training window.

```bash
# Local Python (uses your venv). No MLflow logging — registry JSON only.
python -m src.pipelines.mlops_smoke_test --config config/mlops_smoke_test.yaml

# Same, but with MLflow tracking enabled (server must be up).
python -m src.pipelines.mlops_smoke_test \
    --config config/mlops_smoke_test.yaml \
    --log-to-mlflow

# Subset for faster iteration during dev
python -m src.pipelines.mlops_smoke_test \
    --config config/mlops_smoke_test.yaml \
    --only-versions v1,v2 \
    --only-models persistence_monthly,ridge

# Via the project container (uses MLFLOW_TRACKING_URI=http://mlflow:5000
# inside the compose network; ENABLE_MLFLOW=true is baked in)
docker compose run --rm app python -m src.pipelines.mlops_smoke_test \
    --config config/mlops_smoke_test.yaml --log-to-mlflow
```

### Outputs

Every run regenerates these (all gitignored):

| Path | What |
| --- | --- |
| `outputs/mlops_smoke_test/<vN>/<model>/predictions.csv` | Per-(version, model) predictions |
| `outputs/mlops_smoke_test/<vN>/<model>/metrics.json` | MAE/RMSE/MAPE/R²/directional |
| `outputs/mlops_smoke_test/<vN>/<model>/model.pkl` | Pickled fit (some models skip — see warnings) |
| `outputs/mlops_smoke_test/<vN>/<model>/run_summary.json` | Mirror of MLflow params+metrics+tags |
| `outputs/model_registry/<model_name>_registry.json` | Per-model version history (works without MLflow) |
| `outputs/reports/mlops_smoke_test_v1_v5_report.md` | Human-readable summary |
| `outputs/reports/mlops_smoke_test_v1_v5_report.json` | Dashboard-friendly JSON |
| `outputs/metrics/mlops_version_comparison.csv` | Flat (version, model) table |

### Promotion policy

The smoke test never auto-promotes to production. Each row is tagged:

* `historical_backtest`   — v1..v4 fixed-holdout runs
* `latest_candidate`      — v5 rolling-validation runs
* `recommended_historical` — best v1..v4 by MAE (one row, across all models)
* `skipped`               — controlled-skip (insufficient data, model error)

Tag a model as production via the MLflow UI manually, or extend
`outputs/model_registry/*_registry.json` with your own promotion record.

### Dashboard

Streamlit Page 10 (**"MLOps smoke test (v1-v5)"**) renders the
comparison CSV + registry JSON, with a per-metric trend chart and a
promotion-status breakdown per version. Start the dashboard
(`streamlit run dashboard.py`) and pick the page from the sidebar.

### Architecture notes

* MLflow logging is **optional** — every pipeline supports both modes:
    * `--log-to-mlflow` flag, OR `ENABLE_MLFLOW=true` env var → live run.
    * Neither set → no-op log calls, JSON registry still written.
* Server unreachable + logging requested → **fail loud** (not silent).
* `filter_to_cutoff()` is the load-bearing leak-safety helper for
  training data. Test sets pull from the FULL panel since labels become
  observable in retrospect even if they wouldn't be at the simulated
  retraining moment.
* v5 deliberately uses rolling validation — no future-month labels are
  fabricated. Promotion language reflects this ("latest_candidate",
  not "fixed_holdout").

### Out of scope (intentionally)

Per the spec: no cron retraining, no file-arrival watcher, no
Kubernetes/AWS, no automatic production promotion, no copying/deleting
raw files to simulate data arrival. The "staged retraining" is a
filter-based simulation, not a real-data pipeline.

## Web frontend (Next.js + FastAPI)

A production-grade web UI for the model-selection decision, separate
from the legacy Streamlit dashboard.

* `api/` — **FastAPI** backend on port 8000. Read-only JSON endpoints
  for the comparison table, recommendation, registry, solar-integration
  status, and a `POST /api/retrain` that fires the same v1..v5 smoke
  test the dashboard does.
* `web/` — **Next.js 14** (App Router + TypeScript + Tailwind) on
  port 3000. Server-rendered home page consumes the API; the retrain
  button is a client component that polls `/api/retrain/status` every
  5 seconds while a run is in flight.

### Quick start

```bash
# One-time: install Next.js deps (Node 18+ required)
make web-install

# Terminal 1 — FastAPI backend
make api

# Terminal 2 — Next.js dev server
make web

# Visit: http://localhost:3000
```

Or run both manually:

```bash
.venv/bin/uvicorn api.main:app --reload --port 8000   # API
cd web && npm run dev                                  # frontend
```

### Page layout

The single home page at `/` (Korean UI by default) shows:

| Section | Source |
| --- | --- |
| 의사결정 패널 (4 metric cards) | `GET /api/models/recommendation` |
| 후보 모델 안정성 (v4 → v5 변화 표) | `GET /api/models/comparison` |
| 왜 단기 태양광 예측이 SMP 모델 선정에 중요한가 | static markdown |
| 외부 모델 인벤토리 (Solar / LNG) | `GET /api/solar/integration` |
| 학습 트리거 (강제 재학습 버튼 + 6시간마다 자동 재학습 안내) | `POST /api/retrain` + polling |

### Production build

```bash
make web-build                                        # builds .next/
cd web && npm run start                               # serves :3000
```

For real deployment behind a reverse proxy, set
`NEXT_PUBLIC_API_BASE=https://your-domain/api` in `web/.env.production`
so the browser hits the same origin instead of `localhost:8000`.

### Architecture note

The FastAPI subprocess for retrain uses `sys.executable` so it works
in any environment (Docker, fresh venv, etc.) — no hardcoded
`.venv/bin/python`. The `outputs/_retrain_status.json` lock file is
shared with the Streamlit dashboard's retrain button, so both UIs
correctly detect "already running" state.

## License

MIT.
