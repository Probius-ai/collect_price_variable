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

## License

MIT.
