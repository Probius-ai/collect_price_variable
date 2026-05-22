# kpx-price-forecast

Korean electricity-market price-variable forecasting (SMP → 정산단가 → REC).
Phase 1 (this commit) focuses on a **reliable SMP data-collection + feature +
baseline-model pipeline** for the mainland (육지) hourly SMP target. See
`Plan.md` for the long-form design.

## Status

| Stage | Status |
| --- | --- |
| Project skeleton, config, utils | ✅ |
| BaseCollector + raw/snapshot/metadata persistence | ✅ |
| KPX SMP + demand-forecast collector | ✅ (schema fields are TBD until first live response) |
| Schema-discovery CLI | ✅ |
| Time / lag / rolling features with no-leakage contract | ✅ |
| Naive / Seasonal-naive / Ridge / LightGBM models | ✅ |
| Tests: time alignment, duplicates, missing hours, leakage | ✅ (24 tests) |
| KPX SMP-decision / generation / fuel-cost / settlement / REC collectors | ⏳ skeleton only |
| KMA weather / ECOS FX collectors | ⏳ skeleton only |

## Quick start

```bash
# 1. Create env + install
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# 2. Configure
cp .env.example .env
# Edit .env and set KPX_PUBLIC_API_KEY (data.go.kr decoded key).

# 3. Run tests
pytest -q

# 4. Discover the real SMP API schema (one-shot)
python -m src.pipelines.discover_schema --source kpx_smp_day_ahead \
    --param base_date=20240301
# → writes data/raw/kpx/smp/<date>/response_*.json + metadata
# → prints the JSON key structure so you can fill column_mapping.
# Update src/config/sources.yaml::sources.kpx_smp_day_ahead with:
#   - base_url, operation
#   - column_mapping (replace TBD_AFTER_FIRST_RESPONSE entries)
#   - units (demand_forecast: MW vs MWh — verify!)
#   - schema_version (bump it)

# 5. Backfill SMP
python -m src.pipelines.collect_all --source kpx_smp \
    --start 2024-01-01 --end 2024-12-31

# 6. Build features (T+24h target)
python -m src.pipelines.build_features --target smp_hourly --area mainland --horizon 24

# 7. Train baselines + LightGBM
python -m src.pipelines.train --features-path data/processed/smp_hourly_mainland_h24.parquet --model naive
python -m src.pipelines.train --features-path data/processed/smp_hourly_mainland_h24.parquet --model seasonal_naive
python -m src.pipelines.train --features-path data/processed/smp_hourly_mainland_h24.parquet --model ridge
python -m src.pipelines.train --features-path data/processed/smp_hourly_mainland_h24.parquet --model lightgbm

# 8. Compare
python -m src.pipelines.evaluate
```

## Repository layout

```
src/
  config/        settings + sources.yaml (per-source assumptions, schema versions)
  collectors/    BaseCollector + KpxSmpCollector (skeletons for the rest)
  features/      time_features, lag_features, build (SMP hourly feature table)
  models/        naive, seasonal_naive, ridge, lightgbm + metrics
  pipelines/     discover_schema, collect_all, build_features, train, evaluate
  utils/         io (raw/snapshot/metadata writers), time, logging, storage (DuckDB)
  validation/    leakage_checks (duplicates, gaps, future-leakage)
tests/          24 tests covering time, duplicates, leakage, parser, models
data/           raw/ interim/ processed/ snapshots/   (gitignored payloads)
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

## Known TBDs (must be confirmed against live responses)

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

Anything still TBD when running collectors will surface as a `CollectorError`
that names the source and the specific YAML key to fix — by design.

## Models

* **Naive (lag_24h)** — predicts `target = smp_lag_24h`. (Plan.md §9.1)
* **Seasonal naive (lag_168h)** — same-hour-last-week.
* **Ridge** — scaled linear baseline over a curated feature subset.
* **LightGBM** — main tabular model; reports gain/split importance.

Splits are strictly chronological (`train -> valid -> test`). Random splits
are not supported. Walk-forward evaluation is intentionally not wired into
the CLI yet — Plan.md §11.1 lists it as a follow-up after we have ≥ 2 years
of hourly SMP loaded.

## Tests

```bash
pytest -q
```

The suite includes:

* `test_time_alignment.py` — KPX 1..24 trade-hour to interval-end mapping.
* `test_duplicates_and_missing.py` — duplicate-key and hourly-gap detection.
* `test_no_future_leakage.py` — lag/rolling correctness + explicit leakage canary.
* `test_kpx_smp_parser.py` — parser refuses TBD mappings; renames vendor keys correctly.
* `test_models.py` — naive, ridge, lightgbm round-trip.

All tests run offline; the only network code is gated behind real
collectors that need a valid `KPX_PUBLIC_API_KEY`.

## What's intentionally NOT in this commit

Per Plan.md §9 and Priority 9 in the brief:

* REC and settlement model pipelines (collectors are stubbed; modelling not built).
* Generation / fuel-cost / weather feature joins (collectors stubbed; columns to add later).
* SARIMAX / XGBoost / TFT models (post-MVP per Plan.md §9.7).

## License

MIT.
