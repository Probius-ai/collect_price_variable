# 시스템 아키텍처 — kpx-price-forecast

**문서 종류:** 시스템/아키텍처 명세
**작성일:** 2026-05-26
**범위:** round 1–9 누적 결과 (LNG/JKM/EIA/KMA 통합, iterative models, MLOps 운영, 알림/스케줄링 포함)

기존 [`project_report_kr.md`](project_report_kr.md)는 5월 25일자 방법론/결과 보고서이며,
본 문서는 **시스템 구성·데이터 흐름·운영**에 초점을 둔 별도 참조 문서.

---

## 목차

1. [한 줄 요약](#1-한-줄-요약)
2. [시스템 다이어그램](#2-시스템-다이어그램)
3. [디렉토리 구조](#3-디렉토리-구조)
4. [데이터 파이프라인](#4-데이터-파이프라인)
5. [수집기 카탈로그](#5-수집기-카탈로그)
6. [모델 카탈로그](#6-모델-카탈로그)
7. [평가 프레임워크](#7-평가-프레임워크)
8. [인터페이스 레이어](#8-인터페이스-레이어)
9. [운영 / MLOps](#9-운영--mlops)
10. [보안 / 시크릿 정책](#10-보안--시크릿-정책)
11. [개발 워크플로우](#11-개발-워크플로우)
12. [의존성](#12-의존성)

---

## 1. 한 줄 요약

한국 전력시장 SMP(System Marginal Price) **월별 1개월 선행 예측** 파이프라인.
KPX/공공 API + 파일 드롭 inbox로 데이터 수집 → 월별 panel feature → 13개 모델 학습 →
chronological holdout + walk-forward CV 평가 → MLflow 추적 + Streamlit/Next.js UI + Discord 알림.

데이터 빈도는 월별이지만 시간별로 확장 가능한 이중 백엔드 구조
(`BaseCollector` for APIs, `BaseFileLoader` for manual drops)로 설계됨.

---

## 2. 시스템 다이어그램

```
┌─────────────────────────────────────────────────────────────────┐
│  외부 데이터 소스                                                  │
│  - KPX EPSIS (SMP, 설비용량, 거래량/금액, REC)                       │
│  - data.go.kr (KMA 단기/중기예보)                                  │
│  - EIA STEO API (Brent, WTI, Henry Hub)                          │
│  - investing.com (JKM LNG 일별/forward curve, oil/FX 월별)         │
│  - solar_beam DB (raw_weather 시간별 → 월별 집계)                  │
└──────────┬───────────────────────────────────────┬──────────────┘
           │ API                                   │ CSV/XLSX manual drop
           ▼                                       ▼
   ┌──────────────────┐                  ┌──────────────────────┐
   │ src/collectors/  │                  │ data/raw/manual_     │
   │  - kpx_smp.py    │                  │   or_filedata/<src>/ │
   │  - eia_steo.py   │                  │  (drop inbox)        │
   │  - kma_*.py      │                  └──────────┬───────────┘
   └────────┬─────────┘                             │
            │ (verbatim raw payload + parsed)       │ src/collectors/file_loader.py
            ▼                                       ▼
   ┌─────────────────────────────────────────────────────────────┐
   │ data/raw/<source>/<YYYY>/<MM>/<DD>/                          │
   │  - response_<stamp>.{json,xml}    (verbatim, key redacted)   │
   │  - parsed_<stamp>__<sha>.parquet  (tidy long-format)         │
   │  - metadata_<stamp>__<sha>.json   (collected_at + request log)│
   └────────┬────────────────────────────────────────────────────┘
            │ src/features/build_monthly.py
            ▼
   ┌─────────────────────────────────────────────────────────────┐
   │ data/processed/                                              │
   │  - smp_monthly_<area>_h1m.parquet     (모델 입력)             │
   │  - smp_monthly_<area>_h1m.sideinfo.json (dedup log 등)        │
   └────────┬────────────────────────────────────────────────────┘
            │
   ┌────────┴──────────────────────────────────────────────────────────┐
   │ 학습 + 평가                                                          │
   │  src/pipelines/                                                    │
   │   - train.py            (단일 70/15/15 split)                      │
   │   - walk_forward.py     (시점별 refit)                              │
   │   - mlops_smoke_test.py (v1..v5 chronological cutoff + rolling)    │
   │   - train_lng_forecast.py (LNG 전용 PDF-guided 모델)                │
   └────────┬─────────────────────────┬──────────────────────────┬───────┘
            │                         │                          │
            ▼                         ▼                          ▼
   ┌──────────────────┐  ┌──────────────────────────┐  ┌──────────────┐
   │ outputs/models/  │  │ MLflow (Postgres + S3-   │  │ outputs/     │
   │  <model>/        │  │ compatible artifact dir) │  │ metrics/     │
   │   predictions    │  │  - 65 detail runs        │  │  comparison  │
   │   model.pkl      │  │  - 7 per-model summary   │  │  .csv        │
   └──────────────────┘  │  - 1 overview run        │  └──────────────┘
                         └──────────────────────────┘

   ┌─────────────────────────── 표시 / 모니터링 레이어 ──────────────────┐
   │  dashboard.py (Streamlit, :8501)  — read-only 뷰어 (9 pages)        │
   │  api/main.py  (FastAPI, :8000)    — JSON API, /api/health 등         │
   │  web/        (Next.js, :3000)     — SPA 프론트엔드                   │
   │  MLflow UI    (:5000)             — 학습 곡선 비교 (Compare-runs)    │
   └─────────────────────────────────────────────────────────────────────┘

   ┌─────────────────────────── 알림 / 자동화 ───────────────────────────┐
   │  cron */5 → scripts/health_check_wrapper.sh → /api/health 점검 →     │
   │            Discord 웹훅 (정상/실패 알림)                              │
   │  mlops_smoke_test.py → Discord (start/per-version/done/error)        │
   └─────────────────────────────────────────────────────────────────────┘
```

---

## 3. 디렉토리 구조

```
collect_price_variable/
├── api/                  # FastAPI 백엔드 (:8000) — 모델 메타데이터 / 예측 API
│   └── main.py
├── web/                  # Next.js 프론트엔드 (:3000)
│   └── app/_components/
├── src/
│   ├── collectors/       # 데이터 수집 — API + 파일 양쪽
│   │   ├── base.py            # BaseCollector + FetchResult + CollectorRun
│   │   ├── file_loader.py     # BaseFileLoader (drop-inbox 패턴)
│   │   ├── kpx_smp.py         # KPX SMP API (승인 대기 중 — 파일 백업 경로 운영)
│   │   ├── kpx_files.py       # KPX file loaders (SMP/설비/거래/JKM/oil/FX)
│   │   ├── eia_steo.py        # EIA STEO (BREPUUS/WTIPUUS/NGHHMCF)
│   │   ├── kma_mid_temperature.py    # KMA 중기예보 (regId × tmFc)
│   │   ├── kma_village_fcst.py       # KMA 단기예보 (nx,ny × baseDate/Time)
│   │   └── quarantine.py      # 파일명-내용 mismatch 격리
│   ├── features/
│   │   ├── build_monthly.py   # 월별 panel 빌더 (SMP + 모든 leading indicators)
│   │   └── lag_features.py    # lag/rolling helper
│   ├── models/           # 모델 클래스 (모두 fit/predict 인터페이스 동일)
│   │   ├── naive.py           # PersistenceMonthly, NaiveLag1m, SeasonalNaiveLag12m
│   │   ├── ridge_model.py     # sklearn Ridge wrapper (SimpleImputer 포함)
│   │   ├── ar_monthly.py      # MonthlyARRidge (lag-only Ridge)
│   │   ├── delta_models.py    # Δ-target wrappers (DeltaRidge/DeltaLightGBM/…)
│   │   ├── lightgbm_model.py  # LightGBM + eval_history 캡처
│   │   ├── iterative_models.py # MLP/XGBoost/Torch MLP/Torch LSTM (epoch curve)
│   │   ├── lng_forecast/      # PDF-guided LNG 전용 패키지 (round 7)
│   │   ├── metrics.py         # RMSE/MAE/MAPE/R²/directional_accuracy
│   │   └── registry.py        # baseline vs trainable 분류
│   ├── pipelines/        # CLI 진입점 (typer)
│   │   ├── load_files.py             # drop inbox → raw partitions
│   │   ├── collect_snapshot.py       # 여러 collector 일괄 실행
│   │   ├── build_monthly_features.py # raw → processed
│   │   ├── train.py / walk_forward.py / evaluate.py
│   │   ├── feature_group_ablation.py
│   │   ├── mlops_smoke_test.py       # v1..v5 retraining + MLflow + Discord
│   │   └── train_lng_forecast.py
│   ├── tracking/         # MLflow 헬퍼 + 모델 레지스트리 기록
│   ├── utils/
│   │   ├── io.py              # 파일 경로 규약 (raw_response_dir 등)
│   │   ├── redact.py          # secret_variants + redact_secrets_in_text
│   │   ├── discord.py         # 알림 헬퍼 (silent no-op when unset)
│   │   ├── time.py / logging.py / storage.py
│   ├── validation/       # data-quality 검증 helper
│   └── config/
│       ├── settings.py        # pydantic-settings (env 로딩)
│       └── sources.yaml       # 모든 source의 frequency/unit/license 메타
├── config/
│   └── mlops_smoke_test.yaml  # v1..v5 cutoff + 모델 리스트
├── scripts/
│   ├── health_check.py            # cron 5분 주기 헬스 체크
│   ├── health_check_wrapper.sh    # .env 로드 + python 호출
│   └── plot_smp_dataset_distributions.py
├── tests/                # pytest — 178+ canaries (leakage, redaction, etc.)
├── data/                 # ↓ 모두 gitignored
│   ├── raw/<source>/<Y>/<M>/<D>/    # 수집기 출력 파티션
│   ├── raw/manual_or_filedata/<src>/ # 드롭 인박스
│   ├── interim/                      # 중간 캐시
│   └── processed/                    # 모델 입력 parquet
├── outputs/              # 일부 gitignored — 보고서/스냅샷만 트래킹
│   ├── lng_forecast/          # tracked (CSV만 작음)
│   ├── eda/                   # tracked (분포 PNG)
│   ├── models/ metrics/ figures/ predictions/  # ↓ ignored
│   ├── mlops_smoke_test/      # ↓ ignored
│   └── reports/               # ↓ ignored
├── docs/
│   ├── architecture_kr.md     # ← 본 문서
│   ├── project_report_kr.md   # 5/25 방법론 보고서
│   ├── slides_mlops_kr.md
│   ├── data_catalog.md
│   ├── external_models_inventory.md  # solar/ solar_beam/ 참조
│   └── figures/baseline_*/    # 5번 스냅샷 (tracked)
├── dashboard.py          # Streamlit (:8501) — 9 pages
├── logs/                 # ↓ ignored
├── artifacts/mlflow/     # ↓ ignored (MLflow 아티팩트 루트)
├── mlflow.db             # ↓ ignored (SQLite backend)
├── docker-compose.yml    # mlflow + postgres
├── .streamlit/config.toml  # 외부 접속 허용 설정
├── .env                  # ↓ ignored — 모든 시크릿
├── .env.example          # placeholder만
└── pyproject.toml / requirements.txt
```

**gitignored** 핵심: 모든 `data/raw/**/*` 페이로드 (재생성 가능), `outputs/{models,metrics,figures}` (학습 결과),
`mlflow.db` (운영 DB), `additional_data_{1,2,3}/` (외부 dump), 루트 `*.pdf/*.docx/*.xlsx` (LLM 참조자료),
`solar/` `solar_beam/` (별도 sister projects).

---

## 4. 데이터 파이프라인

### 4.1 수집 단계

두 종류의 진입점이 동일한 디스크 레이아웃을 공유:

| 진입점 | 입력 | 코드 |
|--------|------|------|
| API collector | HTTP GET → response | `BaseCollector.collect()` → `fetch_one()` |
| File loader | drop-inbox CSV/XLSX | `BaseFileLoader.load_one()` → `parse_file()` |

공통 출력 (`data/raw/<source>/<Y>/<M>/<D>/`):
- `response_<stamp>.{json,xml}` — **verbatim 원본** (serviceKey 등 시크릿은 redacted)
- `parsed_<stamp>__<sha>.parquet` — 정제된 long-format DataFrame
- `metadata_<stamp>__<sha>.json` — `collected_at` (UTC-naive), 요청 로그, 스키마 버전

핵심 invariant:
- `collected_at`은 **항상 UTC-naive** (외부 contract, round 4)
- 같은 wall-second 충돌 방지를 위해 microsecond + sha256 디스앰비귀에이터
  (`name_suffix` 매개변수, round 2)
- API 에러 응답도 verbatim 보존 (`persist_partial_error_envelope`, round 9)

### 4.2 변환 단계

`src/features/build_monthly.py:build_smp_monthly_features()`:

1. 각 raw source에서 최신 snapshot 선택
2. 동일 (period_month, area) 중복은 `(source_priority, collected_at, sha256, parsed_path)` 순으로 dedup
   → dedup 결정 로그는 `.sideinfo.json`에 남김
3. SMP + 모든 leading indicator를 `period_month`로 join
4. `_MAX_LAG_MONTHS=2` 만큼 grid extension (newest usable lag 노출)
5. 타깃 컬럼 `target_smp_t_plus_h_months` 생성 (M → M+1)

**leak-safety 원칙:**
- JKM monthly: M+1 중순 발표 → M 시점에 미관측 → `lag_1m` 이상만 사용
- LNG `pct_change`: `fill_method=None` 명시 (pandas 2.x pad가 NaN을 0-change로 fabricate 방지)
- `smp_t_observed`: **합법** — M 시점에 이미 관측됨 (delta-target reconstruction의 근간)

### 4.3 학습/평가 단계

세 가지 평가 모드 (모두 chronological):

| 모드 | 코드 | 분할 방식 |
|------|------|----------|
| 단일 holdout (70/15/15) | `pipelines/train.py:_split_chronological` | 전체를 시간순 정렬 후 비율 컷 |
| Walk-forward | `pipelines/walk_forward.py` | 매 시점 expanding window refit + 1-step ahead |
| v1..v5 retraining | `pipelines/mlops_smoke_test.py` | YAML에 명시한 cutoff별 split + v5만 12-fold rolling |
| TimeSeriesSplit + embargo | `models/lng_forecast/split.py:ts_folds` | PDF guide §5 (LNG 전용) |

---

## 5. 수집기 카탈로그

| Source ID | 진입점 | 빈도 | 단위 | Round 도입 |
|-----------|--------|------|------|-----------|
| `kpx_smp_monthly_*` | file loader | 월 | KRW/kWh | 1 |
| `kpx_settlement_monthly` | file loader | 월 | KRW/kWh | 1 |
| `kpx_capacity_*_monthly` | file loader | 월 | MW / share | 1 |
| `kpx_transaction_*_*` | file loader | 시 / 일 / 월 | MWh / KRW | 1 |
| `kpx_rec_weekly` | file loader | 주 | KRW | 2 |
| `lng_price_monthly_file` | file loader | 월 | USD/MMBtu | 6 |
| `oil_price_monthly_file` | file loader | 월 | USD/bbl | 6 |
| `fx_usd_krw_monthly_file` | file loader | 월 | KRW/USD | 6 |
| `jkm_lng_daily_history_file` | file loader | 일 | USD/MMBtu | 8 |
| `jkm_lng_futures_curve_file` | file loader | quote-as-of | USD/MMBtu | 8 |
| `raw_weather_monthly_db_file` | file loader | 월 (집계) | mixed | 8 |
| `kma_grid_xy_mapping_file` | file loader | static | nx/ny | 9 |
| `eia_steo` | API | 월 | mixed | 8 |
| `kma_mid_temperature` | API | 2회/일 | °C (lead 3~10일) | 8 |
| `kma_village_fcst` | API | 8회/일 | TMP/SKY/PTY/POP/PCP/REH/WSD | 9 |

자세한 메타: [`src/config/sources.yaml`](../src/config/sources.yaml)와 [`docs/data_catalog.md`](data_catalog.md).

**API collector 공통 패턴** (data.go.kr 서비스 대상):

- 키 두 형태 처리: RAW (`params=`) vs URL-ENCODED (수동 URL build) — 이중 인코딩 방지 (round 8)
- 모든 에러 응답도 verbatim 저장 (HTTP 4xx / 5xx / HTML SERVICE_KEY 에러 페이지 / `resultCode != "00"`)
- exception 메시지에서 serviceKey 자동 마스킹 (`raise … from None`로 cause chain 끊기)
- tenacity 재시도 → 매 시도가 microsecond stamp + sha 다이스밤귀에이터로 분리 저장

---

## 6. 모델 카탈로그

현재 13개 모델 (`src/pipelines/mlops_smoke_test.py:MODEL_FACTORIES`):

| 모델 | 학습 방식 | epoch curve | 코드 |
|------|----------|:----------:|------|
| `persistence_monthly` | 학습 없음 (lookup) | – | `naive.py:PersistenceMonthly` |
| `naive_lag_1m` | 학습 없음 | – | `naive.py:NaiveLag1m` |
| `seasonal_naive_lag_12m` | 학습 없음 | – | `naive.py:SeasonalNaiveLag12m` |
| `ridge` | sklearn closed-form | – | `ridge_model.py` |
| `monthly_ar_ridge` | sklearn closed-form (lag-only) | – | `ar_monthly.py` |
| `delta_ridge` | Δ-target Ridge | – | `delta_models.py:DeltaRidge` |
| `delta_ar_ridge` | Δ-target AR Ridge | – | `delta_models.py:DeltaARRidge` |
| **`lightgbm`** | boosting | ✅ ~200 iter | `lightgbm_model.py` |
| **`delta_lightgbm`** | Δ + boosting | ✅ ~80-200 iter | `delta_models.py:DeltaLightGBM` |
| **`mlp`** | sklearn MLPRegressor | ✅ ~500 iter | `iterative_models.py:MLPMonthlyModel` |
| **`xgboost`** | boosting | ✅ ~300 iter | `iterative_models.py:XGBoostMonthlyModel` |
| **`torch_mlp`** | PyTorch FFN | ✅ 300 epoch | `iterative_models.py:TorchMLPModel` |
| **`torch_lstm`** | PyTorch LSTM (12-step seq) | ✅ 300 epoch | `iterative_models.py:TorchLSTMModel` |

**공통 fit 인터페이스:** `fit(X, y, *, X_valid=None, y_valid=None)` — 모든 iterative 모델은
`X_valid` 받으면 `eval_history` 채움. `_extract_learning_curve()`가 flat dict로 변환
(`rmse_train`, `rmse_valid`, `r2_valid`) → MLflow에 step별 log.

**파생 클래스 (LNG 전용 — `src/models/lng_forecast/`):**
LightGBMForecaster + MLPForecaster + NaiveLagForecaster + TimeSeriesSplit+embargo 평가기.

---

## 7. 평가 프레임워크

### 7.1 메트릭 (`src/models/metrics.py`)

표준 회귀: RMSE / MAE / MAPE / sMAPE / R².
영업 의미 메트릭: `directional_accuracy` (전월대비 방향 맞추기), `peak_precision/recall/f1` (상위 분위 적중).

### 7.2 MLOps smoke test — v1..v5

[`config/mlops_smoke_test.yaml`](../config/mlops_smoke_test.yaml):

| version | data cutoff | test 기간 | 모드 |
|---------|------------|-----------|------|
| v1 | 2021-12 | 2022-01~12 | fixed_holdout |
| v2 | 2022-12 | 2023-01~12 | fixed_holdout |
| v3 | 2023-12 | 2024-01~12 | fixed_holdout |
| v4 | 2024-12 | 2025-01~08 | fixed_holdout |
| v5 | 2025-08 | rolling 12-fold | latest_rolling_validation |

각 (version, model) → MLflow 1개 run (총 13 × 5 = 65). 추가로:
- 7개 `summary_<model>` per-model run (v1..v5를 step=1..5 stepped metric으로 기록)
- 1개 `summary_overview_v1_v5` run (rendered overlay PNG/HTML artifact)

**Discord 통합** (`src/utils/discord.py`): start / per-version / done / error 4지점에 알림.

### 7.3 추가 평가 pipeline

- `src/pipelines/walk_forward.py` — monthly walk-forward (150 refits)
- `src/pipelines/feature_group_ablation.py` — group 단위 leave-one-out 영향도
- `src/pipelines/save_baseline_plots.py` — `docs/figures/baseline_<tag>/`에 스냅샷 (`--docs-copy` 플래그로 트래킹)

---

## 8. 인터페이스 레이어

| 서비스 | 포트 | 진입점 | 역할 |
|--------|------|--------|------|
| Streamlit dashboard | 8501 | `dashboard.py` | 9 페이지 read-only 뷰어 (모델/예측/feature/DQ/walk-forward/LNG/collectors) |
| FastAPI backend | 8000 (127.0.0.1) | `api/main.py` | `/api/health`, `/api/models/*`, `/api/forecast/*`, `/api/retrain` |
| Next.js frontend | 3000 | `web/app/page.tsx` | SPA — hero price + model selection + retrain |
| MLflow tracking UI | 5000 | `docker-compose up` | Compare-runs / Metric history / Artifacts |

**Streamlit 외부 접속 설정** (`.streamlit/config.toml`):
`server.address = "0.0.0.0"` + `enableCORS = false` + `enableXsrfProtection = false`.
같은 LAN에서 `http://192.168.50.100:8501` 접근 가능. 보안: read-only이지만 메트릭/예측이 전부 노출되니
신뢰 네트워크 또는 터널(tailscale/cloudflared) 뒤에서만.

---

## 9. 운영 / MLOps

### 9.1 MLflow

- **백엔드:** `docker-compose.yml`에 postgres 포함 (운영용)
- **로컬 폴백:** `sqlite:///mlflow.db` (gitignored, ~2 MB)
- **artifact root:** `artifacts/mlflow/` (gitignored, 수십 MB 가능)
- **Experiment:** `kpx-smp-monthly` — 모든 smoke test run 집결

### 9.2 헬스 체크 cron

```cron
*/5 * * * * /home/probius/collect_price_variable/scripts/health_check_wrapper.sh \
            >> /home/probius/collect_price_variable/logs/health_check.log 2>&1
```

- 대상: `http://127.0.0.1:8000/api/health` (FastAPI 라이브니스)
- 4가지 healthy shape 수용 (`ok:true` / `healthy:true` / `status: "ok"` / `health: "ok"`)
- 알림: 성공/실패 모두 Discord. 알림 자체 실패 시에도 cron 종료 코드는 endpoint 상태만 반영
- User-Agent: `kpx-health-check/1.0 (+cron)` (Python-urllib 기본은 Discord가 403 거부)

### 9.3 Discord 통합 두 채널

| 용도 | env 변수 | 코드 |
|------|---------|------|
| mlops_smoke_test run-status | `MLOPS_DISCORD_WEBHOOK_URL` | `src/utils/discord.py` |
| /api/health 5분 모니터링 | `HEALTH_CHECK_DISCORD_WEBHOOK_URL` | `scripts/health_check.py` |

둘 다 unset → 통합 silent no-op. 코드/로그 어디에도 URL 노출 없음.

### 9.4 Docker (선택)

`docker-compose up -d` → postgres + mlflow server. 로컬 개발은 `mlflow.db` SQLite로도 가능.

---

## 10. 보안 / 시크릿 정책

**원칙:** 시크릿은 절대 git에 들어가지 않는다. `.env`만 신뢰원으로, `.env.example`은 빈 placeholder.

| 시크릿 종류 | 저장 위치 | 보호 메커니즘 |
|------------|----------|--------------|
| KPX API key (raw + encoded) | `.env` | `.gitignore` line 1 |
| KMA serviceKey | `.env` | 동상 |
| EIA STEO API key | `.env` | 동상 |
| MLflow Discord webhook | `.env` | 동상 |
| Health-check Discord webhook | `.env` | 동상 |
| Postgres password | `.env` | 동상 |
| solar_beam DB DSN | `.env` (별도) | 동상 + docs는 redacted DSN만 기록 |

**런타임 마스킹** (`src/utils/redact.py`):
- `secret_variants(value)` — raw, URL-encoded, URL-decoded 3가지 형태 동시 생성
- `redact_secrets_in_text(text, secrets)` — 페이로드/exception/URL에서 일괄 마스킹
- 적용 지점: collector raw envelope 저장 시, dashboard envelope 미리보기, Discord 메시지, base.py kwarg 스크럽

**대시보드 표시 시 redaction** (`dashboard.py:_redact_preview`): 23 카나리로 핀.
JWT, AWS SigV4, OAuth1 quoted, HMAC, JSON-형 Authorization, x-api-key 헤더, cookie/session/sso 토큰까지 커버.

---

## 11. 개발 워크플로우

### 11.1 새 데이터 소스 추가

1. **소스 타입 결정:** API인가 (BaseCollector) 파일인가 (BaseFileLoader)?
2. **수집기 작성:** `src/collectors/<source>.py`
3. **메타 등록:** `src/config/sources.yaml`에 frequency/unit/license/verified_columns
4. **feature 통합:** `src/features/build_monthly.py`에 reader + lag 처리 추가
5. **테스트 추가:** `tests/test_round<N>_<source>.py` — parser/leakage/redaction canary
6. **드롭 인박스 위치:** `data/raw/manual_or_filedata/<source_id>/`

### 11.2 새 모델 추가

1. **클래스 작성:** `src/models/`. 인터페이스:
   ```python
   class MyModel:
       name = "my_model"
       def fit(self, X, y, *, X_valid=None, y_valid=None) -> "MyModel": ...
       def predict(self, X) -> np.ndarray | pd.Series: ...
       # (iterative이면) self.eval_history = {"train": {"rmse": [...]}, ...}
   ```
2. **factory 등록:** `src/pipelines/mlops_smoke_test.py:MODEL_FACTORIES`
3. **config에 노출:** `config/mlops_smoke_test.yaml`에 `- name: my_model` 추가
4. **registry 분류:** `src/models/registry.py`에서 baseline/trainable 결정
5. **테스트:** `tests/test_models.py`에 fit/predict 라운드트립 + (해당시) `_extract_learning_curve` 카나리

### 11.3 일반 CLI 명령

```bash
# 수집
python -m src.pipelines.load_files                     # drop inbox 흡수
python -m src.pipelines.collect_snapshot --source eia_steo  # API 1회

# feature
python -m src.pipelines.build_monthly_features --area mainland --horizon-months 1

# 학습/평가
python -m src.pipelines.train --model ridge
python -m src.pipelines.walk_forward main --features-path ... --model lightgbm
python -m src.pipelines.mlops_smoke_test --config config/mlops_smoke_test.yaml --log-to-mlflow

# DQ + 보고서
python -m src.pipelines.dq_report
python -m src.pipelines.save_baseline_plots --tag my_run --docs-copy

# UI
streamlit run dashboard.py             # :8501
uvicorn api.main:app --reload --port 8000   # :8000
cd web && npm run dev                       # :3000
docker compose up -d                        # mlflow :5000
```

### 11.4 테스트

```bash
.venv/bin/python -m pytest -q          # 178+ tests, ~30 s
.venv/bin/python -m pytest tests/test_round8_jkm_eia_kma.py -q
```

핵심 회귀 가드 (leakage / redaction / dedup / no-overwrite):
`test_round{4,5,6,7,8,9}_*.py`, `test_dashboard_redaction.py`, `test_mlops_smoke_test.py`.

---

## 12. 의존성

**Python 인터프리터:** `requires-python = ">=3.10"` (pyproject.toml).
개발 환경 런타임은 3.14.4 (`.venv/`). 실배포는 Docker 이미지가 핀하는 것을 따른다 —
3.10/3.11/3.12 어느 쪽으로 빌드해도 호환되어야 한다.

**Python 패키지 핀** (`requirements.txt` + `pyproject.toml` 동기, 2026-05-26 기준):

| 카테고리 | 패키지 | 핀 | 비고 |
|----------|--------|-----|------|
| 데이터 | pandas | `>=2.1` | 2.x 시리즈 (round 6 `pct_change(fill_method=None)` 의존) |
| | numpy | `>=1.26` | |
| | pyarrow | (transitively pulled by pandas) | parquet I/O |
| 모델 — 클래식 | scikit-learn | `>=1.4` | MLPRegressor, Ridge, SimpleImputer, StandardScaler |
| 모델 — boosting | lightgbm | `>=4.3` | `record_evaluation` 콜백 사용 |
| | xgboost | `>=3.0,<4.0` | round 9 `iterative_models.py` 필수 |
| 모델 — neural | torch | `>=2.5,<3.0` | CPU 사용 (`device="cpu"`); 기본 PyPI는 CUDA wheels ~2 GB 동반 |
| 추적/저장 | mlflow | `>=3.0,<4.0` | UI Compare-runs Chart view 의존 |
| | duckdb | `>=0.10` | |
| | psycopg2-binary | `>=2.9` | MLflow postgres backend store |
| API/UI | fastapi | `>=0.110` | `api/main.py` |
| | uvicorn | `>=0.27` | dev server `--reload` |
| | streamlit | `>=1.57,<2.0` | `width="stretch"` 일관성 |
| | plotly | `>=6.0,<7.0` | dashboard overlay 차트 |
| | matplotlib | `>=3.8,<4.0` | snapshot PNG + MLflow overlay artifact |
| 수집 | requests | `>=2.31` | tenacity 재시도와 함께 |
| | tenacity | `>=8.2` | API collector retry |
| | xmltodict | `>=0.13` | KPX XML 파싱 |
| | **openpyxl** | `>=3.1` | pandas `read_excel(.xlsx)` 엔진 — KMA 그리드 + KPX HOME loaders가 필수 의존. pandas는 *optional*로 분류하지만 우리 환경엔 load-bearing |
| | (lxml) | (transitively) | xml loader |
| 설정 | pydantic | `>=2.6` | |
| | pydantic-settings | `>=2.2` | `.env` 로딩 |
| | typer | `>=0.12` | CLI 진입점 |
| | PyYAML | `>=6.0` | `mlops_smoke_test.yaml` 파싱 |
| | python-dotenv | `>=1.0` | |
| | rich | `>=13.7` | typer 출력 |

**Optional 그룹** (pyproject `[project.optional-dependencies]`):
- `dev` — pytest, pytest-cov
- `postgres` — psycopg[binary] (psycopg3, MLflow 외 용도)
- `extra-models` — statsmodels (홀가분한 시계열 baseline 추가용)
- `polars` — polars (실험적 대용량 처리)

**Node (web/):** Next.js 15 / React 19 / Tailwind CSS.

**시스템:** docker (선택), cron (헬스 체크), bash, WSL2/Linux.

**외부 sister projects** (gitignored, 참조용 — [`external_models_inventory.md`](external_models_inventory.md)):
- `solar/` — LNG 가격 예측 외부 모델
- `solar_beam/` — 태양광 발전량 예측 + `raw_weather` 시간별 DB

> **버전 동기화 책임:** 새 dep 추가 시 `requirements.txt` + `pyproject.toml`을 함께 수정.
> 본 표는 두 파일에서 파생되며, drift가 발생하면 (예: 설치된 mlflow 3.x ↔ 핀 `<3.0`) 본 표 갱신이 늦어진 것이 아니라 핀 자체 갱신이 늦은 것이 원인. 최신 설치 버전 확인:
> ```bash
> .venv/bin/pip list --format=freeze | grep -iE 'mlflow|xgboost|torch|sklearn|pandas'
> ```

---

**마지막 업데이트:** 2026-05-26. 시스템 변경 시 본 문서도 동기화 (특히 [3. 디렉토리 구조](#3-디렉토리-구조)와 [6. 모델 카탈로그](#6-모델-카탈로그)).
