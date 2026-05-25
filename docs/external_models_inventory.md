# 외부 예측 모델 인벤토리 (solar/, solar_beam/)

본 프로젝트(`kpx-price-forecast`)에 통합 대기 중인 두 외부 모델의 파악
결과. 향후 LNG 모델 결합 시 즉시 활용할 수 있도록 핵심 인터페이스만
요약했다.

---

## 1. `solar/` — LNG 가격 예측 모델 + SMP 예측 비교

### 1.1 디렉터리 구조

```
solar/
├── .env                                       # (gitignored) DATABASE_URL + DATA_API_KEY
├── lng_model.py                               # Naive baseline (전월값 그대로) — LNG 1단계
├── smp_model.py                               # LightGBM SMP 예측 (LNG/유가/환율/기온 features)
├── benchmark.py                               # LightGBM/XGBoost/CatBoost/Stacking 벤치마크
├── catboost_info/                             # CatBoost 학습 로그
├── regions.txt                                # 17개 시도명
├── PNGASJPUSDM.xlsx                           # FRED LNG (JKM) Japan/Korea spot price
├── POILDUBUSDM (1).csv                        # FRED 두바이 유가 (195행, 2010-01~)
├── DEXKOUS (1).csv                            # FRED USD/KRW 환율 (4270행 일별, 2010-01~)
└── HOME_*.csv                                 # KPX SMP 파일 (참조용)
```

### 1.2 데이터 소스 (PostgreSQL `solar_beam` DB)

| 테이블 | 컬럼 | 빈도 | 비고 |
|---|---|---|---|
| `lng_price` | month, lng_price_usd | 월 | USD/MMBtu (JKM 기반) |
| `oil_price` | month, oil_price_usd | 월 | 두바이 유가 |
| `exchange_rate` | date, usd_krw | 일 | (모델에서는 month로 별칭) |
| `smp` | datetime, smp_land | 시간 | smp_land > 0 필터 후 월평균 |
| `raw_weather` | datetime, temperature, ... | 시간 | 전국 평균 기온 사용 |

### 1.3 LNG 모델 (`lng_model.py`)

**현재 구현:** 극단적으로 단순한 naive baseline:
```python
pred = df['lng_price_usd'].shift(1)   # 전월값 그대로
```

- Target: 다음달 `lng_price_usd` (USD/MMBtu)
- Train/test 분할: `< 2024-01-01` vs `>= 2024-01-01`
- 메트릭: MAE, MAPE

→ 본 프로젝트 SMP 모델에 통합 시: **LNG 예측값 1개월 lead**를 leading
exogenous로 주입할 수 있는 후보. 단, 현재 baseline은 persistence 수준이므로
실제로 lead 정보를 줄 수 있는지는 결과로 확인 필요.

### 1.4 SMP 모델 (`smp_model.py`)

**더 발전된 모델**: LightGBM + 20 features

**Features (20개):**
```
lng_price_usd, lng_lag{1,2,3}, lng_ma3       — LNG 현재 + 과거
oil_price_usd, oil_lag1, usd_krw, fx_lag1    — 유가 + 환율
month_sin, month_cos, quarter                — 캘린더
smp_lag{1,2,3}, smp_ma3                      — SMP 자기회귀
lng_chg1                                     — LNG 변화율
temp_avg, temp_lag1, temp_extreme            — 기온 + 극단(>25 or <3) 플래그
```

**핵심 차이점** (vs 본 프로젝트):
1. `lng_price_usd` 현재값을 **직접 feature로 사용** → 본 프로젝트는
   미보유. 결합 후 가장 큰 신호원 후보
2. `temp_avg` 전국 평균 기온 사용 → 본 프로젝트는 미보유
3. `usd_krw` 환율 사용 → 본 프로젝트는 ECOS placeholder만 있음
4. CV: `TimeSeriesSplit(n_splits=3)` 사용

**하이퍼파라미터 (small-data):**
```python
max_depth=4, num_leaves=15, min_child_samples=3, subsample=0.8
```
n_estimators ∈ {100, 200, 300}, learning_rate ∈ {0.03, 0.05, 0.1} grid search

### 1.5 Benchmark 결과 (예상 — 실측은 DB 접속 시)

`benchmark.py`는 LightGBM/XGBoost/CatBoost/Stacking 4개를 LNG와 SMP 양쪽에 적용.
- LNG는 `use_log=True` (log1p 변환)
- SMP는 원래 단위

---

## 2. `solar_beam/` — 태양광 발전량 예측 모델

### 2.1 디렉터리 구조

```
solar_beam/
├── .env, requirements.txt
├── DATACOLLECT.md                             # 데이터 수집 설계
├── SOLAR_BEAM_MODEL.md                        # 모델 학습 문서 (가장 자세함)
├── 태양광.md                                  # 도메인 노트
├── main.py                                    # 진입점 (수집)
├── train_model.py                             # LightGBM 학습 (310줄)
├── tune_model.py                              # GridSearch (729 조합)
├── shap_analysis.py                           # SHAP 분석
├── test.py
├── data/
│   ├── features.csv                           # 701,200행 학습 데이터
│   ├── kpx/                                   # KPX raw 파일
│   └── training_progress.json
├── models/
│   └── 20260525_1818_daytime/
│       ├── lgbm_daytime.{pkl,txt}             # 학습된 모델
│       ├── capacity_daytime.{csv,pkl}         # 연도×지역 용량 지수
│       └── capacity_trend_daytime.csv         # 미래 외삽용 trend
└── pipeline/
    ├── orchestrator.py                        # 메인 워크플로우 (25KB)
    ├── config.py
    ├── regions.py                             # 17개 시도 매핑
    ├── collectors/                            # KMA/KIER/NASA POWER API
    ├── features/
    └── storage/
```

### 2.2 모델 핵심 사양

| 항목 | 값 |
|---|---|
| 타깃 | `solar_power` (MWh, 시간 × 지역 단위) |
| 데이터 | 2020-01 ~ 2024-12, 16개 시도 (세종 제외) |
| 행 수 | 375,327 (주간 필터 후, `solar_altitude > 0`) |
| 분할 | train 2020-2022, val 2023, test 2024 |
| 모델 | 단일 LightGBM, region을 `categorical_feature`로 처리 |
| Features | 21개 (기상 7 + 천문 4 + 패널 1 + 캘린더 6 + region 1) |

### 2.3 성능 (GridSearch 최적, `lgbm_best`)

| 분할 | MAE (MWh) | RMSE | R² |
|---|---|---|---|
| val (2023) | — | — | 0.9265 |
| **test (2024)** | **23.58** | **59.31** | **0.9438** |

### 2.4 핵심 설계 결정

| 결정 | 이유 |
|---|---|
| **용량 정규화** (capacity index) | 태양광 설비가 매년 증가 (2020→2024 1.5~3배) → 정규화 안 하면 train/test 분포 차이로 R²=0.62. 정규화 후 0.94 |
| **단일 모델 + region categorical** | 지역 16개는 분리 학습보다 단일 모델 + categorical이 cross-region 패턴 학습 가능 |
| **주간 필터** (`solar_altitude > 0`) | 야간 발전량=0 행이 학습 노이즈 |
| **시간 분할 (k-fold X)** | 시계열 누수 방지 — train 2020-2022, val 2023, test 2024 고정 |
| **irradiance NaN → clear_sky_ghi fallback** | pvlib 이론값으로 결측 보완 |

### 2.5 SHAP 결과 (가장 중요한 feature)

| 순위 | Feature | mean\|SHAP\| |
|---|---|---|
| 1 | `irradiance` | **0.469** (80% 차지) |
| 2 | `cloud_cover` | 0.063 |
| 3 | `region` | 0.052 |
| 4 | `humidity` | 0.028 |
| 5 | `clear_sky_index` | 0.028 |

---

## 3. 본 프로젝트와의 통합 시 고려사항

### 3.1 LNG 모델 통합 (1차 목표)

**가장 가치 있는 leading-indicator 후보:**

| 통합 방법 | 코드 위치 | 효과 |
|---|---|---|
| **a) `solar/lng_model.py` 예측값을 본 프로젝트 feature로 주입** | 새 source `lng_price_forecast_monthly` 추가 → `_exogenous_lag_1m` 적용 (또는 forecast이므로 lag 없이 사용 가능) | persistence 6.43 → 트레이너블 < 6.43 목표 |
| b) `solar/benchmark.py`의 LNG 4-모델(LGB/XGB/Cat/Stacking) 중 선택 | 별도 LNG forecaster 모듈로 import | LNG forecast의 정확도가 SMP 개선폭의 상한 |
| c) `solar_beam` capacity_trend를 이용한 신재생 비중 예측 | `capacity_yearly_*` 컬럼에 forecast 추가 | 부차적, SMP 영향 미미할 가능성 |

### 3.2 시간/스키마 정합성

- 본 프로젝트 `period_month` = month start timestamp (e.g., 2024-01-01)
- `solar/`의 `lng_price` DB 컬럼 `month` = same convention
- 단위: USD/MMBtu (변환 없음, SMP는 KRW/kWh) → ratio/log 변환 고려
- **DB 접속 필요**: PostgreSQL `solar_beam` (DSN은 `solar/.env`에 보관, 본 문서에서 redact)
  - 직접 접속 가능하면 SQL로 호출
  - 불가능하면 `lng_model.py` 결과를 CSV export 후 file loader 패턴 적용

### 3.3 Forecast contract 일치 확인

본 프로젝트의 forecast contract:
- `forecast_origin_month` = M
- `information_cutoff` = end-of-M
- `target_month` = M+1
- `horizon` = "1M"

`solar/lng_model.py`도 같은 1-step naive forecast → contract 일치.
다만 LNG forecast의 발표 시점이 SMP forecast의 information_cutoff 이전이어야 함
(통상 JKM 월별 데이터는 월 중순 발표 → 이전 month_end 정보로 그 다음달 예측)
**→ 한 달 더 lag 필요 가능성** 검증 필요.

### 3.4 통합 시 비교 가능한 baseline plot

이미 저장됨: [docs/figures/baseline_20260525/](figures/baseline_20260525/)
- Plot 02 (persistence lag zoom)의 빨간 면적이 LNG forecast 결합 후 줄어드는지가 핵심 검증 포인트
- `save_baseline_plots --tag baseline_post_lng_v1 --docs-copy` 로 동일 형식 snapshot 후 diff

---

## 4. 다음 단계 추천 순서

1. **DB 접속 가능성 확인** — `devine.my:5432/solar_beam` 접근 가능 여부
2. **LNG forecast 추출** — 가능한 horizon (1m, 2m, 3m) 별로 `solar/benchmark.py`
   결과 비교 후 가장 정확한 모델 선택
3. **새 source `lng_price_forecast_monthly_file`** 정의 → `sources.yaml` + `kpx_files.py`에 loader 추가
4. **build_monthly_features에 `lng_forecast_<horizon>_usd_per_mmbtu`** 컬럼 추가
5. **각 모델 default에 새 feature 추가** + retrain
6. **`save_baseline_plots --tag baseline_post_lng_v1 --docs-copy`** 후 baseline_20260525와 diff

---

## 5. 참고: solar/ vs solar_beam/ 관계

| | solar/ | solar_beam/ |
|---|---|---|
| 타깃 | LNG/SMP 가격 (월) | 태양광 발전량 (시간×지역) |
| 데이터 | 5개 외부 source (FRED 등) + DB | NASA POWER + KMA + KPX |
| 모델 | LightGBM 위주, benchmark도 | LightGBM 단일 + GridSearch |
| DB | `solar_beam` PostgreSQL 공유 | (동일) |
| 본 프로젝트와의 관계 | **LNG 가격 forecast 제공** | 신재생 capacity forecast 보조 |

두 모델 모두 `solar_beam` 같은 DB를 사용 — solar는 가격 가공 측, solar_beam은
물량/물리량 측을 담당하는 자매 프로젝트로 보임.
