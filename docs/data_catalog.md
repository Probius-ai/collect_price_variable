# 데이터 카탈로그 — 한국 전력시장 가격 변수 (Round 2)

이 문서는 본 프로젝트가 현재 사용 중인 모든 파일 기반 데이터셋의 **데이터 전문가용 reference**다.
각 데이터셋에 대해 (1) 출처와 생성 메커니즘, (2) 스키마와 단위, (3) 실데이터에서 발견된 품질 이슈,
(4) 모델링에 쓸 때의 함정을 정리한다.

---

## 0. 한눈에 보기

| source_id | 빈도 | 단위 | 시간 범위 | 행 수 | 신뢰도 | priority |
|---|---|---|---|---|---|---|
| `kpx_smp_monthly_kepco_file` | 월 | KRW/kWh | 2015-02 ~ 2025-08 (127개월 × 3 areas) | 378 | 공식 KPX 다운로드 | **1** |
| `kpx_smp_monthly_home_avg_file` | 월 | KRW/kWh | 2001-04 ~ 2026-04 (301개월 × 3 areas) | 693 | EPSIS 웹 export | 2 |
| `kpx_settlement_monthly_file` | 월 | KRW/kWh | 2002-01 ~ 2026-04 (10 fuel types) | 2,842 | EPSIS, **수정정산** | 1 |
| `kpx_rec_weekly_file` | trading-day | KRW/REC, KRW | 2017-03-28 ~ 2023-03-30 (601 거래일) | 601 | 공공데이터포털 | 1 |
| `kpx_generation_yearly` *(quarantined)* | 연 | MWh (추정) | n/a | n/a | filename mismatch — 사용 금지 | n/a |

`source_priority`가 낮을수록(1이 최강) 동일 (period, area) 충돌 시 우선 선택된다.
실제로 두 SMP 소스가 겹치는 381쌍 모두 **KEPCO**(priority 1)가 채택된다.

---

## 1. `kpx_smp_monthly_kepco_file` — KPX 공식 월별 SMP

### 출처와 생성 메커니즘

- **출처**: 공공데이터포털 — *한국전력거래소_육지 제주 통합 월별 계통한계가격*.
- **갱신**: 매월 M+1월 초~중에 전월(M) 행이 추가되는 누적 스냅샷.
- **생성 방식**: KPX가 시간별 SMP(원/kWh)를 월별로 집계(가중평균). 가중치는 *시장
  운영규칙 제2장 4절*에 따라 *시간별 거래량*. 산술평균과는 약간 다르다.
- **인코딩**: cp949(한국어 Windows 표준). 헤더 1행, 데이터 본문 무사.

### 스키마

| vendor 컬럼 | 캐노니컬 | 단위 |
|---|---|---|
| `년도` | period_year | int |
| `월` | period_month_num | int 1..12 |
| `육지계통한계가격` | smp_mainland | KRW/kWh |
| `제주계통한계가격` | smp_jeju | KRW/kWh |
| `통합계통한계가격` | smp_integrated | KRW/kWh |

로더는 wide → long으로 melt해서 `(period_month, area, smp_krw_per_kwh)`의 long format으로 저장한다.

### 데이터 품질 이슈 (실데이터에서 발견)

1. **2015-06 단발 공란**: 세 area 모두 vendor가 literal `0.00`을 적어두었다. 5월=68.72, 7월=66.79
   사이의 1개월 vendor gap. 로더가 0을 NaN으로 강제 변환 후 drop. (3 rows dropped — DQ 노트
   `zero_coerced_rows: {integrated: 1, jeju: 1, mainland: 1}`)
2. **통합 SMP의 retroactive 산출**: KPX 공식 *통합 SMP* 정의는 **2024-02-01부터** 적용된다.
   이 파일은 2015-02부터 통합 컬럼에 값이 있는데, 이는 vendor가 *과거 데이터에 같은 공식을
   소급 적용해서 채워 넣은 것*으로 보인다. 즉 2024년 1월 이전의 통합 SMP는 *정책상 존재하지 않던
   가격*이다. EDA에는 써도 되지만, **"2024-02-01 이전에 시장 참여자가 알 수 있던 정보"로
   가정해서는 안 된다.**
3. **육지/제주 분리 추적 시작 시점**: 이 파일은 2015-02부터 시작한다. 그 이전 데이터를 원하면
   HOME export(소스 2)를 봐야 한다.

### 모델링 시 주의사항

- **육지와 제주를 절대 합치지 말 것**. 송전제약·발전믹스(제주는 재생에너지 비중↑, LNG 비중↑,
  출력제어 영향)·계통 특성이 다르다. 통합 SMP는 *별도의 정의*가 있는 *제3의 시리즈*다.
- **현재 값(period=M)을 (M+1) 타깃의 feature로 쓰는 건 합법**. KPX는 M월이 끝난 직후 다음달
  초에 M월 값을 공표하므로, M+1을 예측하는 M+1월 초 시점에 M월 값은 이미 알려져 있다.
  단, 이번 라운드의 `smp_lag_1m` baseline은 "row M의 lag_1m = SMP[M-1]"이라는 *벡터 시점 정의*를
  쓰기 때문에 직관과 다를 수 있다. 의미상 persistence baseline은 `smp_krw_per_kwh` 컬럼
  그대로 쓰는 게 더 강하다(실측: MAE 9.4 vs lag_1m baseline의 MAE 9.4 → 동일 효과).
- **통합 SMP를 학습 타깃으로 쓸 때**: 2024-02-01 이전 데이터를 train에 넣되, 정책 정의가
  바뀐 시점이라는 걸 인지하고 *structural break*로 처리할 것. CUSUM이나 단순 indicator
  변수(`is_post_integrated_smp_era`) 권장.

---

## 2. `kpx_smp_monthly_home_avg_file` — HOME(EPSIS) 가중평균 SMP

### 출처와 생성 메커니즘

- **출처**: EPSIS 웹사이트(`epsis.kpx.or.kr`)에서 차트 페이지의 "데이터 받기" 기능으로 export.
- **갱신**: 사용자가 다운로드할 때마다 *최신값까지의 누적 스냅샷*. KPX 공식 API가 아니라
  EPSIS의 *대시보드용 가공 데이터*. 그래서 source 1보다 신뢰도가 낮다고 본다(priority 2).
- **생성 방식**: KPX가 *수요 또는 거래량 가중평균*으로 산정한 가중평균 SMP. KEPCO 파일과
  "논리적으로 같은 메트릭"이지만 *가중치 정의가 미세하게 다를 수 있어* 동일 (period, area)에서
  값이 약간 다르게 나올 때가 있다. 두 소스를 합치는 dedup 로그에서 차이를 추적할 수 있다.
- **포맷**: `.csv`(cp949)와 `.xlsx` 두 형식이 동일 내용으로 제공된다.
- **인코딩**: cp949. **헤더 2행** (병합셀):

  ```
  row 0:  기간,  SMP,    ,    , BLMP
  row 1:      ,  육지, 제주, 통합,
  ```

  로더가 부모 행을 forward-fill하고 leaf 행은 literal 유지해서 `SMP|육지`, `SMP|제주`,
  `SMP|통합`, `BLMP|` 로 평탄화한다.

### 스키마

| flattened 컬럼 | 캐노니컬 | 단위 |
|---|---|---|
| `기간\|` | period (YYYY/MM string) → period_month | timestamp |
| `SMP\|육지` | smp_mainland | KRW/kWh |
| `SMP\|제주` | smp_jeju | KRW/kWh |
| `SMP\|통합` | smp_integrated | KRW/kWh |
| `BLMP\|` | blmp *(별도)* | KRW/kWh? |

### 데이터 품질 이슈

1. **2001-04 ~ 2009-12 mainland/jeju가 0**: 105 + 105 = 210행. KPX가 2010년 무렵까지
   육지/제주 분리 추적을 하지 않았던 것으로 추정된다. 이 시기 *통합* 컬럼만 실값이고
   육지/제주는 공란 marker(0)으로 채워졌다. **2001~2009년 데이터를 mainland 모델에
   넣으면 안 된다** — 모두 0이라 학습이 망가진다. 로더가 강제로 drop. (DQ 노트
   `zero_coerced_rows: {jeju: 105, mainland: 105}`)
2. **BLMP 컬럼의 의미**: 추정상 *Bilateral 시장 평균가*(상대거래/양자거래 평균). 최근
   대부분 0으로 나옴 → 양자거래가 거의 없는 상태이거나 vendor가 산정 중지. **이번
   라운드에서는 BLMP를 feature/target으로 쓰지 않는다.** 컬럼은 보존(audit), 사용은 보류.
3. **KEPCO 파일과의 미세 차이**: 동일 (period, area)에서 두 소스 값이 ±0.05 KRW/kWh
   정도 다른 경우가 있다(반올림 차이로 추정). priority 1(KEPCO)이 항상 채택되며,
   conflict는 `data/processed/smp_monthly_*_h1m.sideinfo.json::smp_priority_dedup_log`에
   전부 기록된다.

### 모델링 시 주의사항

- **2001~2009년 장기 시계열의 가치**: 다른 source에는 없는 긴 히스토리(25년). 다만 통합
  컬럼만 실값이라 *통합 모델*에서만 가치가 있다.
- **가중평균 SMP는 산술평균과 다르다**. 시간별 SMP를 산술평균하면 가중평균과 1% 내외로
  벌어진다. 이 데이터를 "월평균 SMP"라고 말할 때는 항상 *수요·거래량 가중*임을 명시해야 한다.
- **xlsx와 csv 중복 로드 방지**: 같은 데이터를 두 포맷으로 받아두면 같은 (period, area)
  쌍이 두 번 들어와도 build_monthly의 dedup 로직이 처리하지만, 로더 입장에서는 둘 다
  parsed_*.parquet에 쓴다. 디스크 낭비를 피하려면 하나만 drop 디렉터리에 두는 게 좋다.

---

## 3. `kpx_settlement_monthly_file` — 월별 연료원별 정산단가

### 출처와 생성 메커니즘

- **출처**: EPSIS *연료원별 정산단가* 페이지에서 export.
- **갱신**: 매월 M+1월 초.
- **결정적 특징 — 수정정산**: KPX 시장운영규칙상 정산은 *최초 정산 → 수정정산 → 연간
  전력시장통계 확정*의 3단계를 거친다. 따라서 같은 (period, fuel_type)을 *다른 시점에
  다시 받으면 값이 변할 수 있다*. 이 프로젝트가 **collected_at + sha256**를 raw + parsed +
  metadata 전반에 기록하는 핵심 이유다.
- **정산단가의 KPX 공식 정의** (EPSIS 페이지 인용):

  > "정산단가 = 전력거래대금 / 발전량. 정산단가 산정 시 *RPS 의무이행비용정산금*과
  > *배출권거래비용정산금*은 전력거래대금에서 **제외**된다."

  즉 발전사가 실제로 받는 돈 ≠ 정산단가. RPS/배출권 인센티브가 *별도 정산금*으로
  지급되며 정산단가에는 안 들어간다.
- **인코딩**: utf-8-sig(BOM 포함). 단일 헤더.

### 스키마 (wide → long 변환 후)

| 캐노니컬 fuel_type | vendor 컬럼 | 비고 |
|---|---|---|
| `nuclear` | 원자력 | |
| `coal_bituminous` | 석탄_유연탄 | |
| `coal_anthracite` | 석탄_무연탄 | |
| `coal_total` | 석탄_계 | 유연탄+무연탄 합계 (vendor 산정) |
| `oil` | 유류 | |
| `lng` | LNG | |
| `pumped_storage` | 양수 | 양수발전 |
| `renewable` | 신재생_계 | 8개 서브카테고리의 vendor 합 |
| `other` | 기타 | 부정기/특수 |
| `total` | 합계 | 전체 가중평균 |

**드롭된 서브카테고리** (8개, DQ 노트 기록): 신재생_연료전지, 신재생_석탄가스화,
신재생_태양, 신재생_풍력, 신재생_수력, 신재생_해양, 신재생_바이오, 신재생_폐기물.
필요할 경우 raw_copy_path에서 재로드 가능.

### 데이터 품질 이슈

1. **`other` 카테고리는 일부 기간에만 존재**: 292개월 중 214개월에만 값이 있음(284개월 중
   78개월 결측). 이전 기간엔 vendor가 "기타" 분류를 안 했을 가능성. lag feature를 쓸 때
   해당 시기는 자연스럽게 NaN으로 잡힌다.
2. **수정정산으로 인한 시계열 비정상성**: 최근 12개월 데이터는 *최종 확정이 아닐 수 있음*.
   특히 연말~연초에 재수집하면 작년 1~12월 값이 살짝 바뀌어 있을 수 있다. 학습/검증
   split이 안정적이려면 *최근 12개월 데이터는 valid/test에만 쓰고 train에는 안전을 위해
   최소 13개월 lag 후 데이터만 쓰는 것* 권장.
3. **단위 검증**: `합계` 행(total)은 KPX가 발전량 가중평균으로 산정한 값이어야 한다.
   `합계 ≈ Σ(연료원_정산단가 × 연료원_발전량비중)`이 성립하는지 *발전량 데이터가 들어오면
   필수 검증*. 현재는 발전량 데이터가 quarantined 상태라 검증 불가.

### 모델링 시 주의사항

- **정산단가는 SMP의 *결과(후행)***: 발전사가 SMP로 정산받은 뒤, 자체 정산단가 발전기는
  별도 보정. 따라서 정산단가를 SMP 예측 feature로 쓸 때는 **반드시 lag만 가능**. 같은 달
  정산단가를 같은 달 SMP feature로 넣으면 미래 정보 누출이 되지는 않지만, *결과 변수를
  원인 변수로 쓰는* 인과 오인이 된다.
- **수정정산 보호**: lag_1m, lag_2m, lag_3m 같은 짧은 lag는 *수정 가능성이 살아있는*
  값이다. 안전한 default는 `settlement_unit_price_*_lag_12m`(전년 동월값, 거의 확정).
  이번 라운드 Ridge 디폴트 features는 lag_1m만 쓰지만, prod 모델에서는 lag_12m 고려 필요.
- **`total` vs 개별 연료원**: 둘 다 잡으면 다중공선성 강함(합계가 가중합으로 거의
  설명됨). 둘 중 하나만 선택하는 것이 권장(보통 개별).

---

## 4. `kpx_rec_weekly_file` — 주간 REC 현물시장 거래현황

### 출처와 생성 메커니즘

- **출처**: 공공데이터포털 — *한국전력거래소_주간 신재생에너지 인증서(REC) 거래현황*.
- **파일명은 "주간"이지만 실제 row 단위는 *거래일(trading day)***. KPX는 주 2~3회 REC
  현물시장을 운영하므로, 한 주에 여러 행이 들어가거나 0 행일 수도 있다. 우리 metadata
  `frequency: weekly_file_trading_day`에서 이 사실을 명시한다.
- **인코딩**: cp949. 단일 헤더.
- **시간 범위**: 2017-03-28 ~ 2023-03-30. **2023년 4월 이후 데이터는 아직 없음** — 다음
  라운드에 zip 파일(REC 일별/시간대별)에서 보충 예정.

### 스키마

| vendor 컬럼 | 캐노니컬 | 단위 |
|---|---|---|
| `거래일` | trade_date | date |
| `체결 수량` | rec_volume | REC |
| `평균가` | avg_price_krw | KRW/REC |
| `체결총액` | total_amount_krw | **KRW** (백만원 아님) |
| `시작가` | open_price_krw | KRW/REC |
| `종가` | close_price_krw | KRW/REC |
| `기준가` | base_price_krw | KRW/REC |
| `최고가` | high_price_krw | KRW/REC |
| `최저가` | low_price_krw | KRW/REC |

**단위 검증 포인트**: `체결총액 ≈ 평균가 × 체결수량`이 ±1% 이내로 맞아야 한다.
체결총액 단위가 KRW가 맞는지 의심되면 이 항등식으로 확인. (예: 2017-03-28
체결수량 2568 × 평균가 119012 ≈ 305,622,816 ≈ 305,624,400 — 정확히 매치.)

### 데이터 품질 이슈

1. **거래일 calendar 불연속**: REC 현물시장은 정해진 요일(주 2~3회)에만 운영. 우리
   parquet에는 거래일만 등장하므로 *비거래일 row가 아예 없다*. SMP와 join할 때 weekday
   calendar로 reindex하면 안 되고, 거래일 인덱스를 유지하든 monthly로 집계해야 한다.
2. **기준가(`기준가`)의 의미**: KPX 시장운영규칙상 *기준가 = 전 거래일 가격 + 정책 조정*.
   시작가/종가와 기준가는 따로 분리된 *시장 메커니즘 변수*. 시작가 ≠ 기준가일 수 있고,
   둘 다 의미 있는 신호다.
3. **가격대 변동성**: 2017년 ~120K KRW/REC, 2020년 ~30K, 2022년 ~50K 등 정책 변동에 따라
   ±300% 흔들린다. 시계열 outlier가 아니라 *정책 충격*이므로 winsorize/clip은 부적절.

### 모델링 시 주의사항

- **SMP와 시간 단위 불일치**: SMP는 월, REC는 거래일. join하려면 *월 단위 집계*가 필요.
  권장 집계는 **거래량 가중평균**:

  ```
  monthly_rec_avg = Σ(avg_price × volume) / Σ(volume)
  ```

  단순 산술평균은 거래량 적은 날의 가격에 과민하다.
- **forward-fill 금지**: REC는 시장 가격이라 비거래일의 *가격*을 그 전 거래일 값으로
  forward-fill하면 *시장이 닫혔는데 가격이 그대로*라는 잘못된 신호. 대신
  `is_trading_day` 같은 indicator를 추가하고 비거래일은 NaN을 유지하는 게 안전.
- **거래량과 가격의 인과**: 거래량 급증이 가격을 끌어내리는 패턴(공급 충격) vs 가격 상승이
  거래량을 유발하는 패턴(수요 충격)이 모두 가능. Granger causality나 VAR 분석 권장.
- **육지/제주 구분 없음**: 이 파일은 *전국 통합* REC 가격. KPX의 다른 데이터셋(REC 현물시장
  *일별/시간대별*, 이번 라운드 미사용)에는 육지/제주 분리가 있다.

---

## 5. `kpx_generation_yearly` *(quarantined — 사용 금지)*

### 발견 경위

원본 파일명 두 개:

- `HOME_전력거래_계통한계가격_가중평균SMP.csv`
- `HOME_전력거래_계통한계가격_가중평균SMP (2).csv`

**파일명에 "SMP"가 들어가 있지만 실내용은 SMP가 아니라 연도별 발전원별 발전량(MWh)**.
이는 EPSIS 사용자가 "발전량" 차트를 본 후 "SMP" 차트를 보고 같은 이름으로 export 받은
실수의 결과로 추정된다.

### 실내용 (헤더 발췌)

```
row 0: 연도, 수력, , , , 기력, , , , , 복합화력, , , , , 원자력, 신재생, 집단, 내연력, 기타, 총계, 상용자가, ...
row 1: , 일반수력, 양수, 소수력, 소계, 무연탄, 유연탄, 중유, 가스, 소계, 일반, 열공급, LNG, 유류, 계, ...
row 2: 2024, 3538974, 4677237, 761339, 8977549, 1990754, 159749558, ...
```

연도별 발전원별 *발전량* 데이터(단위 MWh 추정). 굉장히 가치 있는 데이터이지만 *이번 라운드에는*
별도 loader가 없어 quarantine.

### 격리 사실

- 격리 위치: `data/raw/manual_or_filedata/quarantine/kpx_generation_yearly/`
- 매니페스트: `manifest_*.json` — 원본 경로, 격리 경로, sha256, reason 기록.
- 상태: `status: pending_schema_verification`, `reason: filename_content_mismatch`.

### 사용 시 주의

**이번 라운드에서 SMP 파이프라인에 절대 흘러들어가게 하지 말 것**. KEPCO 로더의
`column_mapping`은 `년도/월/육지계통한계가격...`을 요구하므로 자동으로 실패하지만,
사용자가 새 로더를 만들면서 이 파일을 SMP로 분류하는 실수를 막기 위해 regression test
(`test_no_filename_based_source_assumption`)를 두었다.

### 다음 라운드 작업 (현재 미구현)

1. `kpx_generation_yearly` 로더 신규 작성: 2-row header 평탄화 → long-format
   `(year, fuel_type, gen_mwh)`로 melt.
2. 단위 확정: MWh / MW·연 / TWh — vendor 명시 없음. 다른 출처(전력거래소 전력시장통계
   PDF)와 cross-check 필요.
3. 이 데이터를 SMP feature로 쓰려면: *연도 단위*라 월별 모델에 직접 join 불가.
   "전년도 발전믹스" 같은 lag feature로만 의미 있음.

---

## 6. 데이터셋 간 정합성 체크

이 프로젝트의 데이터들은 서로 종속되어 있다. 발견되는 이상은 다음 체크로 detect 가능하다.

### 6.1 SMP 두 소스(KEPCO vs HOME) 일치

```python
# build_smp_monthly_features 호출 후 sideinfo.json 확인
side = json.load(open("data/processed/smp_monthly_mainland_h1m.sideinfo.json"))
dedup_log = side["smp_priority_dedup_log"]
# 같은 (period_month, area)에 대해 두 소스의 smp_krw_per_kwh 차이를 본다
```

실측: KEPCO와 HOME의 mainland 값 차이 평균 |Δ| < 0.05 KRW/kWh (반올림 차).
이상이 1 KRW/kWh 이상 벌어지면 EPSIS export 시점의 *수정정산 반영 차이* 가능성 의심.

### 6.2 SMP vs 정산단가의 부등식

이론적으로 *월별 가중평균 SMP* ≤ *총합 정산단가(`total`)* 가 *일반적*이다.

- 이유: 정산단가는 SMP에 더해 *자체 정산단가 발전기*(원자력 등 SMP 미만 변동비 발전기에
  대한 기준연료비 보정)와 *용량요금* 등을 반영하므로 SMP보다 *높은* 게 보통.
- 단, 시기에 따라 역전 가능 (LNG 가격 폭등기에 SMP가 자체정산단가를 일시적으로 추월).

### 6.3 REC × 발전량 일관성

REC 발급량은 신재생 발전량의 단조 함수(가중치 곱). 발전량 데이터가 로드되면:

```
REC_월별_발급량 ≈ Σ(신재생_연료원 × REC_가중치)
```

검증 가능. 가중치는 RPS 고시(태양광 1.0~1.5x, 풍력 1.0x 등).

---

## 7. 한국 전력시장 도메인 컨텍스트

### 7.1 SMP의 미시 정의

SMP(System Marginal Price, 계통한계가격)는 *한 시간 동안 마지막으로 급전된 발전기의
변동비*. KPX는 매일 23시경 다음날 24시간의 시간별 SMP를 결정·공시한다.

- **육지 SMP**: 제주를 제외한 전국 계통의 SMP.
- **제주 SMP**: 제주 계통은 독립적으로 운영되며, 특이 조건(*송전 한계*)에서 *전체 발전기 중
  가장 비싼 발전기 가격*으로 결정될 수 있다.
- **통합 SMP**: 2024-02-01부터 시행. 육지·제주 통합 시장을 가정한 *가상의 단일 SMP*.

월별 SMP는 *시간별 SMP의 가중평균(가중치=시간별 거래량)* — 단순 평균과 다르다.

### 7.2 정산단가의 회계 정의

발전사가 받는 돈 = SMP 정산 + 자체 정산단가 정산 + 용량요금(CP) + 보조서비스 정산.
KPX EPSIS의 "정산단가"는 이 중 *SMP 정산금 + 자체 정산단가 정산금*의 *발전량당 평균*만
포함하며, *RPS 의무이행비용정산금*과 *배출권거래비용정산금*은 **제외**된다.

따라서:

```
정산단가 (EPSIS) ≠ 발전사 실제 회계 단가
정산단가 (EPSIS) ≈ SMP + 자체정산단가 보정
```

### 7.3 REC와 RPS 정책

REC(공급인증서)는 *신재생 발전 1MWh당 1REC* 발급. RPS 의무자(주로 대형 발전사)가
연간 의무이행을 위해 시장에서 매입한다.

- **현물시장**: 주 2~3회 거래일에 호가/체결. *이 프로젝트가 다루는 데이터*.
- **계약시장**: 장외 양자계약. 가격은 *반드시 현물보다 낮다는 보장은 없다*. 보통 더 낮음.
- **자가발전 인증**: RPS 의무자가 자기 신재생 설비로 자가 충당.

가격은 RPS 의무량 고시·신재생 설비 보급 속도·LNG 발전비용에 강하게 종속.

---

## 8. 공통 사용 주의사항

### 8.1 시간 정합성

- 모든 monthly 데이터는 `period_month`(월초 timestamp)로 정렬되어 있다. `period_month`는
  타임존-naive이며 *대한민국 표준시(KST) 기준*이다. DST 없음.
- REC weekly의 `trade_date`도 KST 기준의 거래일자. 시간 정보 없음.
- 모든 *post-event* 시각(`collected_at`)은 *UTC naive*. 한국 시각으로 보려면 +9h.

### 8.2 결측 정책

| 의미 | 표현 |
|---|---|
| vendor가 데이터를 공표하지 않은 (period, area) | row 자체가 없거나 NaN |
| vendor가 0으로 표기한 "no published value" | **로더가 NaN으로 강제 후 drop** |
| 수치형으로 변환 실패 ("-" 등 marker) | NaN |
| 미래 시점 (lag feature 워밍업 부족) | NaN |

**0이라는 값이 의미 있는 데이터셋은 현재 없다**(SMP, 정산단가, REC 가격 모두 항상 양수).
0을 본다면 거의 항상 vendor 결측 marker.

### 8.3 데이터 버전 관리

- 모든 raw 파일은 `data/raw/<ns>/<rest>/YYYY/MM/DD/raw_<stamp>.<ext>`로 *복사*된다.
  원본은 보존된다.
- `parsed_*.parquet`과 `metadata_*.json`도 *시점별*로 따로 저장된다.
- 같은 데이터를 다시 받으면 **새 stamp의 파일이 추가**된다. build_monthly는 (period, area)
  단위로 *최신을 가져온다*. **이전 스냅샷도 디스크에 살아 있다** — 정산단가 수정정산이
  실제로 일어났는지 확인하려면 *예전 parsed 파일들을 직접 비교*하면 된다.

### 8.4 자주 빠지는 함정

1. **육지 + 제주 합산** — 절대 금지. 항상 area별로 모델링.
2. **정산단가를 같은 달 SMP feature로 사용** — 인과 오인. 적어도 lag_1m 이상.
3. **REC 가격 forward-fill** — 시장 휴일을 거래일로 위장하지 말 것.
4. **0 KRW/kWh를 정상값으로 간주** — vendor의 결측 marker. 로더가 막아주지만 직접
   parquet을 읽어 쓰는 코드는 별도 가드 필요.
5. **train/test 랜덤 split** — 시계열은 *반드시 chronological split*. 본 프로젝트
   `train.py`의 `_split_chronological`이 강제.
6. **HOME 가중평균 SMP를 산술평균으로 칭하기** — 보고서 작성 시 주의.
7. **자가 정산단가 발전기와 RPS 정산금을 정산단가에 포함된 것으로 해석** — EPSIS 정산단가
   정의에서 *제외*되어 있다. 발전사 실제 수익 분석에는 별도 보정 필요.

---

## 9. 다음 라운드 추가 예정 (현재 미구현)

| 데이터셋 | 출처 | 비고 |
|---|---|---|
| REC 일별 시간대별 거래 | KPX zip (2개 CSV) | 육지/제주 분리, 시간대별 가격 |
| REC 연도별 거래량/금액 요약 | 4개 xlsx (2021~2024) | 멀티-row 헤더, 현물/계약 시장 분리 |
| 월별 연료원별 전력거래금액 | EPSIS csv | 정산단가의 분자(거래금액)만 |
| 연료전지 REC 연도별 거래대금 | KPX csv | 단일 연료원 시계열 |
| **연도별 발전원별 발전량** | **quarantined → 신규 loader 필요** | 위 6번 정합성 검증의 핵심 |
| KPX SMP day-ahead API (시간별) | 공공데이터 API | API key 승인 후 |
| 기상청 ASOS API | KMA API | 기온/습도/일사로 SMP 모델 확장 |

각 데이터셋 추가 시 본 카탈로그에 동일 양식으로 항목을 추가한다.
