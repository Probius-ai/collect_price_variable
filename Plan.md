아래는 Claude Code에 그대로 넣기 좋은 Plan.md 초안입니다. 핵심 방향은 **“메인 프로젝트를 위한 전력시장 가격 변수 예측 모듈”**이고, 타깃은 SMP → 정산단가 → REC 순서로 단계 확장하는 구조로 잡았습니다.

⸻

Plan.md

0. 프로젝트 한 줄 정의

한국 전력시장 데이터를 기반으로 SMP, 정산단가, REC 가격/거래지표를 수집·정제·예측하여, 메인 프로젝트에서 사용할 전력시장 가격 변수 예측 모듈을 구축한다.

⸻

1. 프로젝트 목적

본 프로젝트의 목적은 한국 전력시장 가격 변수들을 자동 수집하고, 시계열 예측 모델을 통해 향후 전력시장 가격 지표를 예측하는 것이다.

예측 대상은 다음 3개 축으로 구분한다.

1. SMP, System Marginal Price, 계통한계가격
2. 정산단가
3. REC, Renewable Energy Certificate, 신재생에너지 공급인증서 가격/거래지표

전력거래소 자료 기준으로 SMP는 전력시장가격이며 시간별로 산정되고, 하루 24개의 값이 공표된다. 공공데이터포털의 육지 SMP 데이터 설명에서도 SMP는 전력시장가격 원/kWh이며, 시간별로 산정된다고 명시한다.  ￼

⸻

2. 프로젝트 범위

2.1 1차 범위: SMP 예측

가장 먼저 구현할 타깃은 SMP 예측이다.

이유는 다음과 같다.

* 데이터 접근성이 좋다.
* 시간 단위 데이터가 존재한다.
* 전력수요, 연료비, 발전믹스, 기상 변수와의 설명 관계가 비교적 명확하다.
* 정산단가와 REC보다 예측 모델 검증이 쉽다.

공공데이터포털에는 한국전력거래소_계통한계가격 및 수요예측(하루전 발전계획용) API가 있으며, 이 API는 육지와 제주의 1시간 단위 계통한계가격 정보 및 수요예측 값을 제공한다. 하루 1회, 23시경 갱신되는 데이터라고 설명되어 있다.  ￼

2.2 2차 범위: 정산단가 예측

정산단가는 발전기 또는 연료원별로 전력시장 정산 결과를 반영한 가격 지표다.

전력거래소 EPSIS의 연료원별 정산단가 페이지는 자료출처를 전력시장 최종정산 기준자료로 설명하며, 연간 전력시장통계 확정 시점까지 변경될 수 있다고 안내한다. 또한 정산단가 산정 시 전력거래대금에서 RPS 의무이행비용정산금 및 배출권거래비용정산금은 제외된다고 명시한다.  ￼

공공데이터포털의 한국전력거래소_월별 정산단가 데이터는 전력시장에 참여하여 정산받는 발전기 대상 자료로 계산한 연료원별/회원사별 정산단가 자료이며, 수정정산 등에 의해 변동될 수 있다고 설명한다.  ￼

따라서 정산단가는 실시간 예측보다는 월별 예측 또는 후행 분석에 적합하다.

2.3 3차 범위: REC 예측

REC는 신재생에너지 공급인증서 거래시장 지표다.

공공데이터포털의 한국전력거래소_REC 현물시장 정보는 REC 현물시장 장운영일의 거래건수, 평균가 등 장운영실적을 육지/제주 구분으로 제공한다고 설명한다.  ￼

또한 한국전력거래소_신에너지 및 재생에너지 공급인증서 거래량 데이터는 REC 거래시장의 월간 거래현황을 제공하며, 항목으로 현물시장 거래량, 계약시장 거래량, 현물시장 거래금액, 계약시장 거래금액을 포함한다고 설명한다.  ￼

REC는 SMP보다 시장 구조와 정책 영향이 강하므로, 1차 모델 이후 확장 타깃으로 둔다.

⸻

3. 핵심 용어 정의

3.1 SMP

SMP는 전력시장가격이다. 전력거래소의 연료원별 SMP 결정 횟수 API 설명에 따르면, SMP는 거래시간별로 적용되는 전력량에 대한 전력시장가격 원/kWh이며, 육지 및 제주지역으로 구분된다.  ￼

모델링 관점

SMP는 다음 변수들의 영향을 받을 가능성이 높다.

* 전력수요
* 예비율
* LNG 발전 비중
* 유연탄 발전 비중
* 원자력 발전 비중
* 연료비
* 환율
* 기온
* 계절성
* 요일/공휴일
* SMP 결정 연료원

⸻

3.2 SMP 결정 횟수

SMP 결정 횟수는 특정 연료원이 시장가격을 결정한 횟수다.

공공데이터포털의 연료원별 SMP 결정 횟수(일별) API는 전력시장의 연료원별 시장가격을 결정한 횟수를 일별로 합산하여 제공하며, LNG, 유류, 유연탄, 무연탄, 원자력 항목을 제공한다고 설명한다.  ￼

또한 2024년 8월 20일부터 제주와 육지의 연료원별 SMP 결정 횟수를 구분하여 제공한다고 되어 있다.  ￼

모델링 관점

SMP 예측에서 매우 중요한 설명 변수다.

예시:

lng_smp_decision_count_daily
coal_smp_decision_count_daily
oil_smp_decision_count_daily
nuclear_smp_decision_count_daily
anthracite_smp_decision_count_daily

단, 이 변수는 실제 예측 시점에 이미 알 수 있는지 확인해야 한다.

* 당일/익일 예측에서는 누출 가능성 있음
* 월별 사후 예측이나 설명 분석에서는 사용 가능
* 실시간 예측에서는 lag 변수로만 사용 권장

⸻

3.3 정산단가

정산단가는 전력시장 정산 결과 기반 가격 지표다.

전력거래소 EPSIS 설명에 따르면 정산단가는 전력시장 최종정산 기준자료이고, 변경 가능성이 있다. 또한 RPS 의무이행비용정산금과 배출권거래비용정산금은 정산단가 산정 시 전력거래대금에서 제외된다.  ￼

모델링 관점

정산단가는 SMP보다 후행성이 강하다.

예측 단위는 다음 중 하나로 제한하는 것이 적절하다.

* 월별 연료원별 정산단가
* 월별 회원사별 정산단가
* 월별 평균 정산단가

일별/시간별 정산단가 예측은 데이터 구조 확인 전까지 범위에서 제외한다.

⸻

3.4 REC

REC는 신재생에너지 공급인증서 거래시장 지표다.

공공데이터포털에는 REC 현물시장 정보 조회 서비스가 있으며, 장운영일의 거래건수, 평균가 등 장운영실적을 육지/제주로 구분하여 제공한다.  ￼

월간 REC 거래현황 데이터는 현물시장 거래량, 계약시장 거래량, 현물시장 거래금액, 계약시장 거래금액을 제공한다.  ￼

모델링 관점

REC는 다음 변수의 영향을 받을 수 있다.

* 신재생 발전량
* 태양광 발전량
* REC 거래량
* REC 거래금액
* RPS 정책 변화
* 현물시장 평균가
* 계약시장 거래량
* SMP
* 계절성
* 태양광 발전 계절성

단, 정책·제도 변수의 정량화가 어렵기 때문에, 초기에는 시계열 + 거래량 기반 예측으로 시작한다.

⸻

4. 데이터 소스 우선순위

4.1 공식 소스 우선순위

데이터 신뢰도는 다음 순서로 둔다.

1순위: 전력거래소 KPX / EPSIS / 공공데이터포털의 KPX 제공 API
2순위: 기상청 기상자료개방포털
3순위: 한국은행 ECOS
4순위: KOSIS / 공공데이터포털의 타 기관 데이터
5순위: 민간 블로그, 뉴스, 2차 가공 데이터

민간 블로그와 뉴스는 구현 근거로 사용하지 않는다. 데이터 API 명세 확인 또는 배경 설명용으로만 사용한다.

⸻

4.2 주요 데이터셋 후보

우선순위	데이터	사용 목적	출처
1	계통한계가격 및 수요예측	SMP 타깃, 수요 변수	KPX 공공데이터 API
2	육지 SMP	육지 SMP 타깃	KPX 공공데이터 파일
3	제주 SMP	제주 SMP 타깃	KPX 공공데이터 파일
4	연료원별 SMP 결정 횟수	가격결정 연료원 변수	KPX 공공데이터 API
5	발전원별 발전량	발전믹스 변수	KPX 공공데이터 API
6	월간 연료비용 정보	연료비 변수	KPX 공공데이터 API
7	월별 정산단가	정산단가 타깃	KPX 공공데이터
8	REC 현물시장 정보	REC 타깃	KPX 공공데이터 API
9	REC 월간 거래현황	REC 보조 변수	KPX 공공데이터
10	ASOS 기상자료	기온, 습도, 강수, 일사	기상청
11	환율	수입연료비 보정	한국은행 ECOS

⸻

5. 데이터셋별 Fact Check

5.1 SMP + 수요예측 API

확인된 사실

공공데이터포털의 한국전력거래소_계통한계가격 및 수요예측(하루전 발전계획용) API는 다음을 제공한다.

* 육지와 제주
* 1시간 단위 계통한계가격
* 수요예측 값
* 하루 1회 갱신
* 23시경 갱신
* 각 시간은 해당 단위기간의 끝점으로 표시됨
    예: 거래시간 06시는 05:00 직후부터 06:00에 종료하는 기간을 의미  ￼

구현 주의사항

시간 해석을 잘못하면 lag feature가 틀어진다.

거래시간 06시 = 05:00~06:00 구간

따라서 내부 timestamp는 다음 둘 중 하나로 통일한다.

interval_end_time = 06:00
interval_start_time = 05:00

추천은 interval_end_time 기준이다.

⸻

5.2 육지 SMP 파일 데이터

확인된 사실

공공데이터포털의 한국전력거래소_육지 계통한계가격(SMP) 데이터는 다음을 설명한다.

* SMP는 전력시장가격 원/kWh
* 시간별로 산정
* 1일 총 24개 값 공표
* 일자별/시간별 육지 SMP 확인 가능
* 갱신주기: 매일
* 산정방법은 전력시장운영규칙 제2장 제4절 가격결정 부분 참조  ￼

구현 주의사항

육지와 제주는 분리해서 모델링한다.

초기 모델은 육지만 사용한다.

area = mainland
target = smp_mainland

제주 SMP는 송전제약, 계통 특성, 재생에너지 비중, 출력제어 영향이 크므로 별도 모델이 적절하다.

⸻

5.3 제주 SMP 파일 데이터

확인된 사실

공공데이터포털의 한국전력거래소_제주(하루전시장) 계통한계가격(SMP) 데이터는 제주 하루전시장 SMP를 일자별, 시간별로 조회하는 데 활용 가능하다고 설명한다. 또한 SMP는 시간별로 산정되며 하루 24개의 값이 공표된다고 설명한다.  ￼

제주지역 계통한계가격은 일부 조건에서 전체 발전기의 유효 발전가격 중 가장 높은 가격으로 한다고 설명되어 있다.  ￼

구현 주의사항

제주는 육지와 같은 모델에 섞지 않는다.

bad: target = national_smp
good: target_mainland, target_jeju 분리

⸻

5.4 연료원별 SMP 결정 횟수

확인된 사실

공공데이터포털의 한국전력거래소_연료원별 SMP 결정 횟수(일별) API는 전력시장의 연료원별 시장가격 결정 횟수를 일별 합산하여 제공한다. LNG, 유류, 유연탄, 무연탄, 원자력 항목을 제공한다.  ￼

2024년 8월 20일부터 제주와 육지의 연료원별 SMP 결정 횟수를 구분하여 제공한다.  ￼

공공데이터 공지에 따르면 변경 후 URL은 SmpDecByFuel2/getSmpDecByFuel2 형태이며, 지역 구분 항목 areaNm이 추가되었다.  ￼

구현 주의사항

해당 API는 2024년 8월 20일 전후 스키마가 다를 수 있다.

따라서 ETL에서 다음 처리를 해야 한다.

if date < 2024-08-20:
    areaNm may be missing
else:
    areaNm required

⸻

5.5 발전원별 발전량

확인된 사실

공공데이터포털의 한국전력거래소_발전원별 발전량(계통기준) API는 5분 단위 발전원별 발전량을 조회할 수 있는 서비스다. 항목은 수력, 유류, 유연탄, 원자력, 양수, 가스, 국내탄, 신재생, 태양광 등으로 제공된다. 단, 데이터 스케일과 취득 상황에 따라 일부 차이가 발생할 수 있으므로 참고용으로 사용하라고 안내되어 있다.  ￼

구현 주의사항

5분 단위 데이터를 시간 단위 SMP와 결합하려면 집계가 필요하다.

추천 집계:

hourly_generation_by_fuel = mean(5min_generation within hour)
hourly_generation_share_by_fuel = fuel_generation / total_generation

SMP의 거래시간이 구간 종료시각 기준이므로 발전량도 같은 시간 기준으로 맞춘다.

⸻

5.6 월간 연료비용 정보

확인된 사실

공공데이터포털의 한국전력거래소_월간 연료비용 정보 API는 REST 방식이며 JSON+XML 포맷을 제공한다. 키워드는 연료비용, 열량단가, 연료원별, 월간, 발전비용평가로 등록되어 있다. 수정일은 2026년 4월 15일로 확인된다.  ￼

구현 주의사항

월간 데이터이므로 시간별 SMP에 붙일 때는 다음 중 하나를 선택해야 한다.

option 1: 해당 월 모든 시간에 같은 월간 연료비 적용
option 2: 전월 연료비를 lag 변수로 적용
option 3: rolling 3개월 평균 적용

실제 예측 시점에서 미래 월의 연료비를 모르면 누출이 발생한다.

추천:

fuel_cost_lag_1m
fuel_cost_lag_2m
fuel_cost_rolling_3m

⸻

5.7 월별 정산단가

확인된 사실

공공데이터포털의 한국전력거래소_월별 정산단가 데이터는 전력시장에 참여하여 정산받는 발전기 대상 자료로 계산한 연료원별/회원사별 정산단가 자료다. 해당 자료는 수정정산 등에 의해 변동될 수 있다고 설명한다.  ￼

EPSIS 연료원별 정산단가 설명은 갱신시기가 M+1월 초이고, 전력시장 최종정산 기준자료이며, 연간 전력시장통계 확정 시점까지 변경될 수 있다고 설명한다.  ￼

구현 주의사항

정산단가 예측은 월별 모델로 구축한다.

target_frequency = monthly
target = settlement_unit_price_by_fuel

정산단가는 수정될 수 있으므로 데이터 버전 관리가 필요하다.

raw_snapshot_date
source_modified_date
etl_run_at

⸻

5.8 REC 현물시장 정보

확인된 사실

공공데이터포털의 한국전력거래소_REC 현물시장 정보는 신재생에너지공급인증서 현물시장 장운영일의 거래건수, 평균가 등 장운영실적을 육지/제주 구분하여 제공하는 서비스다.  ￼

구현 주의사항

REC 현물시장은 장운영일 기준 데이터이므로 일별 캘린더와 바로 1:1 매칭되지 않을 수 있다.

처리 방식:

거래일 기준 REC 데이터 수집
비거래일은 NaN 유지
모델 입력용으로는 forward fill 여부를 실험

단, 가격 자체를 forward fill하면 시장가격 예측 오류가 발생할 수 있으므로 is_trading_day 변수를 추가한다.

⸻

5.9 REC 월간 거래현황

확인된 사실

공공데이터포털의 한국전력거래소_신에너지 및 재생에너지 공급인증서 거래량 데이터는 REC 거래시장의 월간 거래현황을 제공하며, 항목은 현물시장 거래량, 계약시장 거래량, 현물시장 거래금액, 계약시장 거래금액이다. 단위는 REC와 백만원이다.  ￼

구현 주의사항

월간 REC 평균단가를 직접 계산할 수 있다.

spot_rec_avg_price = spot_trade_amount_million_krw * 1_000_000 / spot_trade_volume_rec
contract_rec_avg_price = contract_trade_amount_million_krw * 1_000_000 / contract_trade_volume_rec

단, 원자료에 이미 평균가가 있으면 원자료 값을 우선 사용하고, 계산값은 검증용으로 사용한다.

⸻

5.10 기상 데이터

확인된 사실

기상청 기상자료개방포털의 ASOS 자료는 분, 시간, 일, 월, 연 자료를 제공하고, 제공 요소에는 기온, 강수, 바람, 기압, 습도, 일사, 일조, 눈, 구름, 시정, 지면상태 등이 포함된다. 제공기간은 1904년부터이며 지점별, 요소별로 다를 수 있다.  ￼

공공데이터포털의 기상청 ASOS 설명도 관측요소로 기온, 강수량, 습도, 기압, 풍향, 풍속 등을 제공하며, 관측주기는 1분이고 시간자료, 일자료, 월자료, 연자료를 제공한다고 설명한다.  ￼

구현 주의사항

전국 전력수요 모델에는 단일 서울 기온만 쓰면 편향될 수 있다.

초기 버전:

weather_station = Seoul, Busan, Daegu, Gwangju, Daejeon
weighted_temperature = 단순 평균

고도화 버전:

weighted_temperature = 지역별 전력수요 또는 인구 기반 가중평균

⸻

6. 예측 타깃 정의

6.1 Target A: Hourly Mainland SMP

target_name: smp_mainland_hourly
unit: KRW/kWh
frequency: hourly
region: mainland
source: KPX
priority: P0

예측 horizon

T+1 hour
T+24 hours
T+7 days

초기 구현은 T+24 hours를 목표로 한다.

⸻

6.2 Target B: Daily Average SMP

target_name: smp_mainland_daily_avg
unit: KRW/kWh
frequency: daily
region: mainland
source: KPX
priority: P0

생성 방식:

daily_avg_smp = mean(hourly_smp_1_to_24)
daily_max_smp = max(hourly_smp_1_to_24)
daily_min_smp = min(hourly_smp_1_to_24)
daily_peak_hour_smp = max hourly smp

⸻

6.3 Target C: Monthly Settlement Unit Price

target_name: settlement_unit_price_monthly
unit: KRW/kWh
frequency: monthly
source: KPX/EPSIS
priority: P1

분해 타깃:

settlement_unit_price_total
settlement_unit_price_lng
settlement_unit_price_coal
settlement_unit_price_nuclear
settlement_unit_price_oil
settlement_unit_price_renewable

⸻

6.4 Target D: REC Spot Average Price

target_name: rec_spot_avg_price
unit: KRW/REC
frequency: trading_day or weekly/monthly
source: KPX
priority: P2

추천 시작 단위:

monthly_rec_spot_avg_price

REC 현물시장 일별/장운영일 데이터는 결측과 거래일 처리가 필요하므로, 초기에는 월별로 시작한다.

⸻

7. Feature 설계

7.1 시간 변수

year
month
day
hour
day_of_week
is_weekend
is_holiday
season
is_summer
is_winter
is_peak_load_season

주의:

hour는 SMP 거래시간의 interval_end_time 기준으로 생성

⸻

7.2 수요 변수

demand_forecast
demand_forecast_lag_1h
demand_forecast_lag_24h
demand_forecast_lag_168h
demand_forecast_rolling_24h_mean
demand_forecast_rolling_7d_mean
demand_forecast_daily_max
demand_forecast_daily_min

KPX의 계통한계가격 및 수요예측 API는 1시간 단위 SMP와 수요예측 값을 함께 제공한다.  ￼

⸻

7.3 SMP lag 변수

smp_lag_1h
smp_lag_2h
smp_lag_3h
smp_lag_24h
smp_lag_48h
smp_lag_168h
smp_rolling_24h_mean
smp_rolling_24h_std
smp_rolling_7d_mean
smp_rolling_7d_std
smp_daily_avg_lag_1d
smp_daily_avg_lag_7d

주의:

모든 lag/rolling feature는 반드시 target 시점 이전 데이터만 사용한다.

⸻

7.4 발전믹스 변수

KPX 발전원별 발전량 계통기준 API는 5분 단위 발전원별 발전량을 제공하며, 수력, 유류, 유연탄, 원자력, 양수, 가스, 국내탄, 신재생, 태양광 항목을 포함한다.  ￼

gen_hydro_mw
gen_oil_mw
gen_coal_mw
gen_nuclear_mw
gen_pumped_storage_mw
gen_gas_mw
gen_domestic_coal_mw
gen_renewable_mw
gen_solar_mw

비중 변수:

share_hydro
share_oil
share_coal
share_nuclear
share_gas
share_renewable
share_solar

파생 변수:

thermal_generation_share = share_coal + share_gas + share_oil
low_marginal_cost_share = share_nuclear + share_renewable
gas_minus_nuclear_share = share_gas - share_nuclear

⸻

7.5 연료비 변수

KPX 월간 연료비용 정보 API는 연료비용, 열량단가, 연료원별, 월간, 발전비용평가 관련 데이터로 등록되어 있다.  ￼

fuel_cost_lng
fuel_cost_coal
fuel_cost_oil
fuel_cost_nuclear
fuel_cost_lng_lag_1m
fuel_cost_coal_lag_1m
fuel_cost_oil_lag_1m
fuel_cost_lng_rolling_3m
fuel_cost_coal_rolling_3m
fuel_cost_oil_rolling_3m

주의:

월간 연료비를 시간별 SMP에 붙일 때는 미래 정보 누출 방지를 위해 lag 변수를 기본으로 한다.

⸻

7.6 SMP 결정 연료원 변수

smp_decision_lng_count_daily
smp_decision_oil_count_daily
smp_decision_bituminous_coal_count_daily
smp_decision_anthracite_count_daily
smp_decision_nuclear_count_daily

비중 변수:

smp_decision_lng_ratio
smp_decision_coal_ratio
smp_decision_oil_ratio

주의:

일별 SMP 결정 횟수는 당일 전체 결과를 알아야 생성될 가능성이 높으므로, 시간별 예측에서는 lag 처리한다.

smp_decision_lng_count_lag_1d
smp_decision_lng_count_lag_7d

⸻

7.7 기상 변수

ASOS는 기온, 강수, 바람, 기압, 습도, 일사, 일조 등 요소를 제공한다.  ￼

temperature
humidity
precipitation
wind_speed
solar_radiation
sunshine_duration

파생 변수:

CDD = max(temperature - 24, 0)
HDD = max(18 - temperature, 0)
temp_squared = temperature ** 2
is_heatwave = temperature >= 33
is_coldwave = temperature <= -12

CDD/HDD 기준온도는 프로젝트 내부 기준으로 명시한다. 한국 공식 기준과 다를 수 있으므로, 보고서에서는 “모델링 목적의 파생 변수”라고 표기한다.

⸻

7.8 REC 변수

REC 현물시장 정보는 거래건수, 평균가 등 장운영실적을 제공한다.  ￼

rec_spot_avg_price
rec_spot_trade_count
rec_spot_trade_volume
rec_spot_trade_amount
rec_is_trading_day

월간 REC 거래현황은 현물시장 거래량, 계약시장 거래량, 현물시장 거래금액, 계약시장 거래금액을 제공한다.  ￼

rec_spot_monthly_volume
rec_contract_monthly_volume
rec_spot_monthly_amount
rec_contract_monthly_amount
rec_spot_monthly_avg_price_calculated
rec_contract_monthly_avg_price_calculated

⸻

8. 데이터 누출 방지 규칙

이 프로젝트에서 가장 중요한 것은 미래 정보 누출 방지다.

8.1 금지

target 시점 이후의 SMP 사용 금지
target 시점 이후의 발전량 사용 금지
target 시점 이후의 SMP 결정 횟수 사용 금지
target 월의 최종 연료비용을 예측 시점에 이미 알고 있다고 가정 금지
정산단가 수정정산 이후 값을 과거 예측에 그대로 사용하는 것 금지

8.2 허용

과거 SMP lag
과거 수요예측 lag
과거 발전량 lag
과거 연료비 lag
과거 REC 가격 lag
예측 시점 이전에 공개된 기상 관측값
예측 시점 이전에 공개된 기상 예보값

8.3 데이터셋에 반드시 포함할 컬럼

event_time
published_time
collected_time
source_name
source_url_or_api_name
is_final
is_revised

정산단가처럼 사후 수정 가능성이 있는 데이터는 published_time과 collected_time을 분리한다.

⸻

9. 모델링 전략

9.1 Baseline 0: Naive

prediction = smp_lag_24h

목적:

* 모든 모델이 최소한 이 기준보다 나아야 함
* 전력가격의 일중 패턴을 반영하는 강력한 baseline

⸻

9.2 Baseline 1: Seasonal Naive

prediction = smp_lag_168h

즉, 전주 동일 요일 동일 시간 SMP를 예측값으로 사용한다.

⸻

9.3 Baseline 2: Linear Regression / Ridge

설명력 확보용.

features:
- demand_forecast
- temperature
- CDD
- HDD
- fuel_cost_lng_lag_1m
- share_gas_lag
- share_nuclear_lag
- smp_lag_24h
- hour
- month

⸻

9.4 Main Model 1: LightGBM

초기 메인 모델로 추천한다.

이유:

* tabular 시계열 feature에 강함
* 결측 처리 용이
* 변수 중요도 확인 가능
* 학습 속도 빠름
* Claude Code 구현 난이도 낮음

타깃:

hourly_smp_mainland_t_plus_24h
daily_avg_smp_mainland_t_plus_1d

⸻

9.5 Main Model 2: XGBoost

LightGBM 비교용.

same features
same split
same metrics

⸻

9.6 Main Model 3: SARIMAX

전통 시계열 모델 비교용.

target = daily_avg_smp
exogenous = demand, temperature, fuel_cost_lag, generation_share

⸻

9.7 Optional Model: TFT / LSTM / GRU

초기 구현에서는 제외한다.

도입 조건:

데이터 수집 안정화 완료
baseline 대비 tree model 성능 한계 확인
충분한 시간별 데이터 확보

⸻

10. 평가 지표

10.1 회귀 지표

MAE
RMSE
MAPE
sMAPE
R2

전력가격은 spike가 존재할 수 있으므로 RMSE만 사용하지 않는다.

10.2 방향성 지표

directional_accuracy = sign(y_t - y_t-1) == sign(pred_t - y_t-1)

가격 상승/하락 방향 예측이 중요할 경우 사용한다.

10.3 Spike 평가

peak_threshold = 90th percentile of SMP
peak_precision
peak_recall
peak_f1

SMP 급등 구간을 잘 잡는지 별도로 평가한다.

⸻

11. 데이터 분할 전략

랜덤 split 금지.

train: 과거 구간
validation: train 이후 구간
test: 가장 최근 구간

예시:

train: 2018-01-01 ~ 2023-12-31
valid: 2024-01-01 ~ 2024-12-31
test: 2025-01-01 ~ latest

단, 실제 데이터 확보 가능 기간에 따라 조정한다.

11.1 Walk-forward Validation

최종 평가는 walk-forward 방식으로 수행한다.

for each month in validation_period:
    train on all data before month
    predict next month
    evaluate

⸻

12. 저장 구조

12.1 디렉터리 구조

electricity-price-forecast/
  Plan.md
  README.md
  .env.example
  pyproject.toml
  requirements.txt
  data/
    raw/
      kpx/
      kma/
      ecos/
    interim/
    processed/
    external/
    snapshots/
  notebooks/
    01_data_check.ipynb
    02_feature_analysis.ipynb
    03_model_baseline.ipynb
  src/
    config/
      settings.py
      sources.yaml
    collectors/
      kpx_smp.py
      kpx_smp_decision.py
      kpx_generation.py
      kpx_fuel_cost.py
      kpx_settlement.py
      kpx_rec.py
      kma_weather.py
      ecos_exchange_rate.py
    pipelines/
      collect_all.py
      build_features.py
      train.py
      evaluate.py
      predict.py
    features/
      time_features.py
      lag_features.py
      weather_features.py
      generation_features.py
      rec_features.py
    models/
      naive.py
      lightgbm_model.py
      xgboost_model.py
      sarimax_model.py
    validation/
      leakage_checks.py
      schema_checks.py
      source_checks.py
    utils/
      io.py
      time.py
      logging.py
  tests/
    test_time_alignment.py
    test_no_future_leakage.py
    test_feature_generation.py
    test_schema.py
  outputs/
    models/
    metrics/
    figures/
    predictions/

⸻

13. 데이터베이스 스키마 초안

초기에는 DuckDB 또는 PostgreSQL을 사용한다.

13.1 raw_kpx_smp_hourly

CREATE TABLE raw_kpx_smp_hourly (
    id BIGSERIAL PRIMARY KEY,
    area TEXT NOT NULL,
    trade_date DATE NOT NULL,
    trade_hour INTEGER NOT NULL,
    interval_start TIMESTAMP,
    interval_end TIMESTAMP,
    smp NUMERIC,
    demand_forecast NUMERIC,
    source_name TEXT,
    collected_at TIMESTAMP NOT NULL,
    raw_payload JSONB,
    UNIQUE(area, trade_date, trade_hour, collected_at)
);

13.2 raw_kpx_generation_5min

CREATE TABLE raw_kpx_generation_5min (
    id BIGSERIAL PRIMARY KEY,
    observed_at TIMESTAMP NOT NULL,
    hydro_mw NUMERIC,
    oil_mw NUMERIC,
    coal_mw NUMERIC,
    nuclear_mw NUMERIC,
    pumped_storage_mw NUMERIC,
    gas_mw NUMERIC,
    domestic_coal_mw NUMERIC,
    renewable_mw NUMERIC,
    solar_mw NUMERIC,
    source_name TEXT,
    collected_at TIMESTAMP NOT NULL,
    raw_payload JSONB
);

13.3 raw_kpx_fuel_cost_monthly

CREATE TABLE raw_kpx_fuel_cost_monthly (
    id BIGSERIAL PRIMARY KEY,
    year_month DATE NOT NULL,
    fuel_type TEXT NOT NULL,
    fuel_cost NUMERIC,
    heat_value_unit_price NUMERIC,
    source_name TEXT,
    collected_at TIMESTAMP NOT NULL,
    raw_payload JSONB,
    UNIQUE(year_month, fuel_type, collected_at)
);

13.4 raw_kpx_settlement_monthly

CREATE TABLE raw_kpx_settlement_monthly (
    id BIGSERIAL PRIMARY KEY,
    year_month DATE NOT NULL,
    category_type TEXT,
    category_name TEXT,
    settlement_unit_price NUMERIC,
    unit TEXT DEFAULT 'KRW/kWh',
    source_name TEXT,
    source_modified_date DATE,
    collected_at TIMESTAMP NOT NULL,
    raw_payload JSONB
);

13.5 raw_kpx_rec_spot

CREATE TABLE raw_kpx_rec_spot (
    id BIGSERIAL PRIMARY KEY,
    trade_date DATE NOT NULL,
    area TEXT,
    trade_count NUMERIC,
    avg_price NUMERIC,
    trade_volume NUMERIC,
    trade_amount NUMERIC,
    source_name TEXT,
    collected_at TIMESTAMP NOT NULL,
    raw_payload JSONB
);

13.6 feature_smp_hourly

CREATE TABLE feature_smp_hourly (
    area TEXT NOT NULL,
    target_time TIMESTAMP NOT NULL,
    target_smp NUMERIC,
    hour INTEGER,
    day_of_week INTEGER,
    is_weekend BOOLEAN,
    is_holiday BOOLEAN,
    month INTEGER,
    season TEXT,
    demand_forecast NUMERIC,
    demand_lag_24h NUMERIC,
    demand_lag_168h NUMERIC,
    smp_lag_1h NUMERIC,
    smp_lag_24h NUMERIC,
    smp_lag_168h NUMERIC,
    smp_rolling_24h_mean NUMERIC,
    smp_rolling_7d_mean NUMERIC,
    temperature NUMERIC,
    humidity NUMERIC,
    cdd NUMERIC,
    hdd NUMERIC,
    gen_gas_share NUMERIC,
    gen_coal_share NUMERIC,
    gen_nuclear_share NUMERIC,
    gen_renewable_share NUMERIC,
    fuel_cost_lng_lag_1m NUMERIC,
    fuel_cost_coal_lag_1m NUMERIC,
    fuel_cost_oil_lag_1m NUMERIC,
    created_at TIMESTAMP NOT NULL,
    PRIMARY KEY(area, target_time)
);

⸻

14. ETL 구현 순서

Phase 1: Source Connection

1. 공공데이터포털 API key 환경변수 설정
2. KPX SMP + 수요예측 API 연결
3. 응답 JSON/XML 저장
4. raw table 저장
5. CSV snapshot 저장

완료 조건:

최근 7일 SMP 데이터 수집 가능
육지/제주 구분 확인
시간 컬럼 파싱 확인

⸻

Phase 2: Historical Backfill

1. 과거 기간 지정
2. 월 단위 또는 일 단위 반복 호출
3. rate limit 대응
4. 실패 요청 retry
5. 수집 로그 저장

완료 조건:

최소 2년 이상 hourly SMP 확보
결측률 리포트 생성

⸻

Phase 3: Feature Build

1. hourly SMP 정렬
2. lag feature 생성
3. rolling feature 생성
4. 기상 데이터 결합
5. 발전량 데이터 결합
6. 연료비 월간 데이터 결합
7. leakage check 수행

완료 조건:

feature_smp_hourly 생성
target_time 기준 미래 데이터 미사용 검증 통과

⸻

Phase 4: Baseline Modeling

1. naive lag_24h 모델
2. seasonal lag_168h 모델
3. Ridge 모델
4. LightGBM 모델
5. validation/test metrics 저장

완료 조건:

LightGBM MAE < seasonal naive MAE
metrics JSON 저장
feature importance 저장

⸻

Phase 5: Settlement / REC Extension

1. 월별 정산단가 수집
2. 월별 feature table 생성
3. 월별 정산단가 예측 baseline 생성
4. REC 현물시장 정보 수집
5. REC 월별 거래현황 수집
6. REC 월별 가격 예측 baseline 생성

완료 조건:

smp_model, settlement_model, rec_model 각각 독립 실행 가능

⸻

15. API Collector 구현 규칙

15.1 모든 collector 공통 규칙

class BaseCollector:
    def fetch(self, start_date, end_date):
        pass
    def parse(self, response):
        pass
    def validate_schema(self, df):
        pass
    def save_raw(self, df):
        pass
    def save_snapshot(self, df):
        pass

15.2 raw response 저장 필수

API 응답은 정제 데이터만 저장하지 말고 원문도 저장한다.

data/raw/kpx/smp/YYYY/MM/DD/response.json

이유:

API 스키마 변경 추적
값 검증
재파싱 가능

15.3 API 변경 대응

공공데이터 API는 스키마 또는 URL이 바뀔 수 있다. 실제로 연료원별 SMP 결정 횟수 API는 2024년 8월 20일부터 제주/육지 구분 제공으로 변경되었고 areaNm 항목이 추가되었다.  ￼

따라서 collector는 다음을 지원해야 한다.

schema_version
required_columns
optional_columns
unknown_columns logging

⸻

16. Hallucination 방지 규칙

Claude Code는 다음 규칙을 따라야 한다.

16.1 공식 출처 외 변수명 단정 금지

API 응답 필드명은 반드시 실제 응답을 보고 확정한다.

예시:

금지: "필드는 smpPrice일 것이다"
허용: "raw response를 확인한 뒤 컬럼 매핑 파일에 기록한다"

16.2 데이터 설명과 실제 응답 분리

문서 설명에는 존재하지만 실제 응답에 없을 수 있다.

따라서 다음 파일을 둔다.

src/config/source_schema_registry.yaml

예시:

kpx_smp_day_ahead:
  official_name: "한국전력거래소_계통한계가격 및 수요예측(하루전 발전계획용)"
  expected_frequency: "hourly"
  expected_regions: ["육지", "제주"]
  verified_columns:
    - TBD_AFTER_FIRST_RESPONSE
  source_fact_checked: true

16.3 불확실한 값은 TBD 처리

TBD_API_FIELD_NAME
TBD_DATE_RANGE
TBD_UNIT
TBD_UPDATE_TIME

16.4 README에 출처와 한계 명시

모든 데이터셋에 대해 다음을 작성한다.

source
official_description
frequency
unit
known_limitations
last_verified_at

⸻

17. 품질 검증 체크리스트

17.1 시간 정렬 검증

SMP trade_hour가 1~24인지 확인
interval_end_time 생성 확인
중복 timestamp 확인
timezone Asia/Seoul 적용
DST 없음 확인

17.2 단위 검증

SMP 단위 = KRW/kWh
정산단가 단위 = KRW/kWh
발전량 단위 = MW
REC 가격 단위 = KRW/REC
REC 거래금액 단위 확인 필요

17.3 결측 검증

hourly SMP 하루 24개 존재 여부
발전량 5분 데이터 하루 288개 존재 여부
월간 연료비 월 1개 존재 여부
REC 비거래일 처리 여부

17.4 이상치 검증

SMP < 0 여부
SMP 극단값 percentile 확인
발전량 음수 여부
REC 가격 0 이하 여부

⸻

18. 추천 실행 명령어

# 1. 환경 설정
cp .env.example .env
# 2. SMP 데이터 수집
python -m src.pipelines.collect_all --source kpx_smp --start 2024-01-01 --end 2024-12-31
# 3. 발전량 데이터 수집
python -m src.pipelines.collect_all --source kpx_generation --start 2024-01-01 --end 2024-12-31
# 4. 연료비 데이터 수집
python -m src.pipelines.collect_all --source kpx_fuel_cost --start 2020-01 --end 2024-12
# 5. feature 생성
python -m src.pipelines.build_features --target smp_hourly --area mainland
# 6. baseline 학습
python -m src.pipelines.train --model naive --target smp_hourly
# 7. LightGBM 학습
python -m src.pipelines.train --model lightgbm --target smp_hourly
# 8. 평가
python -m src.pipelines.evaluate --target smp_hourly

⸻

19. Claude Code 작업 지시문

아래 지시문을 Claude Code에 그대로 넣으면 된다.

You are implementing a Korean electricity market price forecasting project.
Read Plan.md carefully and implement the project step by step.
Priority:
1. Build reliable data collection and raw snapshot storage first.
2. Do not invent API field names. Inspect actual API responses and create schema mapping files.
3. Implement KPX SMP + demand forecast collector first.
4. Implement feature generation with strict no-future-leakage rules.
5. Implement naive, seasonal naive, Ridge, and LightGBM models.
6. Add tests for time alignment, duplicate timestamps, missing hourly records, and leakage.
7. Keep all source-specific assumptions in YAML config files.
8. Every collector must save raw response, parsed dataframe, collection timestamp, and source metadata.
9. Do not implement REC and settlement models until SMP pipeline is stable.
10. Any uncertain source field or unit must be marked as TBD and logged.
Use Python.
Prefer pandas, polars optional, duckdb or PostgreSQL, scikit-learn, lightgbm, pydantic, pytest, python-dotenv.

⸻

20. 초기 개발 우선순위

가장 먼저 할 일은 아래 5개다.

P0-1. KPX SMP + 수요예측 API 연결
P0-2. raw response 저장
P0-3. hourly SMP dataframe 생성
P0-4. lag_24h / lag_168h baseline 모델
P0-5. LightGBM 모델

그다음 확장한다.

P1-1. 발전원별 발전량 결합
P1-2. 월간 연료비 결합
P1-3. 기상 데이터 결합
P1-4. 정산단가 월별 예측
P1-5. REC 월별 예측

⸻

21. 최종 산출물

1. 전력시장 가격 데이터 수집 파이프라인
2. SMP 시간별/일별 예측 모델
3. 정산단가 월별 예측 모델
4. REC 가격/거래지표 예측 모델
5. 변수 중요도 분석
6. 예측 결과 CSV
7. 모델 성능 리포트
8. 데이터 출처 및 한계 문서
9. 메인 프로젝트 연동용 prediction API 또는 CSV export

⸻

22. 이번 프로젝트에서 절대 하면 안 되는 것

1. 소비자 전기요금을 SMP와 동일한 가격으로 취급하지 말 것
2. 육지 SMP와 제주 SMP를 무작정 합치지 말 것
3. 정산단가를 실시간 가격처럼 다루지 말 것
4. REC 가격을 매일 존재하는 일반 시계열처럼 가정하지 말 것
5. API 필드명을 문서 확인 없이 상상해서 쓰지 말 것
6. 미래 발전량, 미래 SMP 결정 횟수, 미래 연료비를 예측 feature로 넣지 말 것
7. random train/test split 사용하지 말 것
8. 수정정산 가능성이 있는 데이터를 버전 없이 덮어쓰지 말 것

⸻

23. 현재 기준 가장 안전한 프로젝트 제목

전력시장 가격 변수 예측을 위한 SMP·정산단가·REC 데이터 수집 및 시계열 예측 모듈 개발

조금 더 연구형 제목:

전력수요·발전믹스·연료비 기반 한국 전력시장 가격 변수 예측 모델 개발

코드 프로젝트명:

kpx-price-forecast

⸻

24. 결론

초기 구현은 SMP 예측에 집중하는 것이 가장 안전하다.

추천 MVP는 다음이다.

Target:
- 육지 시간별 SMP
- 육지 일평균 SMP
Features:
- 과거 SMP lag
- 수요예측
- 시간/요일/계절
- 기상
- 발전믹스
- 월간 연료비 lag
Models:
- lag_24h naive
- lag_168h seasonal naive
- Ridge
- LightGBM
Evaluation:
- MAE
- RMSE
- MAPE
- spike recall

정산단가와 REC는 SMP 파이프라인이 안정화된 뒤 확장한다. 이 순서가 데이터 수집 난이도, 모델 검증 가능성, 메인 프로젝트 연동성을 모두 고려했을 때 가장 안정적이다.