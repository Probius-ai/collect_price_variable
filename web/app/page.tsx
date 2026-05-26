import { api } from "@/lib/api";
import { RetrainPanel } from "./_components/RetrainPanel";
import { ComparisonTable } from "./_components/ComparisonTable";
import { MetricCard } from "./_components/MetricCard";
import { SolarStatus } from "./_components/SolarStatus";
import { PriceHero } from "./_components/PriceHero";
import { HourlyChart } from "./_components/HourlyChart";

// Server component — fetches on the server at request time so the first
// paint already has data. The retrain panel is the only piece that
// needs to be a client component (interactive button + polling).

async function safeFetch<T>(fn: () => Promise<T>, fallback: T): Promise<T> {
  try {
    return await fn();
  } catch (e) {
    console.error("API fetch failed:", e);
    return fallback;
  }
}

export const dynamic = "force-dynamic"; // always re-fetch — comparison.csv changes

export default async function HomePage() {
  const [recommendation, comparison, solar, forecast, hourly] =
    await Promise.all([
      safeFetch(api.recommendation, null),
      safeFetch(api.comparison, null),
      safeFetch(api.solarIntegration, []),
      safeFetch(api.forecastNext, null),
      safeFetch(api.forecastHourly, null),
    ]);

  if (!recommendation || !comparison) {
    return (
      <main className="mx-auto max-w-7xl px-6 py-12">
        <h1 className="text-3xl font-bold mb-4">⚡ KPX SMP 모델 선정</h1>
        <div className="rounded-lg border border-amber-300 bg-amber-50 p-6">
          <p className="font-semibold text-amber-900">
            API 백엔드에 연결할 수 없습니다.
          </p>
          <p className="mt-2 text-sm text-amber-800">
            <code className="rounded bg-amber-100 px-1.5 py-0.5">
              uvicorn api.main:app --reload --port 8000
            </code>{" "}
            를 실행한 뒤 새로고침 해주세요. NEXT_PUBLIC_API_BASE 환경변수로
            엔드포인트를 바꿀 수 있습니다.
          </p>
        </div>
      </main>
    );
  }

  const rec = recommendation.recommended_historical;
  const latest = recommendation.latest_candidate;
  const baselineMae = recommendation.persistence_baseline_v5_mae;
  const improvement = recommendation.improvement_vs_persistence_mae;

  // Build stability table (v4 holdout vs v5 rolling)
  const v4 = new Map(
    comparison.rows
      .filter((r) => r.version === "v4" && !r.skipped)
      .map((r) => [r.model, r.mae])
  );
  const v5 = new Map(
    comparison.rows
      .filter((r) => r.version === "v5" && !r.skipped)
      .map((r) => [r.model, r.mae])
  );
  const stability = Array.from(v4.keys())
    .filter((m) => v5.has(m))
    .map((m) => {
      const a = v4.get(m) ?? null;
      const b = v5.get(m) ?? null;
      const delta = a !== null && b !== null ? b - a : null;
      const mean = a !== null && b !== null ? (a + b) / 2 : null;
      return { model: m, v4_holdout_mae: a, v5_rolling_mae: b, delta, mean };
    })
    .sort((x, y) => (x.mean ?? Infinity) - (y.mean ?? Infinity));

  return (
    <main className="mx-auto max-w-7xl px-6 py-10 space-y-10">
      {/* Header */}
      <header className="space-y-2">
        <p className="text-sm font-medium uppercase tracking-wider text-blue-700">
          MLOps · Model Selection
        </p>
        <h1 className="text-3xl font-bold sm:text-4xl">
          ⚡ KPX SMP — 단기 태양광 발전량 예측을 통한 전력 가격 모델 선정
        </h1>
        <p className="max-w-3xl text-base text-slate-600">
          v1~v5 staged-retraining 결과를 한 화면에 모아서{" "}
          <strong>production에 올릴 모델 1개를 고르는</strong> 의사결정 뷰입니다.
          마지막 데이터 갱신:{" "}
          <code className="rounded bg-slate-100 px-1.5 py-0.5 text-xs">
            {new Date(comparison.generated_at).toLocaleString("ko-KR")}
          </code>{" "}
          · 총 {comparison.n_rows}개 (version × model) row.
        </p>
      </header>

      {/* Price hero — the actual KRW/kWh number the selection service exists to produce */}
      <PriceHero forecast={forecast} />

      {/* Hourly profile — disaggregates the monthly forecast to 24 hours via solar shape */}
      {hourly && (
        <section className="space-y-3">
          <h2 className="text-xl font-semibold">
            시간대별 가격대 (태양광 CF × 월간 SMP)
          </h2>
          <div className="rounded-md border border-sky-200 bg-sky-50 p-4 text-sm text-sky-900">
            <p className="font-semibold">
              📌 갱신 주기 ─ 두 신호는 별도로 움직입니다
            </p>
            <ul className="mt-2 list-disc space-y-1 pl-5 leading-relaxed">
              <li>
                <strong>월간 SMP 예측 (108.15 원/kWh)</strong> — 매월 새 데이터가
                입력되면 재학습. 본 페이지의 <em>일평균</em> 기준선이며,
                같은 달 안에서는 값이 바뀌지 않습니다.
              </li>
              <li>
                <strong>시간별 태양광 발전 예측 (CF)</strong> —{" "}
                <em>매 1시간마다</em> 단기 날씨 API를 호출해 갱신. 이 값이
                LNG 화력 호출량을 결정하므로, 같은 일평균 SMP라도 시간대별
                가격 분포가 달라집니다.
              </li>
            </ul>
            <p className="mt-2 text-xs opacity-80">
              따라서 위 차트의 <strong>bar 높이 (SMP)</strong>는 월간 모델이
              결정한 daily-mean에 <strong>line (Solar CF)</strong> 의 시간 패턴을
              곱한 값입니다.
            </p>
          </div>
          <HourlyChart data={hourly} />
        </section>
      )}

      {/* Decision panel */}
      <section className="space-y-3">
        <h2 className="text-xl font-semibold">선정 근거 — 모델 비교</h2>
        <p className="text-sm text-slate-600">
          MAE 기준. 추천 모델은 (1) v1~v4 holdout 평균 우수 + (2) v5 rolling
          검증 안정성을 함께 본 결과입니다.
        </p>
        <div className="grid grid-cols-1 gap-4 sm:grid-cols-2 lg:grid-cols-4">
          <MetricCard
            label="추천 모델 (v1~v4 best)"
            value={rec?.model ?? "—"}
            sub={rec?.version ? `version ${rec.version}` : ""}
            tone="recommend"
          />
          <MetricCard
            label="Test MAE"
            value={
              rec?.mae !== undefined && rec?.mae !== null
                ? rec.mae.toFixed(3)
                : "—"
            }
            sub={`evaluation_mode: ${rec?.evaluation_mode ?? "—"}`}
          />
          <MetricCard
            label="v5 latest_candidate"
            value={latest?.model ?? "—"}
            sub={
              latest?.mae !== undefined && latest?.mae !== null
                ? `${latest.mae.toFixed(3)} MAE`
                : ""
            }
            tone="candidate"
          />
          <MetricCard
            label="vs Persistence (v5)"
            value={
              improvement !== null && improvement !== undefined
                ? `${improvement >= 0 ? "+" : ""}${improvement.toFixed(3)} MAE`
                : "—"
            }
            sub={
              baselineMae !== null && baselineMae !== undefined
                ? `baseline MAE ${baselineMae.toFixed(3)}`
                : ""
            }
            tone={
              improvement !== null &&
              improvement !== undefined &&
              improvement > 0
                ? "good"
                : "bad"
            }
          />
        </div>
        <div className="rounded-md border border-blue-200 bg-blue-50 p-4 text-sm text-blue-900">
          <strong>판단 근거:</strong> {recommendation.rationale}
        </div>
      </section>

      {/* Stability table */}
      <section className="space-y-3">
        <h2 className="text-xl font-semibold">
          후보 모델 안정성 (v4 → v5 변화)
        </h2>
        <p className="text-sm text-slate-600">
          v4는 fixed-holdout (forward labels 있음), v5는 rolling validation
          (future labels 없음). 두 점수 차이가 작을수록{" "}
          <strong>새 데이터에 강건한</strong> 모델입니다.
        </p>
        <ComparisonTable
          rows={stability}
          recommendedModel={rec?.model ?? null}
          latestCandidateModel={latest?.model ?? null}
        />
      </section>

      {/* Why solar matters */}
      <section className="space-y-3">
        <h2 className="text-xl font-semibold">
          🌞 왜 단기 태양광 예측이 SMP 모델 선정에 중요한가
        </h2>
        <div className="rounded-md border border-amber-200 bg-amber-50 p-5 text-sm leading-relaxed text-amber-900 space-y-3">
          <p>
            한국 전력시장의 SMP(System Marginal Price)는 마지막에 호출된{" "}
            <strong>한계 발전기의 변동비</strong>가 결정합니다. 일반적으로
            한계 발전기는 LNG 화력이고, LNG는 가격 변동이 큰 연료입니다.
          </p>
          <pre className="rounded bg-white/60 p-3 font-mono text-xs leading-5 text-slate-800">
{`↑ 단기 태양광 발전량 예측
        ↓
↓ LNG 화력 호출량 (재생에너지가 우선 송전)
        ↓
↓ LNG 수요·도입가 충격 가능성
        ↓
↓ SMP (특히 일간·시간대별)`}
          </pre>
          <p>
            <strong>모델 선정 관점</strong>: 단순 MAE 최저가 아니라 (a) 안정성
            (v5 rolling 강건) + (b) 외부 LNG·날씨 신호에 반응하는 구조를
            우선합니다.
          </p>
        </div>
      </section>

      {/* Solar integration status */}
      <section className="space-y-3">
        <h2 className="text-xl font-semibold">
          외부 모델 인벤토리 (Solar / LNG)
        </h2>
        <SolarStatus resources={solar} />
      </section>

      {/* Retrain panel — client component for the button + polling */}
      <section className="space-y-3">
        <h2 className="text-xl font-semibold">학습 트리거</h2>
        <RetrainPanel />
      </section>

      <footer className="border-t border-slate-200 pt-6 text-xs text-slate-500">
        <p>
          이 페이지는 FastAPI 백엔드(<code>api/main.py</code>)의 read-only
          엔드포인트를 소비합니다. 깊은 분석은 Streamlit dashboard
          (<code>streamlit run dashboard.py</code>) 또는 MLflow UI(:5000)에서
          가능합니다.
        </p>
      </footer>
    </main>
  );
}
