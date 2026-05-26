import type { ForecastResponse } from "@/lib/api";

export function PriceHero({ forecast }: { forecast: ForecastResponse | null }) {
  if (!forecast) {
    return (
      <div className="rounded-2xl border border-amber-300 bg-amber-50 p-8 shadow-sm">
        <p className="text-sm font-semibold text-amber-900">
          가격 예측을 가져올 수 없습니다.
        </p>
        <p className="mt-1 text-xs text-amber-800">
          모델 pickle이 없거나 FastAPI 백엔드(:8000)가 꺼져 있을 수 있습니다.
          아래 강제 재학습 버튼을 눌러 새 모델을 학습하세요.
        </p>
      </div>
    );
  }

  const predicted = forecast.predicted_smp_krw_per_kwh;
  const actual = forecast.most_recent_actual_smp_krw_per_kwh;
  const delta =
    actual !== null && actual !== undefined ? predicted - actual : null;

  return (
    <div className="rounded-2xl border border-blue-200 bg-gradient-to-br from-blue-50 via-white to-sky-50 p-8 shadow-sm">
      <div className="flex flex-col gap-6 sm:flex-row sm:items-end sm:justify-between">
        <div>
          <p className="text-sm font-medium uppercase tracking-wider text-blue-700">
            선정 모델의 출력 — 다음 달 평균 SMP
          </p>
          <p className="mt-1 text-xs text-slate-500">
            <span className="inline-flex items-center rounded-full bg-slate-100 px-2 py-0.5 font-medium text-slate-700">
              📅 한 달에 한 번만 갱신
            </span>{" "}
            · 같은 달 동안은 이 값이 일평균 기준선으로 고정됩니다.
          </p>
          <p className="mt-2 font-mono text-xs text-slate-500">
            target_month: {forecast.target_month} · forecast_origin:{" "}
            {forecast.forecast_origin_month}
          </p>
          <div className="mt-4 flex items-baseline gap-3">
            <span className="text-5xl font-bold text-blue-900 sm:text-6xl">
              {predicted.toFixed(2)}
            </span>
            <span className="text-2xl font-semibold text-blue-700">
              원/kWh
            </span>
          </div>
          {actual !== null && actual !== undefined && (
            <p className="mt-3 text-sm text-slate-600">
              최근 실측 ({forecast.most_recent_actual_month}):{" "}
              <span className="font-mono font-semibold">
                {actual.toFixed(2)} 원/kWh
              </span>
              {delta !== null && (
                <span
                  className={`ml-2 font-medium ${
                    delta >= 0 ? "text-rose-700" : "text-emerald-700"
                  }`}
                >
                  ({delta >= 0 ? "+" : ""}
                  {delta.toFixed(2)})
                </span>
              )}
            </p>
          )}
        </div>
        <div className="space-y-1 text-right text-sm">
          <p className="text-xs uppercase tracking-wider text-slate-500">
            선정된 모델
          </p>
          <p className="text-lg font-bold text-slate-900">
            {forecast.model_name}
          </p>
          <p className="font-mono text-xs text-slate-500">
            v{forecast.version.replace("v", "")} · inference{" "}
            {forecast.inference_seconds.toFixed(2)}s
          </p>
        </div>
      </div>
      <p className="mt-6 text-xs leading-relaxed text-slate-500">{forecast.note}</p>
    </div>
  );
}
