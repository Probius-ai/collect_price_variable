import type { SolarIntegrationStatus } from "@/lib/api";

export function SolarStatus({
  resources,
}: {
  resources: SolarIntegrationStatus[];
}) {
  if (!resources.length) {
    return (
      <p className="text-sm text-slate-500">
        외부 모델 상태를 가져올 수 없습니다.
      </p>
    );
  }
  return (
    <div className="grid grid-cols-1 gap-3 sm:grid-cols-2">
      {resources.map((r) => (
        <div
          key={r.resource}
          className={`rounded-md border p-4 ${
            r.present
              ? "border-emerald-200 bg-emerald-50/40"
              : "border-slate-200 bg-slate-50"
          }`}
        >
          <div className="flex items-start justify-between gap-3">
            <div>
              <p className="font-semibold text-slate-900">{r.resource}</p>
              <p className="mt-1 font-mono text-xs text-slate-500">{r.path}</p>
            </div>
            <span
              className={`inline-flex shrink-0 items-center rounded-full px-2 py-0.5 text-xs font-medium ${
                r.present
                  ? "bg-emerald-100 text-emerald-800"
                  : "bg-slate-200 text-slate-700"
              }`}
            >
              {r.present ? "✓ 있음" : "— 없음"}
            </span>
          </div>
          <p className="mt-2 text-xs uppercase tracking-wider text-slate-500">
            {r.kind}
          </p>
        </div>
      ))}
    </div>
  );
}
