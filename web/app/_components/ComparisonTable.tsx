type StabilityRow = {
  model: string;
  v4_holdout_mae: number | null;
  v5_rolling_mae: number | null;
  delta: number | null;
  mean: number | null;
};

function fmt(n: number | null, digits = 3): string {
  if (n === null || n === undefined || Number.isNaN(n)) return "—";
  return n.toFixed(digits);
}

export function ComparisonTable({
  rows,
  recommendedModel,
  latestCandidateModel,
}: {
  rows: StabilityRow[];
  recommendedModel: string | null;
  latestCandidateModel: string | null;
}) {
  return (
    <div className="overflow-x-auto rounded-lg border border-slate-200 bg-white shadow-sm">
      <table className="min-w-full divide-y divide-slate-200 text-sm">
        <thead className="bg-slate-50">
          <tr>
            <th className="px-4 py-2 text-left font-semibold text-slate-700">
              Model
            </th>
            <th className="px-4 py-2 text-right font-semibold text-slate-700">
              v4 holdout MAE
            </th>
            <th className="px-4 py-2 text-right font-semibold text-slate-700">
              v5 rolling MAE
            </th>
            <th className="px-4 py-2 text-right font-semibold text-slate-700">
              Δ (v5 - v4)
            </th>
            <th className="px-4 py-2 text-right font-semibold text-slate-700">
              평균 MAE
            </th>
            <th className="px-4 py-2 text-left font-semibold text-slate-700">
              상태
            </th>
          </tr>
        </thead>
        <tbody className="divide-y divide-slate-100">
          {rows.map((r) => {
            const isRec = r.model === recommendedModel;
            const isCand = r.model === latestCandidateModel;
            const rowBg = isRec
              ? "bg-emerald-50"
              : isCand
                ? "bg-sky-50"
                : "hover:bg-slate-50";
            const deltaTone =
              r.delta !== null
                ? r.delta > 0
                  ? "text-rose-700"
                  : "text-emerald-700"
                : "text-slate-500";
            return (
              <tr key={r.model} className={rowBg}>
                <td className="px-4 py-2 font-mono text-slate-900">
                  {r.model}
                </td>
                <td className="px-4 py-2 text-right tabular-nums">
                  {fmt(r.v4_holdout_mae)}
                </td>
                <td className="px-4 py-2 text-right tabular-nums">
                  {fmt(r.v5_rolling_mae)}
                </td>
                <td
                  className={`px-4 py-2 text-right tabular-nums font-medium ${deltaTone}`}
                >
                  {r.delta !== null
                    ? `${r.delta >= 0 ? "+" : ""}${r.delta.toFixed(3)}`
                    : "—"}
                </td>
                <td className="px-4 py-2 text-right tabular-nums font-semibold">
                  {fmt(r.mean)}
                </td>
                <td className="px-4 py-2 text-xs">
                  {isRec && (
                    <span className="inline-flex items-center rounded-full bg-emerald-100 px-2 py-0.5 font-medium text-emerald-800">
                      🟢 recommended
                    </span>
                  )}
                  {isCand && !isRec && (
                    <span className="inline-flex items-center rounded-full bg-sky-100 px-2 py-0.5 font-medium text-sky-800">
                      🔵 latest_candidate
                    </span>
                  )}
                </td>
              </tr>
            );
          })}
        </tbody>
      </table>
      <p className="border-t border-slate-200 bg-slate-50 px-4 py-2 text-xs text-slate-500">
        🟢 = v1~v4 recommended_historical · 🔵 = v5 latest_candidate. 두 행이
        같은 모델이면 가장 강한 후보입니다.
      </p>
    </div>
  );
}
