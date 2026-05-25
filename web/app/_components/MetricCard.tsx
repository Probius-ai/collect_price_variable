type Tone = "default" | "recommend" | "candidate" | "good" | "bad";

const TONE_CLASSES: Record<Tone, string> = {
  default: "border-slate-200 bg-white",
  recommend: "border-emerald-300 bg-emerald-50",
  candidate: "border-sky-300 bg-sky-50",
  good: "border-emerald-200 bg-emerald-50",
  bad: "border-rose-200 bg-rose-50",
};

const TONE_VALUE_COLOR: Record<Tone, string> = {
  default: "text-slate-900",
  recommend: "text-emerald-900",
  candidate: "text-sky-900",
  good: "text-emerald-800",
  bad: "text-rose-800",
};

export function MetricCard({
  label,
  value,
  sub,
  tone = "default",
}: {
  label: string;
  value: string;
  sub?: string;
  tone?: Tone;
}) {
  return (
    <div
      className={`rounded-lg border p-4 shadow-sm transition-shadow hover:shadow-md ${TONE_CLASSES[tone]}`}
    >
      <p className="text-xs font-medium uppercase tracking-wider text-slate-500">
        {label}
      </p>
      <p className={`mt-1 text-2xl font-bold ${TONE_VALUE_COLOR[tone]}`}>
        {value}
      </p>
      {sub && <p className="mt-1 text-xs text-slate-500">{sub}</p>}
    </div>
  );
}
