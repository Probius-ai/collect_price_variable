"use client";

import {
  ComposedChart,
  Bar,
  Line,
  XAxis,
  YAxis,
  Tooltip,
  Legend,
  CartesianGrid,
  ResponsiveContainer,
  Cell,
} from "recharts";
import type { HourlyForecastResponse } from "@/lib/api";

const BAND_COLOURS: Record<string, string> = {
  "야간/심야": "#94a3b8",        // slate
  "이행기": "#fbbf24",            // amber
  "주간 저가 (태양광 흡수)": "#10b981", // emerald (solar window)
  "저녁 피크": "#ef4444",         // red (LNG peak)
};

interface ChartPoint {
  hour: string;
  smp: number;
  cf: number;
  band: string;
}

export function HourlyChart({ data }: { data: HourlyForecastResponse }) {
  const chartData: ChartPoint[] = data.points.map((p) => ({
    hour: `${p.hour.toString().padStart(2, "0")}h`,
    smp: p.predicted_smp_krw_per_kwh,
    cf: p.solar_capacity_factor * 100, // percentage for readability
    band: p.band,
  }));

  // Stats for the summary line
  const max = Math.max(...chartData.map((p) => p.smp));
  const min = Math.min(...chartData.map((p) => p.smp));
  const peakHour = chartData.find((p) => p.smp === max)?.hour;
  const troughHour = chartData.find((p) => p.smp === min)?.hour;

  return (
    <div className="space-y-4">
      <div className="grid grid-cols-2 gap-3 sm:grid-cols-4">
        <Stat
          label="일평균 SMP"
          value={`${data.daily_mean_krw_per_kwh.toFixed(2)} 원/kWh`}
          tone="default"
        />
        <Stat
          label={`주간 최저 (${troughHour})`}
          value={`${min.toFixed(2)} 원/kWh`}
          tone="good"
        />
        <Stat
          label={`저녁 피크 (${peakHour})`}
          value={`${max.toFixed(2)} 원/kWh`}
          tone="bad"
        />
        <Stat
          label="피크-저점 스프레드"
          value={`${(max - min).toFixed(2)} 원/kWh`}
          tone="default"
        />
      </div>

      <div className="rounded-lg border border-slate-200 bg-white p-4 shadow-sm">
        <ResponsiveContainer width="100%" height={360}>
          <ComposedChart
            data={chartData}
            margin={{ top: 16, right: 24, left: 0, bottom: 8 }}
          >
            <CartesianGrid stroke="#e2e8f0" strokeDasharray="3 3" />
            <XAxis
              dataKey="hour"
              tick={{ fontSize: 12 }}
              stroke="#64748b"
              interval={1}
            />
            <YAxis
              yAxisId="left"
              tick={{ fontSize: 12 }}
              stroke="#64748b"
              label={{
                value: "SMP (원/kWh)",
                angle: -90,
                position: "insideLeft",
                style: { fontSize: 11, fill: "#475569" },
              }}
            />
            <YAxis
              yAxisId="right"
              orientation="right"
              tick={{ fontSize: 12 }}
              stroke="#0ea5e9"
              label={{
                value: "Solar CF (%)",
                angle: 90,
                position: "insideRight",
                style: { fontSize: 11, fill: "#0369a1" },
              }}
            />
            <Tooltip
              formatter={(value: number, name: string) => {
                if (name === "smp") return [`${value.toFixed(2)} 원/kWh`, "SMP"];
                if (name === "cf") return [`${value.toFixed(0)} %`, "Solar CF"];
                return [value, name];
              }}
              labelStyle={{ color: "#0f172a", fontWeight: 600 }}
            />
            <Legend />
            <Bar
              yAxisId="left"
              dataKey="smp"
              name="시간대별 예상 SMP"
              radius={[3, 3, 0, 0]}
            >
              {chartData.map((p, i) => (
                <Cell key={i} fill={BAND_COLOURS[p.band] ?? "#94a3b8"} />
              ))}
            </Bar>
            <Line
              yAxisId="right"
              dataKey="cf"
              type="monotone"
              stroke="#0ea5e9"
              strokeWidth={2}
              dot={false}
              name="태양광 CF"
            />
          </ComposedChart>
        </ResponsiveContainer>
        <div className="mt-3 flex flex-wrap gap-3 text-xs">
          {Object.entries(BAND_COLOURS).map(([band, colour]) => (
            <span key={band} className="inline-flex items-center gap-1.5">
              <span
                className="h-3 w-3 rounded-sm"
                style={{ backgroundColor: colour }}
              />
              <span className="text-slate-700">{band}</span>
            </span>
          ))}
        </div>
      </div>

      <div className="rounded-md border border-slate-200 bg-slate-50 p-4 text-xs leading-relaxed text-slate-600">
        <p className="mb-1 font-semibold text-slate-800">산정 방식</p>
        <p>{data.methodology}</p>
        <p className="mt-2 text-amber-700">⚠️ {data.caveat}</p>
      </div>
    </div>
  );
}

function Stat({
  label,
  value,
  tone,
}: {
  label: string;
  value: string;
  tone: "default" | "good" | "bad";
}) {
  const toneClass = {
    default: "border-slate-200 bg-white text-slate-900",
    good: "border-emerald-200 bg-emerald-50 text-emerald-900",
    bad: "border-rose-200 bg-rose-50 text-rose-900",
  }[tone];
  return (
    <div className={`rounded-md border p-3 shadow-sm ${toneClass}`}>
      <p className="text-xs uppercase tracking-wider opacity-70">{label}</p>
      <p className="mt-1 font-mono text-lg font-bold">{value}</p>
    </div>
  );
}
