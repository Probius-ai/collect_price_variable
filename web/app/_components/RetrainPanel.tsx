"use client";

import { useEffect, useState } from "react";
import { api, type RetrainStatus } from "@/lib/api";

const POLL_INTERVAL_MS = 5000; // 5s while running, paused otherwise

export function RetrainPanel() {
  const [status, setStatus] = useState<RetrainStatus | null>(null);
  const [submitting, setSubmitting] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [lastChecked, setLastChecked] = useState<Date | null>(null);

  // Initial load + polling while running
  useEffect(() => {
    let cancelled = false;
    let timer: ReturnType<typeof setTimeout> | undefined;

    const tick = async () => {
      try {
        const s = await api.retrainStatus();
        if (cancelled) return;
        setStatus(s);
        setLastChecked(new Date());
        // Only keep polling while a run is actually in flight
        if (s.state === "running") {
          timer = setTimeout(tick, POLL_INTERVAL_MS);
        }
      } catch (e) {
        if (cancelled) return;
        setError(e instanceof Error ? e.message : String(e));
      }
    };

    tick();
    return () => {
      cancelled = true;
      if (timer) clearTimeout(timer);
    };
  }, []);

  const handleClick = async () => {
    setError(null);
    setSubmitting(true);
    try {
      const s = await api.triggerRetrain();
      setStatus(s);
      setLastChecked(new Date());
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setSubmitting(false);
    }
  };

  const isRunning = status?.state === "running";
  const isCompleted = status?.state === "completed";

  return (
    <div className="rounded-lg border border-slate-200 bg-white p-6 shadow-sm">
      <div className="flex flex-col gap-4 sm:flex-row sm:items-center sm:justify-between">
        <div className="space-y-1">
          <p className="font-semibold text-slate-900">
            ℹ️ 모델은 6시간마다 단기 예보 정보를 통해 자동 재학습됩니다.
          </p>
          <p className="text-sm text-slate-600">
            수동으로 즉시 새 학습을 돌리려면 오른쪽 버튼을 누르세요.
            백그라운드에서 v1~v5 smoke test가 실행되며 MLflow에 자동
            기록됩니다 (보통 5~10분 소요).
          </p>
        </div>
        <button
          type="button"
          onClick={handleClick}
          disabled={submitting || isRunning}
          className={`inline-flex shrink-0 items-center justify-center rounded-md px-5 py-3 text-sm font-semibold shadow-sm transition-colors ${
            submitting || isRunning
              ? "cursor-not-allowed bg-slate-300 text-slate-500"
              : "bg-blue-600 text-white hover:bg-blue-700"
          }`}
        >
          {submitting
            ? "시작 중..."
            : isRunning
              ? "🟡 진행 중..."
              : "🔄 강제 재학습"}
        </button>
      </div>

      {/* Status display */}
      {status && status.state !== "idle" && (
        <div
          className={`mt-4 rounded-md border p-4 text-sm ${
            isRunning
              ? "border-amber-200 bg-amber-50 text-amber-900"
              : "border-emerald-200 bg-emerald-50 text-emerald-900"
          }`}
        >
          <div className="flex items-start gap-2">
            <span className="font-bold">
              {isRunning ? "🟡 재학습 진행 중" : "🟢 최근 재학습 완료"}
            </span>
          </div>
          <dl className="mt-2 grid grid-cols-1 gap-x-6 gap-y-1 text-xs sm:grid-cols-2">
            {status.pid && (
              <div>
                <dt className="font-medium opacity-70">PID</dt>
                <dd className="font-mono">{status.pid}</dd>
              </div>
            )}
            {status.started_at && (
              <div>
                <dt className="font-medium opacity-70">시작 (UTC)</dt>
                <dd className="font-mono">
                  {status.started_at.slice(0, 19).replace("T", " ")}
                </dd>
              </div>
            )}
            {status.log_path && (
              <div>
                <dt className="font-medium opacity-70">로그</dt>
                <dd className="font-mono">{status.log_path}</dd>
              </div>
            )}
            {status.mlflow_uri && (
              <div>
                <dt className="font-medium opacity-70">MLflow</dt>
                <dd>
                  <a
                    href={status.mlflow_uri}
                    target="_blank"
                    rel="noopener noreferrer"
                    className="font-mono underline decoration-dotted hover:decoration-solid"
                  >
                    {status.mlflow_uri}
                  </a>
                </dd>
              </div>
            )}
          </dl>
          {isCompleted && (
            <p className="mt-2 text-xs opacity-80">
              새 50개 run이 MLflow의 <code>kpx-smp-monthly</code> experiment에
              추가되었습니다. 페이지를 새로고침하면 위 비교표가 갱신됩니다.
            </p>
          )}
        </div>
      )}

      {error && (
        <div className="mt-4 rounded-md border border-rose-200 bg-rose-50 p-4 text-sm text-rose-900">
          <p className="font-semibold">재학습 호출 실패</p>
          <p className="mt-1 font-mono text-xs">{error}</p>
        </div>
      )}

      {lastChecked && (
        <p className="mt-3 text-xs text-slate-400">
          마지막 상태 확인: {lastChecked.toLocaleTimeString("ko-KR")}
          {isRunning && ` · 5초마다 자동 새로고침`}
        </p>
      )}
    </div>
  );
}
