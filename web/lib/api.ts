// API client for the FastAPI backend at /api/*.
//
// Base URL falls back to localhost:8000 for dev; in prod set
// NEXT_PUBLIC_API_BASE in `web/.env.local` or via the deploy config.

const API_BASE =
  process.env.NEXT_PUBLIC_API_BASE ?? "http://localhost:8000";

export interface ModelRow {
  version: string;
  model: string;
  data_cutoff_month: string;
  evaluation_mode: string;
  n_train: number;
  n_test: number;
  mae: number | null;
  rmse: number | null;
  mape: number | null;
  r2: number | null;
  directional_accuracy: number | null;
  improvement_vs_persistence: number | null;
  registry_status: string;
  mlflow_run_id: string | null;
  skipped: boolean;
  skip_reason: string | null;
}

export interface Recommendation {
  recommended_historical: ModelRow | null;
  latest_candidate: ModelRow | null;
  persistence_baseline_v5_mae: number | null;
  improvement_vs_persistence_mae: number | null;
  rationale: string;
}

export interface ComparisonResponse {
  generated_at: string;
  n_rows: number;
  rows: ModelRow[];
}

export interface RetrainStatus {
  state: "idle" | "running" | "completed";
  pid: number | null;
  started_at: string | null;
  log_path: string | null;
  mlflow_uri: string | null;
}

export interface SolarIntegrationStatus {
  resource: string;
  path: string;
  present: boolean;
  kind: string;
}

async function fetchJson<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(`${API_BASE}${path}`, {
    ...init,
    // Don't cache during dev — comparison.csv changes after each retrain.
    cache: "no-store",
  });
  if (!res.ok) {
    const body = await res.text().catch(() => "");
    throw new Error(`API ${path} → ${res.status}: ${body.slice(0, 200)}`);
  }
  return (await res.json()) as T;
}

export const api = {
  recommendation: () => fetchJson<Recommendation>("/api/models/recommendation"),
  comparison: () => fetchJson<ComparisonResponse>("/api/models/comparison"),
  retrainStatus: () => fetchJson<RetrainStatus>("/api/retrain/status"),
  triggerRetrain: () =>
    fetchJson<RetrainStatus>("/api/retrain", { method: "POST" }),
  solarIntegration: () =>
    fetchJson<SolarIntegrationStatus[]>("/api/solar/integration"),
};

export { API_BASE };
