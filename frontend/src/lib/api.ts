import { checkVersion, normalizeReport, normalizeSession, object, parseSessionEvent } from "./protocol";
import type { Capabilities, Session, SessionItem } from "./protocol";

const base = (import.meta.env.VITE_API_BASE_URL || "").replace(/\/$/, "") + "/api/v1";
export class ApiError extends Error {
  status: number;
  constructor(message: string, status: number) { super(message); this.status = status; }
}

async function request(path: string, method = "GET", body?: unknown) {
  const response = await fetch(base + path, { method, headers: body ? { "Content-Type": "application/json" } : undefined,
    body: body ? JSON.stringify(body) : undefined });
  if (!response.ok) {
    const data = object(await response.json().catch(() => ({})));
    throw new ApiError(typeof object(data.error).message === "string" ? object(data.error).message as string : "请求未完成，请稍后重试。", response.status);
  }
  return response.status === 204 ? null : response.json();
}

export const api = {
  async capabilities(): Promise<Capabilities> {
    const value = await request("/capabilities"); checkVersion(value);
    return { default_model: value.default_model || "", features: Array.isArray(value.features) ? value.features : [],
      example_available: value.example_available === true, max_query_chars: value.max_query_chars || 12000 };
  },
  async list(): Promise<SessionItem[]> { const data = await request("/sessions"); return Array.isArray(data.items) ? data.items : []; },
  async create(model: string) { return normalizeSession(await request("/sessions", "POST", { model })); },
  async get(id: string) { return normalizeSession(await request(`/sessions/${id}`)); },
  async delete(id: string) { await request(`/sessions/${id}`, "DELETE"); },
  async start(id: string, query: string, request_id: string) { return normalizeSession(await request(`/sessions/${id}/runs`, "POST", { query, request_id })); },
  async reply(id: string, runId: string, body: { request_id: string; issue_id: string; option_id?: string; text?: string; cancelled?: boolean }) {
    return normalizeSession(await request(`/sessions/${id}/runs/${runId}/reply`, "POST", body));
  },
  async cancel(id: string, runId: string) { return normalizeSession(await request(`/sessions/${id}/runs/${runId}/cancel`, "POST")); },
  async retry(id: string, runId: string) { return normalizeSession(await request(`/sessions/${id}/runs/${runId}/retry`, "POST")); },
  async example() { const data = await request("/example-report"); return { query: String(data.query || ""), report: normalizeReport(data.report) }; },
  watch(id: string, revision: number, onSession: (session: Session) => void, onConnection: (connected: boolean) => void) {
    const source = new EventSource(`${base}/sessions/${id}/events?after=${revision}`);
    source.onopen = () => onConnection(true);
    source.onerror = () => onConnection(false);
    source.onmessage = event => {
      try { const session = parseSessionEvent(JSON.parse(event.data)); if (session) onSession(session); }
      catch { onConnection(false); }
    };
    return () => source.close();
  },
};
