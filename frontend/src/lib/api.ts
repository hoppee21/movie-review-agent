import {
  normalizeCapabilities,
  normalizeHistory,
  normalizeReport,
  normalizeSession,
  object,
  parseSessionEvent,
} from "./protocol.ts";
import type { ReplyBody, Session } from "./protocol.ts";

const base = (import.meta.env?.VITE_API_BASE_URL || "").replace(/\/$/, "") + "/api/v1";

export class ApiError extends Error {
  readonly status: number;

  constructor(message: string, status: number) {
    super(message);
    this.status = status;
  }
}

async function request(path: string, method = "GET", body?: unknown, signal?: AbortSignal): Promise<unknown> {
  const response = await fetch(base + path, {
    method,
    headers: body === undefined ? undefined : { "Content-Type": "application/json" },
    body: body === undefined ? undefined : JSON.stringify(body),
    signal,
  });
  if (!response.ok) {
    const data = object(await response.json().catch(() => ({})));
    const message = object(data.error).message;
    throw new ApiError(
      typeof message === "string" ? message : "请求未完成，请稍后重试。",
      response.status,
    );
  }
  return response.status === 204 ? null : response.json();
}

export const api = {
  async capabilities() {
    return normalizeCapabilities(await request("/capabilities"));
  },
  async list() {
    return normalizeHistory(await request("/sessions"));
  },
  async create(model: string) {
    return normalizeSession(await request("/sessions", "POST", { model }));
  },
  async get(id: string, signal?: AbortSignal) {
    return normalizeSession(await request(`/sessions/${id}`, "GET", undefined, signal));
  },
  async delete(id: string) {
    await request(`/sessions/${id}`, "DELETE");
  },
  async start(id: string, query: string, request_id: string) {
    return normalizeSession(await request(`/sessions/${id}/runs`, "POST", { query, request_id }));
  },
  async reply(id: string, runId: string, body: ReplyBody) {
    return normalizeSession(await request(`/sessions/${id}/runs/${runId}/reply`, "POST", body));
  },
  async cancel(id: string, runId: string) {
    return normalizeSession(await request(`/sessions/${id}/runs/${runId}/cancel`, "POST"));
  },
  async retry(id: string, runId: string) {
    return normalizeSession(await request(`/sessions/${id}/runs/${runId}/retry`, "POST"));
  },
  async example() {
    const data = object(await request("/example-report"));
    return { query: String(data.query || ""), report: normalizeReport(data.report) };
  },
  watch(
    id: string,
    revision: number,
    onSession: (session: Session) => void,
    onConnection: (connected: boolean) => void,
    onMissing: () => void,
  ) {
    // The stream and its polling fallback share one lifetime. Closing either
    // prevents late responses from changing a newly selected conversation.
    const source = new EventSource(`${base}/sessions/${id}/events?after=${revision}`);
    const controller = new AbortController();
    let closed = false;
    let connected = false;
    let polling = false;

    source.onopen = () => {
      if (closed) return;
      connected = true;
      onConnection(true);
    };
    source.onerror = () => { connected = false; };
    source.onmessage = event => {
      if (closed) return;
      try {
        const session = parseSessionEvent(JSON.parse(event.data));
        if (session) onSession(session);
      } catch {
        connected = false;
      }
    };

    const timer = setInterval(async () => {
      if (closed || connected || polling) return;
      polling = true;
      try {
        const session = await api.get(id, controller.signal);
        if (closed) return;
        onSession(session);
        onConnection(true);
      } catch (error) {
        if (closed) return;
        if (error instanceof ApiError && error.status === 404) {
          close();
          onMissing();
        } else {
          onConnection(false);
        }
      } finally {
        polling = false;
      }
    }, 3000);

    function close() {
      closed = true;
      source.close();
      controller.abort();
      clearInterval(timer);
    }
    return close;
  },
};
