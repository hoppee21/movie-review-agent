/** Stable presentation models. No graph nodes or RAG implementation types. */
export type Choice = { id: string; label: string; description: string };
export type Clarification = { id: string; question: string; choices: Choice[] };
export type Source = {
  id: string; platform: string; title: string; quote: string; full_text: string;
  stance: string; reason: string | null; url: string | null; rating: number | null;
};
export type Section = {
  id: string; title: string; body: string; citations: string[];
  counter_citations: string[]; note: string | null;
};
export type Report = {
  title: string; summary: string; citations: string[]; sections: Section[]; sources: Source[];
  source_counts: Record<string, number>; coverage_sufficient: boolean | null;
  gaps: string[]; limitations: string[]; links: { title: string; url: string }[];
};
export type Message = {
  id: string; role: "user" | "assistant"; text: string; created_at: string;
  report: Report | null; clarification: Clarification | null;
};
export type Stage = { id: string; label: string; status: string; elapsed_seconds: number | null };
export type Run = {
  id: string; query: string; status: string; error: string | null; stages: Stage[];
  clarification: Clarification | null; started_at: string; finished_at: string | null;
};
export type Session = {
  id: string; revision: number; title: string; model: string; created_at: string; updated_at: string;
  messages: Message[]; run: Run | null;
};
export type SessionItem = { id: string; title: string; updated_at: string; status: string };
export type Capabilities = {
  default_model: string; features: string[]; example_available: boolean; max_query_chars: number;
};

type ObjectValue = Record<string, unknown>;
export const object = (value: unknown): ObjectValue => value && typeof value === "object" && !Array.isArray(value) ? value as ObjectValue : {};
const text = (value: unknown, fallback = "") => typeof value === "string" ? value : fallback;
const number = (value: unknown, fallback = 0) => typeof value === "number" && Number.isFinite(value) ? value : fallback;
const items = (value: unknown): unknown[] => Array.isArray(value) ? value : [];
const strings = (value: unknown) => items(value).filter((item): item is string => typeof item === "string");
export const safeUrl = (value: unknown): string | null => {
  if (typeof value !== "string") return null;
  try { const url = new URL(value); return ["https:", "http:"].includes(url.protocol) ? url.href : null; } catch { return null; }
};

export function checkVersion(value: unknown) {
  const version = text(object(value).protocol_version, "1.0");
  if (version.split(".")[0] !== "1") throw new Error("服务接口已升级，请更新前端后重试。");
}

export function normalizeClarification(value: unknown): Clarification | null {
  const input = object(value);
  if (!text(input.id)) return null;
  return { id: text(input.id), question: text(input.question, "请补充一些信息。"), choices: items(input.choices).map(item => {
    const choice = object(item);
    return { id: text(choice.id), label: text(choice.label), description: text(choice.description) };
  }).filter(choice => choice.id && choice.label) };
}

export function normalizeReport(value: unknown): Report | null {
  if (!value || typeof value !== "object") return null;
  const input = object(value);
  const sources = items(input.sources).map((item, index) => {
    const source = object(item);
    return { id: text(source.id, `source-${index}`), platform: text(source.platform, "来源"), title: text(source.title, "原始评论"),
      quote: text(source.quote), full_text: text(source.full_text, text(source.quote)), stance: text(source.stance, "unclear"),
      reason: text(source.reason) || null, url: safeUrl(source.url), rating: typeof source.rating === "number" ? source.rating : null };
  });
  const validIds = new Set(sources.map(source => source.id));
  const citations = (value: unknown) => strings(value).filter(id => validIds.has(id));
  return {
    title: text(input.title, "分析报告"), summary: text(input.summary), sources,
    citations: citations(input.citations), sections: items(input.sections).map((item, index) => {
      const section = object(item);
      return { id: text(section.id, `section-${index}`), title: text(section.title, "分析"), body: text(section.body),
        citations: citations(section.citations), counter_citations: citations(section.counter_citations), note: text(section.note) || null };
    }),
    source_counts: Object.fromEntries(Object.entries(object(input.source_counts)).filter(([, value]) => typeof value === "number" && Number.isFinite(value))) as Record<string, number>,
    coverage_sufficient: typeof input.coverage_sufficient === "boolean" ? input.coverage_sufficient : null,
    gaps: strings(input.gaps), limitations: strings(input.limitations),
    links: items(input.links).map(item => { const link = object(item); return { title: text(link.title, "来源"), url: safeUrl(link.url) }; })
      .filter((link): link is { title: string; url: string } => Boolean(link.url)),
  };
}

export function normalizeSession(value: unknown): Session {
  checkVersion(value);
  const input = object(value);
  if (!text(input.id)) throw new Error("服务返回的会话格式不完整，请重试。");
  const rawRun = object(input.run);
  return { id: text(input.id), revision: number(input.revision), title: text(input.title, "新的分析"), model: text(input.model),
    created_at: text(input.created_at), updated_at: text(input.updated_at), messages: items(input.messages).map((item, index) => {
      const message = object(item);
      return { id: text(message.id, `message-${index}`), role: message.role === "user" ? "user" : "assistant", text: text(message.text),
        created_at: text(message.created_at), report: normalizeReport(message.report), clarification: normalizeClarification(message.clarification) };
    }),
    run: text(rawRun.id) ? { id: text(rawRun.id), query: text(rawRun.query), status: text(rawRun.status), error: text(rawRun.error) || null,
      started_at: text(rawRun.started_at), finished_at: text(rawRun.finished_at) || null, clarification: normalizeClarification(rawRun.clarification),
      stages: items(rawRun.stages).map((item, index) => { const stage = object(item); return { id: text(stage.id, `stage-${index}`),
        label: text(stage.label), status: text(stage.status), elapsed_seconds: typeof stage.elapsed_seconds === "number" ? stage.elapsed_seconds : null }; }),
    } : null };
}

export function applySnapshot(current: Session | null, incoming: Session): Session {
  return current?.id === incoming.id && current.revision > incoming.revision ? current : incoming;
}

export function parseSessionEvent(value: unknown): Session | null {
  const input = object(value);
  if (input.type !== "session.updated") return null;
  checkVersion(input);
  return normalizeSession(input.session);
}

export const isBusy = (run: Run | null | undefined) => Boolean(run && ["queued", "running", "waiting_for_input"].includes(run.status));
