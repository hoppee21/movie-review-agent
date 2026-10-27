import { useCallback, useEffect, useRef, useState } from "react";
import { api } from "./api";
import { applySnapshot, updateHistory } from "./protocol";
import type { Capabilities, ReplyBody, Report, Session, SessionItem } from "./protocol";

function recalled(key: string): string {
  try { return localStorage.getItem(key) || ""; }
  catch { return ""; }
}
const savedModel = () => recalled("movie-evidence:model");
function remember(key: string, value: string): void {
  try { localStorage.setItem(key, value); }
  catch { /* Private browsing can disable storage. */ }
}
const errorText = (error: unknown) => error instanceof Error ? error.message : "连接暂时中断，请稍后重试。";

export function useWorkspace() {
  const [session, setSession] = useState<Session | null>(null);
  const [activeId, setActiveId] = useState<string | null>(null);
  const [history, setHistory] = useState<SessionItem[]>([]);
  const [capabilities, setCapabilities] = useState<Capabilities | null>(null);
  const [model, setModel] = useState(savedModel);
  const [online, setOnline] = useState<boolean | null>(null);
  const [pending, setPending] = useState(false);
  const [error, setError] = useState("");
  const [example, setExample] = useState<{ query: string; report: Report } | null>(null);
  const active = useRef<string | null>(null);
  const submitting = useRef(false);
  const generation = useRef(0);

  const refreshHistory = useCallback(async () => { setHistory(await api.list()); }, []);
  const activate = useCallback((incoming: Session) => {
    active.current = incoming.id;
    setActiveId(incoming.id);
    setSession(incoming);
    setExample(null);
    remember("movie-evidence:session", incoming.id);
  }, []);
  const accept = useCallback((incoming: Session) => {
    if (active.current !== incoming.id) return;
    setSession(current => applySnapshot(current, incoming));
    setHistory(current => updateHistory(current, incoming));
  }, []);

  useEffect(() => {
    let mounted = true;
    void (async () => {
      try {
        const [caps, items] = await Promise.all([api.capabilities(), api.list()]);
        if (!mounted) return;
        setCapabilities(caps);
        setModel(current => current || caps.default_model);
        setHistory(items);
        setOnline(true);
        const remembered = recalled("movie-evidence:session");
        if (remembered && items.some(item => item.id === remembered)) {
          const restored = await api.get(remembered);
          if (mounted && generation.current === 0 && !active.current) activate(restored);
        }
      } catch { if (mounted) setOnline(false); }
    })();
    return () => { mounted = false; };
  }, [activate]);

  useEffect(() => {
    if (!activeId) return;
    return api.watch(activeId, -1, accept, setOnline, () => {
      if (active.current !== activeId) return;
      active.current = null;
      setActiveId(null);
      setSession(null);
      remember("movie-evidence:session", "");
      setError("这个会话已不存在或已被删除，请开始新的分析。");
    });
  }, [activeId, accept]);

  function newConversation() {
    generation.current += 1;
    active.current = null;
    setActiveId(null);
    setSession(null);
    setExample(null);
    setError("");
    remember("movie-evidence:session", "");
  }

  async function select(id: string) {
    const token = ++generation.current;
    setError("");
    setExample(null);
    try {
      const next = await api.get(id);
      if (generation.current !== token) return;
      activate(next);
    } catch (error) {
      if (generation.current === token) setError(errorText(error));
    }
  }

  async function remove(id: string) {
    try {
      await api.delete(id);
      if (active.current === id) newConversation();
      await refreshHistory();
    } catch (error) {
      setError(errorText(error));
    }
  }

  async function send(query: string, requestId: string) {
    if (submitting.current) return false;
    submitting.current = true;
    setPending(true);
    setError("");
    const token = generation.current;
    try {
      let id = active.current;
      if (!id) {
        const created = await api.create(model || capabilities?.default_model || "");
        id = created.id;
        if (token === generation.current) activate(created);
      }
      const started = await api.start(id, query, requestId);
      accept(started);
      setOnline(true);
      void refreshHistory().catch(() => undefined);
      return true;
    } catch (error) {
      if (generation.current === token) setError(errorText(error));
      return false;
    } finally {
      submitting.current = false;
      setPending(false);
    }
  }

  async function reply(body: ReplyBody) {
    if (!session?.run) throw new Error("没有等待回答的问题。");
    try {
      accept(await api.reply(session.id, session.run.id, body));
    } catch (error) {
      // A lost POST response is ambiguous. Refresh state before offering a retry.
      await api.get(session.id).then(accept).catch(() => undefined);
      throw error;
    }
  }

  async function runAction(action: "cancel" | "retry") {
    if (!session?.run) return;
    const token = generation.current;
    try {
      accept(await api[action](session.id, session.run.id));
      if (generation.current === token) setError("");
    } catch (error) {
      if (generation.current === token) setError(errorText(error));
    }
  }

  async function showExample() {
    const token = ++generation.current;
    setError("");
    try {
      const saved = await api.example();
      if (generation.current !== token) return;
      if (!saved.report) throw new Error("本地示例报告不可用。");
      newConversation();
      setExample({ query: saved.query, report: saved.report });
    } catch (error) {
      if (generation.current === token) setError(errorText(error));
    }
  }

  async function reconnect() {
    try {
      const caps = await api.capabilities();
      setCapabilities(caps);
      setModel(current => current || caps.default_model);
      await refreshHistory();
      setOnline(true);
      setError("");
    } catch { setOnline(false); }
  }

  function updateModel(value: string) {
    setModel(value.trim());
    remember("movie-evidence:model", value.trim());
    newConversation();
  }

  return { session, history, capabilities, model, online, pending, error, example, activeId,
    newConversation, select, remove, send, reply, runAction, showExample, reconnect, updateModel };
}
