import { useCallback, useEffect, useRef, useState } from "react";
import { api, ApiError } from "./api";
import { applySnapshot } from "./protocol";
import type { Capabilities, Report, Session, SessionItem } from "./protocol";

const recalled = (key: string) => { try { return localStorage.getItem(key) || ""; } catch { return ""; } };
const savedModel = () => recalled("movie-evidence:model");
const remember = (key: string, value: string) => { try { localStorage.setItem(key, value); } catch { /* Private browsing can disable storage. */ } };
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
  const streamConnected = useRef(false);

  const refreshHistory = useCallback(async () => { setHistory(await api.list()); }, []);
  const accept = useCallback((incoming: Session) => {
    if (active.current !== incoming.id) return;
    setSession(current => applySnapshot(current, incoming));
    setHistory(current => [{ id: incoming.id, title: incoming.title, updated_at: incoming.updated_at, status: incoming.run?.status || "idle" },
      ...current.filter(item => item.id !== incoming.id)].sort((a, b) => b.updated_at.localeCompare(a.updated_at)));
  }, []);

  useEffect(() => {
    let mounted = true;
    void (async () => {
      try {
        const [caps, items] = await Promise.all([api.capabilities(), api.list()]);
        if (!mounted) return;
        setCapabilities(caps); setModel(current => current || caps.default_model); setHistory(items); setOnline(true);
        const remembered = recalled("movie-evidence:session");
        if (remembered && items.some(item => item.id === remembered)) {
          const restored = await api.get(remembered);
          if (mounted && generation.current === 0 && !active.current) { active.current = remembered; setActiveId(remembered); setSession(restored); }
        }
      } catch { if (mounted) setOnline(false); }
    })();
    return () => { mounted = false; };
  }, []);

  useEffect(() => {
    if (!activeId) return;
    streamConnected.current = false;
    const close = api.watch(activeId, -1, accept, connected => { streamConnected.current = connected; if (connected) setOnline(true); });
    const poll = window.setInterval(() => {
      if (!streamConnected.current) void api.get(activeId).then(value => { accept(value); setOnline(true); }).catch(error => {
        if (error instanceof ApiError && error.status === 404) {
          close(); active.current = null; setActiveId(null); setSession(null); remember("movie-evidence:session", "");
          setError("服务已重新启动。之前的运行会话已释放，请开始新的分析。");
        } else setOnline(false);
      });
    }, 3000);
    return () => { close(); clearInterval(poll); };
  }, [activeId, accept]);

  function newConversation() {
    generation.current += 1; active.current = null; setActiveId(null); setSession(null); setExample(null); setError("");
    remember("movie-evidence:session", "");
  }

  async function select(id: string) {
    const token = ++generation.current;
    setError(""); setExample(null);
    try {
      const next = await api.get(id);
      if (generation.current !== token) return;
      active.current = id; setActiveId(id); setSession(next); remember("movie-evidence:session", id);
    } catch (error) { if (generation.current === token) setError(errorText(error)); }
  }

  async function remove(id: string) {
    try { await api.delete(id); if (active.current === id) newConversation(); await refreshHistory(); }
    catch (error) { setError(errorText(error)); }
  }

  async function send(query: string, requestId: string) {
    if (submitting.current) return false;
    submitting.current = true; setPending(true); setError("");
    const token = generation.current;
    try {
      let id = active.current;
      if (!id) {
        const created = await api.create(model || capabilities?.default_model || ""); id = created.id;
        if (token === generation.current) { active.current = id; setActiveId(id); setSession(created); setExample(null); remember("movie-evidence:session", id); }
      }
      const started = await api.start(id, query, requestId);
      accept(started); setOnline(true); void refreshHistory().catch(() => undefined); return true;
    } catch (error) { setError(errorText(error)); return false; }
    finally { submitting.current = false; setPending(false); }
  }

  async function reply(body: { request_id: string; issue_id: string; option_id?: string; text?: string; cancelled?: boolean }) {
    if (!session?.run) throw new Error("没有等待回答的问题。");
    try { accept(await api.reply(session.id, session.run.id, body)); }
    catch (error) {
      // A lost POST response is ambiguous. Refresh state before offering a retry.
      await api.get(session.id).then(accept).catch(() => undefined);
      throw error;
    }
  }

  async function runAction(action: "cancel" | "retry") {
    if (!session?.run) return;
    try { accept(await api[action](session.id, session.run.id)); setError(""); }
    catch (error) { setError(errorText(error)); }
  }

  async function showExample() {
    const token = ++generation.current;
    setError("");
    try {
      const saved = await api.example();
      if (generation.current !== token) return;
      if (!saved.report) throw new Error("本地示例报告不可用。");
      newConversation(); setExample({ query: saved.query, report: saved.report });
    } catch (error) { setError(errorText(error)); }
  }

  async function reconnect() {
    try {
      const caps = await api.capabilities(); setCapabilities(caps); setModel(current => current || caps.default_model);
      await refreshHistory(); setOnline(true); setError("");
    } catch { setOnline(false); }
  }

  function updateModel(value: string) { setModel(value.trim()); remember("movie-evidence:model", value.trim()); newConversation(); }

  return { session, history, capabilities, model, online, pending, error, example, activeId,
    newConversation, select, remove, send, reply, runAction, showExample, reconnect, updateModel };
}
