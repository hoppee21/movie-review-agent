import { useRef, useState } from "react";
import type { Clarification } from "../lib/protocol";
import { Icon } from "./Icon";

type ReplyBody = { request_id: string; issue_id: string; option_id?: string; text?: string; cancelled?: boolean };
export function ClarificationCard({ issue, active, onReply }: { issue: Clarification; active: boolean; onReply: (body: ReplyBody) => Promise<void> }) {
  const [choice, setChoice] = useState("");
  const [text, setText] = useState("");
  const [pending, setPending] = useState(false);
  const [error, setError] = useState("");
  const lastRequest = useRef<{ fingerprint: string; id: string } | null>(null);
  async function submit(cancelled = false) {
    if (pending) return;
    if (!cancelled && !choice && !text.trim()) { setError("请选择一项，或补充一些信息。"); return; }
    const fingerprint = JSON.stringify({ choice, text: text.trim(), cancelled });
    if (lastRequest.current?.fingerprint !== fingerprint) lastRequest.current = { fingerprint, id: crypto.randomUUID() };
    setPending(true); setError("");
    try { await onReply({ request_id: lastRequest.current.id, issue_id: issue.id, option_id: choice || undefined, text: text.trim() || undefined, cancelled }); }
    catch (error) { setError(error instanceof Error ? error.message : "提交失败，请重试。"); }
    finally { setPending(false); }
  }
  return <section className={`clarification-card ${!active ? "answered" : ""}`}>
    <div className="clarification-eyebrow"><Icon name="chat" size={16} />{active ? "需要你确认" : "已处理的确认"}</div>
    <h3>{issue.question}</h3>
    {active ? <>
      <div className="choice-list">{issue.choices.map(option => <label key={option.id} className={`choice-option ${choice === option.id ? "selected" : ""}`}>
        <input type="radio" name={issue.id} value={option.id} checked={choice === option.id} onChange={() => setChoice(option.id)} disabled={pending} />
        <span><strong>{option.label}</strong>{option.description && <small>{option.description}</small>}</span>
      </label>)}</div>
      <label className="supplement-label" htmlFor={`reply-${issue.id}`}>补充信息 <span>可选</span></label>
      <textarea id={`reply-${issue.id}`} value={text} onChange={event => setText(event.target.value)} disabled={pending} maxLength={4000} rows={2} placeholder="也可以补充片名、年份或你关注的方面…" />
      {error && <p className="field-error" role="alert">{error}</p>}
      <div className="clarification-actions"><button className="text-button" onClick={() => void submit(true)} disabled={pending}>取消分析</button>
        <button className="primary-button small" onClick={() => void submit()} disabled={pending}>{pending ? "提交中…" : "确认并继续"}<Icon name="arrow" size={16} /></button></div>
    </> : <p className="archived-question">你的回答已记录在下方。</p>}
  </section>;
}
