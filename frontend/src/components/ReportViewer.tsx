import { useState } from "react";
import type { Report, Source } from "../lib/protocol";
import { Icon } from "./Icon";
import { Modal } from "./Modal";

export const platformName = (platform: string) => platform.toLowerCase() === "douban" ? "豆瓣" : platform.toLowerCase() === "imdb" ? "IMDb" : platform;
const stanceName = (stance: string) => ({ positive: "认可", negative: "批评", mixed: "条件式 / 混合", unclear: "未明确" }[stance] || stance);

function reportText(report: Report) {
  const references = (ids: string[], label = "评论依据") => {
    const numbers = ids.map(id => report.sources.findIndex(source => source.id === id) + 1).filter(number => number > 0);
    return numbers.length ? `\n\n${label}：${numbers.map(number => `[${number}]`).join(" ")}` : "";
  };
  return [`# ${report.title}`, report.summary + references(report.citations),
    ...report.sections.map(section => `## ${section.title}\n\n${section.body}${references(section.citations)}${references(section.counter_citations, "限定 / 反证")}${section.note ? `\n\n范围说明：${section.note}` : ""}`),
    "## 证据与原评论", ...report.sources.map((source, index) => `### [${index + 1}] ${platformName(source.platform)} · ${source.title}\n\n${source.full_text}`),
    "## 分析边界", ...report.limitations, ...report.gaps].join("\n\n");
}

export function ReportViewer({ report, onSource, onToast, example = false }: { report: Report; onSource: (report: Report, ids: string[]) => void; onToast: (text: string) => void; example?: boolean }) {
  const [evidenceOpen, setEvidenceOpen] = useState(false);
  function citations(ids: string[], counter = false) {
    const valid = ids.filter(id => report.sources.some(source => source.id === id));
    if (!valid.length) return null;
    return <span className={`citations ${counter ? "counter-citations" : ""}`}><span>{counter ? "限定 / 反证" : "评论依据"}</span>
      {valid.map(id => <button type="button" key={id} onClick={() => onSource(report, [id, ...valid.filter(other => other !== id)])} aria-label={`查看证据 ${report.sources.findIndex(source => source.id === id) + 1}`}>
        {report.sources.findIndex(source => source.id === id) + 1}</button>)}</span>;
  }
  async function copy() { try { await navigator.clipboard.writeText(reportText(report)); onToast("报告已复制"); } catch { onToast("复制未完成，你可以下载报告。"); } }
  function download() {
    const url = URL.createObjectURL(new Blob([reportText(report)], { type: "text/markdown;charset=utf-8" }));
    const link = document.createElement("a"); link.href = url; link.download = "movie-evidence-report.md"; link.click();
    setTimeout(() => URL.revokeObjectURL(url), 1000);
  }
  return <article className="report">
    <div className="report-toolbar"><span className="report-eyebrow"><Icon name="book" size={15} />{example ? "本地历史报告 · 示例" : "证据分析报告"}</span>
      <div className="report-tools"><button type="button" className="icon-button" title="复制报告" aria-label="复制报告" onClick={() => void copy()}><Icon name="copy" size={16} /></button>
        <button type="button" className="icon-button" title="下载报告" aria-label="下载报告" onClick={download}><Icon name="download" size={16} /></button></div></div>
    <h2>{report.title}</h2>
    {report.sources.length > 0 && <div className="report-meta"><span>{report.sources.length} 条观点证据</span><span className="meta-divider" />
      {Object.entries(report.source_counts).map(([platform, count]) => <span key={platform}>{platformName(platform)} <b>{count}</b> 条独立评论</span>)}</div>}
    <div className="report-summary"><span className="summary-mark">结论</span><p>{report.summary}</p>{citations(report.citations)}</div>
    {report.sections.map(section => <section className="report-section" key={section.id}><h3>{section.title}</h3><p>{section.body}</p>
      <div className="citation-row">{citations(section.citations)}{citations(section.counter_citations, true)}</div>
      {section.note && <div className="section-note"><Icon name="info" size={14} /><span>{section.note}</span></div>}</section>)}
    {report.links.length > 0 && <div className="report-links">{report.links.map(link => <a key={link.url} href={link.url} target="_blank" rel="noreferrer">{link.title}<Icon name="external" size={13} /></a>)}</div>}
    {report.sources.length > 0 && <div className="evidence-list"><button type="button" className="evidence-toggle" onClick={() => setEvidenceOpen(!evidenceOpen)} aria-expanded={evidenceOpen}>
      <span><Icon name="book" size={16} /> 查看全部证据 <b>{report.sources.length}</b></span><Icon name="chevron" className={evidenceOpen ? "rotated" : ""} size={15} /></button>
      {evidenceOpen && <div className="evidence-items">{report.sources.map((source, index) => <button key={source.id} className="evidence-item" onClick={() => onSource(report, [source.id])}>
        <span className={`platform-badge ${source.platform.toLowerCase()}`}>{platformName(source.platform)}</span><span><strong>{source.title}</strong><small>{source.quote}</small></span><span className="evidence-number">{index + 1}</span></button>)}</div>}</div>}
    {(report.limitations.length > 0 || report.gaps.length > 0) && <details className="limitations"><summary><Icon name="shield" size={16} />分析边界{report.coverage_sufficient === false && <span className="coverage-note">部分证据仍有缺口</span>}<Icon name="chevron" size={14} /></summary>
      <div>{report.limitations.map((item, index) => <p key={index}>{item}</p>)}{report.gaps.length > 0 && <ul>{report.gaps.map((item, index) => <li key={index}>{item}</li>)}</ul>}</div></details>}
  </article>;
}

export function EvidenceDrawer({ report, ids, onClose }: { report: Report; ids: string[]; onClose: () => void }) {
  const sources = ids.map(id => report.sources.find(source => source.id === id)).filter((source): source is Source => Boolean(source));
  const [index, setIndex] = useState(0);
  const source = sources[index];
  if (!source) return null;
  const number = report.sources.findIndex(item => item.id === source.id) + 1;
  return <Modal title="原始评论与证据" onClose={onClose} drawer>
    <div className="drawer-body"><div className="source-topline"><span className={`platform-badge ${source.platform.toLowerCase()}`}>{platformName(source.platform)}</span><span>证据 {number}</span></div>
      <h2>{source.title}</h2><div className="source-meta"><span className={`stance ${source.stance}`}>{stanceName(source.stance)}</span>{source.rating !== null && <span>评论评分 {source.rating}</span>}</div>
      <h4>引用片段</h4><blockquote>{source.quote}</blockquote>
      {source.reason && <div className="source-reason"><span>原文明示的理由 / 条件</span><p>{source.reason}</p></div>}
      <details className="full-review"><summary>完整原评论 <Icon name="chevron" size={15} /></summary><p>{source.full_text}</p></details>
      {source.url && <a className="source-link" href={source.url} target="_blank" rel="noreferrer">打开来源 <Icon name="external" size={14} /></a>}
      <p className="source-disclaimer">这条评论用于支持具体观点，不能代表整个平台的观众。</p>
    </div>
    {sources.length > 1 && <div className="source-pagination"><button disabled={index === 0} onClick={() => setIndex(index - 1)}>上一条</button><span>{index + 1} / {sources.length}</span><button disabled={index === sources.length - 1} onClick={() => setIndex(index + 1)}>下一条</button></div>}
  </Modal>;
}
