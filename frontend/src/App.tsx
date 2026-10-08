import { useCallback, useEffect, useRef, useState } from "react";
import { ClarificationCard } from "./components/ClarificationCard";
import { Icon } from "./components/Icon";
import { Modal } from "./components/Modal";
import { EvidenceDrawer, ReportViewer } from "./components/ReportViewer";
import { isBusy } from "./lib/protocol";
import type { Report, Run } from "./lib/protocol";
import { useWorkspace } from "./lib/useWorkspace";

const prompts = [
  { title: "黑客帝国", english: "THE MATRIX", year: "1999", aspect: "演员表演", style: "matrix", query: "比较 IMDb 和豆瓣观众如何评价《黑客帝国》(1999) 的演员表演？" },
  { title: "盗梦空间", english: "INCEPTION", year: "2010", aspect: "叙事与结局", style: "inception", query: "比较 IMDb 和豆瓣观众如何评价《盗梦空间》(2010) 的叙事与结局？" },
  { title: "星际穿越", english: "INTERSTELLAR", year: "2014", aspect: "情感表达", style: "interstellar", query: "比较 IMDb 和豆瓣观众如何评价《星际穿越》(2014) 的情感表达？" },
  { title: "爱乐之城", english: "LA LA LAND", year: "2016", aspect: "音乐与爱情", style: "lalaland", query: "比较 IMDb 和豆瓣观众如何评价《爱乐之城》(2016) 的音乐与爱情主题？" },
];

function ProgressPanel({ run, onCancel }: { run: Run; onCancel?: () => void }) {
  const current = [...run.stages].reverse().find(stage => stage.status === "running");
  const label = run.status === "queued" ? "正在等待分析开始" : run.status === "waiting_for_input" ? "等待你补充信息" : current?.label || "正在整理分析结果";
  return <div className="progress-panel"><div className="progress-heading"><span className={run.status === "waiting_for_input" ? "status-dot amber" : "spinner"} /><span>{label}</span>
    {run.status !== "waiting_for_input" && onCancel && <button type="button" className="text-button" onClick={onCancel}><Icon name="stop" size={13} />停止</button>}</div>
    {run.stages.length > 0 && <details><summary>查看处理进度 <span>{run.stages.filter(stage => stage.status === "completed").length} 项已完成</span><Icon name="chevron" size={13} /></summary>
      <div className="stage-list">{run.stages.map(stage => <div key={stage.id} className={`stage ${stage.status}`}>
        {stage.status === "running" ? <span className="spinner tiny" /> : <Icon name={stage.status === "failed" ? "close" : "check"} size={13} />}<span>{stage.label}</span>
        {stage.elapsed_seconds !== null && <small>{stage.elapsed_seconds.toFixed(1)}s</small>}</div>)}</div></details>}
  </div>;
}

export default function App() {
  const workspace = useWorkspace();
  const [query, setQuery] = useState("");
  const [filter, setFilter] = useState("");
  const [menuOpen, setMenuOpen] = useState(false);
  const [settingsOpen, setSettingsOpen] = useState(false);
  const [modelDraft, setModelDraft] = useState("");
  const [sourcePanel, setSourcePanel] = useState<{ report: Report; ids: string[] } | null>(null);
  const [toast, setToast] = useState("");
  const input = useRef<HTMLTextAreaElement>(null);
  const scroll = useRef<HTMLDivElement>(null);
  const request = useRef<{ query: string; id: string } | null>(null);
  const busy = isBusy(workspace.session?.run);
  const features = workspace.capabilities?.features || [];
  const hasContent = Boolean(workspace.session?.messages.length || workspace.example);
  const closeSource = useCallback(() => setSourcePanel(null), []);
  const closeSettings = useCallback(() => setSettingsOpen(false), []);
  const showSource = (report: Report, ids: string[]) => setSourcePanel({ report, ids });

  useEffect(() => {
    const shortcuts = (event: KeyboardEvent) => { if (event.key === "Escape") setMenuOpen(false); if ((event.metaKey || event.ctrlKey) && event.key === "k") { event.preventDefault(); input.current?.focus(); } };
    window.addEventListener("keydown", shortcuts); return () => window.removeEventListener("keydown", shortcuts);
  }, []);
  useEffect(() => { if (toast) { const timer = setTimeout(() => setToast(""), 3500); return () => clearTimeout(timer); } }, [toast]);
  useEffect(() => {
    const container = scroll.current;
    if (!container) return;
    if (!hasContent || workspace.example) { container.scrollTop = 0; return; }
    const last = workspace.session?.messages.at(-1);
    const answers = container.querySelectorAll<HTMLElement>(".assistant-message");
    const answer = answers.item(answers.length - 1);
    // A complete report should open at its beginning, including on a phone.
    container.scrollTop = answer && (last?.report || last?.clarification)
      ? answer.getBoundingClientRect().top - container.getBoundingClientRect().top + container.scrollTop - 24
      : container.scrollHeight;
  }, [hasContent, workspace.activeId, workspace.example, workspace.session?.messages.length, workspace.session?.run?.status]);
  useEffect(() => { if (input.current) { input.current.style.height = "auto"; input.current.style.height = `${Math.min(input.current.scrollHeight, 160)}px`; } }, [query]);

  async function submit() {
    const text = query.trim();
    if (!text || busy || workspace.pending) return;
    if (request.current?.query !== text) request.current = { query: text, id: crypto.randomUUID() };
    const submitted = request.current;
    if (await workspace.send(text, submitted.id)) { setQuery(current => current.trim() === text ? "" : current); if (request.current === submitted) request.current = null; }
  }
  function newConversation() { workspace.newConversation(); setQuery(""); request.current = null; setMenuOpen(false); setSourcePanel(null); input.current?.focus(); }
  function usePrompt(value: string) { setQuery(value); input.current?.focus(); }
  function openSettings() { setModelDraft(workspace.model); setSettingsOpen(true); setMenuOpen(false); }
  const items = workspace.history.filter(item => item.title.toLowerCase().includes(filter.toLowerCase()));

  return <div className="workspace-shell">
    {menuOpen && <button className="sidebar-scrim" aria-label="关闭侧边栏" onClick={() => setMenuOpen(false)} />}
    <aside className={`sidebar ${menuOpen ? "open" : ""}`}>
      <button className="brand" onClick={newConversation}><span className="brand-mark"><Icon name="film" size={22} /></span><span>Movie Evidence<small>影评分析工作台</small></span></button>
      <button className="new-conversation" onClick={newConversation}><Icon name="plus" size={18} />新的分析<kbd>＋</kbd></button>
      <label className="history-search"><Icon name="search" size={15} /><input aria-label="搜索历史分析" placeholder="搜索历史分析" value={filter} onChange={event => setFilter(event.target.value)} /><kbd>⌕</kbd></label>
      <div className="sidebar-section-label">最近的分析<span>{workspace.history.length || ""}</span></div>
      <nav className="history-list" aria-label="历史分析">{items.length ? items.map(item => <div className={`history-row ${workspace.activeId === item.id ? "active" : ""}`} key={item.id}>
        <button onClick={() => { void workspace.select(item.id); setMenuOpen(false); setSourcePanel(null); }}><Icon name="chat" size={15} /><span>{item.title}</span>{["queued", "running", "waiting_for_input"].includes(item.status) && <span className="status-dot amber" />}</button>
        <button className="history-delete" aria-label={`删除分析：${item.title}`} title="删除会话" onClick={() => void workspace.remove(item.id)}><Icon name="trash" size={13} /></button>
      </div>) : <div className="empty-history"><Icon name="chat" size={23} /><p>{filter ? "没有匹配的分析" : "你的分析会保存在这里"}</p><span>{filter ? "试试其他关键词" : "随时回来，继续查看观点与依据"}</span></div>}</nav>
      <div className="sidebar-bottom"><div className="sidebar-note"><Icon name="shield" size={15} /><span>从真实评论出发<br /><small>让每个判断都有迹可循</small></span></div>
        <button className="workspace-settings" onClick={openSettings}><span className="profile-avatar">W</span><span>个人工作空间<small>{workspace.model || "连接服务后选择模型"}</small></span><Icon name="settings" size={17} /></button></div>
    </aside>
    <main className="workspace-main">
      <header className="topbar"><div className="topbar-left"><button className="icon-button mobile-menu" aria-label="打开侧边栏" onClick={() => setMenuOpen(true)}><Icon name="menu" /></button>
        <span className="breadcrumb">工作台</span><Icon name="chevron" size={13} /><span className="page-title">{workspace.example ? "示例报告" : workspace.session?.title || "新的分析"}</span></div>
        <div className="topbar-right"><span className={`connection-status ${workspace.online === false ? "offline" : ""}`}><span className={`status-dot ${workspace.online === false ? "red" : workspace.online === null ? "amber" : "green"}`} />{workspace.online === false ? "服务未连接" : workspace.online === null ? "连接中" : "服务已连接"}</span>
          <button className="icon-button" title="工作台设置" aria-label="工作台设置" onClick={openSettings}><Icon name="settings" size={18} /></button></div></header>
      <div className={`conversation-scroll ${hasContent ? "has-content" : ""}`} ref={scroll}>
        {!hasContent ? <div className="welcome">
          <div className="welcome-eyebrow"><span />CROSS-PLATFORM FILM INSIGHTS</div>
          <h1>不同的评分，<br />藏着怎样的<span>观点？</span></h1>
          <p className="welcome-description">从 IMDb 与豆瓣的真实评论出发，<br className="mobile-break" />一起理解评价背后的理由、共识与分歧。</p>
          <div className="platform-pair"><span className="imdb-wordmark">IMDb</span><span className="platform-pair-line" /><span className="douban-wordmark">豆瓣</span><span className="platform-pair-label">两种视角，一部电影</span></div>
          <div className="suggestions-heading"><span>从一个问题开始</span>{workspace.capabilities?.example_available && <button className="text-button" onClick={() => void workspace.showExample()}>查看示例报告<Icon name="arrow" size={14} /></button>}</div>
          <div className="suggestions-grid">{prompts.map(prompt => <button className="suggestion-card" key={prompt.style} onClick={() => usePrompt(prompt.query)}>
            <div className={`mini-poster ${prompt.style}`}><span className="poster-art" /><span className="poster-caption">{prompt.english}</span></div>
            <div className="suggestion-copy"><span className="suggestion-year">{prompt.year}</span><h3>{prompt.title}</h3><p>比较观众如何评价{prompt.aspect}</p><span className="suggestion-bottom">IMDb × 豆瓣<Icon name="arrow" size={14} /></span></div></button>)}</div>
          <div className="welcome-features"><span><Icon name="search" size={16} />检索具体观点</span><span><Icon name="book" size={16} />查看完整原评论</span><span><Icon name="shield" size={16} />保留分析边界</span></div>
        </div> : <div className="message-list">
          {workspace.example && <><div className="user-message"><span className="user-avatar">你</span><p>{workspace.example.query}</p></div><div className="assistant-message"><span className="assistant-avatar"><Icon name="film" size={17} /></span><ReportViewer report={workspace.example.report} onSource={showSource} onToast={setToast} example /></div></>}
          {workspace.session?.messages.map(message => message.role === "user" ? <div className="user-message" key={message.id}><span className="user-avatar">你</span><p>{message.text}</p></div>
            : <div className="assistant-message" key={message.id}><span className="assistant-avatar"><Icon name="film" size={17} /></span><div className="assistant-content">
              {message.report ? <ReportViewer report={message.report} onSource={showSource} onToast={setToast} /> : message.clarification ? <ClarificationCard issue={message.clarification} active={workspace.session?.run?.clarification?.id === message.clarification.id && workspace.session?.run?.status === "waiting_for_input"} onReply={workspace.reply} /> : <p className="plain-answer">{message.text}</p>}
            </div></div>)}
          {busy && workspace.session?.run && <div className="progress-wrap"><ProgressPanel run={workspace.session.run} onCancel={features.includes("cancellation") ? () => void workspace.runAction("cancel") : undefined} /></div>}
          {workspace.session?.run?.status === "failed" && <div className="run-error" role="alert"><Icon name="info" size={18} /><div><strong>分析暂时中断</strong><p>{workspace.session.run.error}</p></div>{features.includes("retry") && <button className="secondary-button" onClick={() => void workspace.runAction("retry")}><Icon name="refresh" size={14} />重试</button>}</div>}
        </div>}
      </div>
      <div className="composer-area">
        {workspace.error && <div className="composer-error" role="alert"><Icon name="info" size={15} /><span>{workspace.error}</span></div>}
        {workspace.online === false && <div className="offline-banner"><span>尚未连接到分析服务。请启动后端后重试。</span><button onClick={() => void workspace.reconnect()}><Icon name="refresh" size={13} />重新连接</button></div>}
        <form className={`composer ${busy ? "disabled" : ""}`} onSubmit={event => { event.preventDefault(); void submit(); }}>
          <textarea ref={input} aria-label="你的电影问题" value={query} onChange={event => setQuery(event.target.value)} maxLength={workspace.capabilities?.max_query_chars || 12000} rows={2}
            placeholder={busy ? "当前分析完成后，可以开始一个新问题…" : "输入电影、平台和你关注的方面，例如：比较《黑客帝国》的演员表演…"} disabled={busy || workspace.pending}
            onKeyDown={event => { if (event.key === "Enter" && !event.shiftKey && !event.nativeEvent.isComposing) { event.preventDefault(); void submit(); } }} />
          <div className="composer-bottom"><span className="composer-hint"><Icon name="sparkle" size={14} />{workspace.example ? "示例来自本地已保存报告" : "用自然语言提出你的问题"}</span><div><span className="enter-hint">Enter 发送</span><button className="send-button" type="submit" disabled={!query.trim() || busy || workspace.pending} aria-label="发送问题">{workspace.pending ? <span className="spinner" /> : <Icon name="send" size={20} />}</button></div></div>
        </form>
        <p className="composer-disclaimer">分析以检索到的评论为依据。证据数量不代表平台整体的观众立场。</p>
      </div>
    </main>
    {sourcePanel && <EvidenceDrawer key={sourcePanel.ids.join("|")} report={sourcePanel.report} ids={sourcePanel.ids} onClose={closeSource} />}
    {settingsOpen && <Modal title="工作台设置" onClose={closeSettings}><div className="settings-content"><span className="settings-icon"><Icon name="settings" size={24} /></span><h2>为下一次分析做好准备</h2><p>选择新会话使用的模型。当前会话会保留自己的运行配置。</p><label htmlFor="model-setting">模型名称</label><input id="model-setting" value={modelDraft} onChange={event => setModelDraft(event.target.value)} placeholder={workspace.capabilities?.default_model || "输入后端可用的模型名称"} maxLength={200} />
      <div className="settings-note"><Icon name="shield" size={16} /><span>密钥由后端管理，浏览器无需填写。历史会话保存在当前服务进程中。</span></div><button className="primary-button" disabled={!modelDraft.trim()} onClick={() => { workspace.updateModel(modelDraft); setQuery(""); setSettingsOpen(false); }}>保存并开始新会话<Icon name="arrow" size={16} /></button></div></Modal>}
    {toast && <div className="toast" role="status"><Icon name="check" size={16} />{toast}</div>}
  </div>;
}
