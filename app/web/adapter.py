"""Translate today's MovieAgent into the stable web protocol."""

from __future__ import annotations

import re
from typing import Any, Protocol
from uuid import uuid4

from app.web.contracts import Choice, Clarification, Link, Outcome, Report, Section, Source


class AgentDriver(Protocol):
    async def run(self, query: str, *, thread_id: str) -> Outcome: ...

    async def reply(self, reply: dict[str, Any], *, thread_id: str) -> Outcome: ...

    def close(self) -> None: ...


def stage_label(label: str) -> str:
    if not label.startswith("阶段 "):
        return label
    return {
        "ask_user": "确认电影与分析需求", "resolve_movie": "匹配电影与来源平台",
        "collect_movie_reviews": "采集跨平台评论", "build_opinion_index": "整理评论与检索索引",
        "query_movie_opinions": "检索与筛选观点证据", "aggregate_movie_opinions": "生成并复核分析报告",
    }.get(label.removeprefix("阶段 "), "执行分析任务")


def _data(value: Any) -> dict[str, Any]:
    if hasattr(value, "model_dump"):
        value = value.model_dump(mode="json")
    return value if isinstance(value, dict) else {}


def _items(value: Any) -> list[dict[str, Any]]:
    return [_data(item) for item in value] if isinstance(value, list) else []


def _refs(value: Any, aliases: dict[str, str]) -> list[str]:
    return list(dict.fromkeys(aliases[ref] for ref in value if isinstance(ref, str) and ref in aliases)) if isinstance(value, list) else []


def report_from_answer(answer: dict[str, Any], *, title: str = "评论分析", links: list[Link] | None = None) -> Report:
    """Old graph/reranker field names never cross the HTTP boundary."""

    evidence = _items(answer.get("evidence"))
    aliases = {}
    sources = []
    for index, item in enumerate(evidence, 1):
        source_id = f"source-{index}"
        for alias in (item.get("evidence_id"), item.get("citation_id")):
            if isinstance(alias, str):
                aliases[alias] = source_id
        sources.append(Source(
            id=source_id, platform=str(item.get("platform", "来源")),
            title=" · ".join(str(v) for v in (item.get("target"), item.get("subaspect")) if v) or "原始评论",
            quote=str(item.get("quote", "")), full_text=str(item.get("full_text") or item.get("quote", "")),
            stance=str(item.get("stance", "unclear")), reason=item.get("reason"), rating=item.get("rating"),
        ))
    # Render generated citation aliases as human labels; never alter source text.
    numbers = {source.id: index for index, source in enumerate(sources, 1)}
    labels = {alias: f"证据 {numbers[source_id]}"
              for alias, source_id in aliases.items() if re.fullmatch(r"e\d+", alias)}

    def prose(value: Any) -> str:
        return re.sub(r"(?<![A-Za-z0-9_])e\d+(?![A-Za-z0-9_])", lambda match: labels.get(match[0], match[0]), str(value or ""))

    sections = []
    comparison = answer.get("platform_comparison")
    if comparison:
        sections.append(Section(id="comparison", title="跨平台对照", body=prose(comparison),
                                citations=_refs(answer.get("platform_comparison_evidence_ids"), aliases)))
    for index, insight in enumerate(_items(answer.get("insights")), 1):
        sections.append(Section(
            id=f"insight-{index}", title=f"分析与解释 {index}",
            body="\n\n".join(prose(insight.get(key)) for key in ("claim", "reasoning") if insight.get(key)),
            citations=_refs(insight.get("evidence_ids"), aliases),
            counter_citations=_refs(insight.get("counterevidence_ids"), aliases), note=prose(insight.get("limitation")) or None,
        ))
    for index, cluster in enumerate(_items(answer.get("clusters")), 1):
        sections.append(Section(id=f"finding-{index}", title=str(cluster.get("label") or "观众观点"),
                                body=prose(cluster.get("summary")), citations=_refs(cluster.get("evidence_ids"), aliases),
                                counter_citations=_refs(cluster.get("counterevidence_ids"), aliases)))
    coverage = _data(answer.get("coverage"))
    gap_labels = {"reviews": "独立评论不足", "subaspect": "缺少直接观点", "reason": "缺少明确理由或条件", "comparison": "缺少另一平台的对应观点"}
    platform_names = {"douban": "豆瓣", "imdb": "IMDb"}
    gaps = [" · ".join(str(v) for v in (platform_names.get(g.get("platform"), g.get("platform")), g.get("target"), g.get("subaspect"), gap_labels.get(g.get("kind"), "证据有待补充")) if v)
            for g in _items(coverage.get("gaps"))]
    return Report(
        title=title, summary=prose(answer.get("conclusion") or "已完成评论分析。"),
        citations=_refs(answer.get("conclusion_evidence_ids"), aliases), sections=sections, sources=sources,
        source_counts=coverage.get("relevant_by_platform") or {}, coverage_sufficient=coverage.get("sufficient"),
        gaps=gaps, limitations=[str(v) for v in answer.get("limitations", [])], links=links or [],
    )


def outcome_from_result(result: Any) -> Outcome:
    result = _data(result)
    data = _data(result.get("data"))
    targets = _items(data.get("movie_targets"))
    links = [Link(title=platform, url=target[key]) for target in targets
             for platform, key in (("IMDb", "imdb_url"), ("豆瓣", "douban_url"))
             if isinstance(target.get(key), str) and target[key].startswith(("https://", "http://"))]
    title = str(targets[0].get("english_title") or targets[0].get("query") or "评论分析") if targets else "评论分析"
    answer = _data(data.get("answer"))
    if answer:
        return Outcome(report=report_from_answer(answer, title=title, links=links))
    if result.get("status") != "completed":
        status, reason = result.get("status"), result.get("reason")
        messages = {
            "cancelled": "本次分析已取消。你可以开始一个新问题。",
            "unresolved": "电影信息仍不够明确，请开始新的分析并补充片名或年份。",
            "not_implemented": "当前服务还没有接入这项任务，请调整分析需求。",
        }
        if reason in {"Multiple movie targets are reserved but not executed in this skeleton.", "The first skeleton executes exactly one movie target."}:
            reason = "当前支持逐部分析电影，请先选择其中一部。"
        return Outcome(text=messages.get(status) or str(reason or "当前问题暂时无法完成，请补充电影信息或调整问题。"),
                       cancelled=status == "cancelled")
    summary = str(data.get("text") or data.get("message") or f"已找到 {title} 的电影资料。")
    for artifacts in _data(data.get("artifacts")).values():
        reviews = _data(_data(artifacts).get("reviews"))
        if reviews:
            summary = "评论采集已完成。\n" + "\n".join(f"{'豆瓣' if platform == 'douban' else platform}：{_data(value).get('review_count', 0)} 条评论"
                                                    for platform, value in reviews.items())
    return Outcome(report=Report(title=title, summary=summary, links=links))


class MovieAgentDriver:
    def __init__(self, model: str) -> None:
        # Lazy imports make the HTTP contract testable without an OpenAI key.
        from app.agent.Agent import MovieAgent
        from app.rag.runtime import build_openai_chat_model, build_openai_rag_runtime
        from config import load_openai_api_key

        api_key = load_openai_api_key()
        llm = build_openai_chat_model(model, api_key=api_key)
        self.runtime = build_openai_rag_runtime(llm, api_key=api_key)
        self.agent = MovieAgent(llm, rag_runtime=self.runtime)
        self.pending: dict[str, Any] | None = None
        self.query = ""

    async def run(self, query: str, *, thread_id: str) -> Outcome:
        self.query = query
        return self._outcome(await self.agent.run(query, thread_id=thread_id))

    async def reply(self, reply: dict[str, Any], *, thread_id: str) -> Outcome:
        if not self.pending or reply["issue_id"] != self.pending["public_id"]:
            raise ValueError("这个澄清问题已经更新，请刷新后重试。")
        snapshot = await self.agent.graph.aget_state({"configurable": {"thread_id": thread_id}})
        if not snapshot.interrupts:
            # A consumed clarification can fail downstream. MovieAgent.run retries
            # the failed checkpoint without submitting the answer a second time.
            return self._outcome(await self.agent.run(self.query, thread_id=thread_id))
        reply = {**reply, "issue_id": self.pending["issue_id"]}
        return self._outcome(await self.agent.resume(reply, thread_id=thread_id))

    def _outcome(self, state: dict[str, Any]) -> Outcome:
        interrupts = state.get("__interrupt__")
        if interrupts:
            payload = _data(interrupts[0].value)
            public_id = uuid4().hex
            self.pending = {**payload, "public_id": public_id}
            choices = []
            for option in _items(payload.get("options")):
                details = _data(option.get("details"))
                description = " · ".join(str(details[key]) for key in ("english_title", "release_year") if details.get(key))
                choices.append(Choice(id=str(option["option_id"]), label=str(option["label"]), description=description))
            return Outcome(clarification=Clarification(id=public_id, question=str(payload["question"]), choices=choices))
        self.pending = None
        return outcome_from_result(state.get("final_result"))

    def close(self) -> None:
        self.runtime.clear()
