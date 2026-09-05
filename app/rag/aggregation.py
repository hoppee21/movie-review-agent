"""Evidence-grounded aggregation of ranked movie-review opinions."""

from __future__ import annotations

import json
from typing import Any, Protocol, Sequence

from langchain_core.language_models import BaseChatModel
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel

from app.rag.models import (
    AggregationDraft,
    EvidenceCard,
    MovieAnswer,
    OpinionCluster,
    RankedEvidence,
    RetrievalResult,
    ReviewPlatform,
)


AGGREGATION_SYSTEM_PROMPT = """
你根据给出的电影评论证据回答用户问题。只能使用 evidence JSON 中的信息。

- conclusion 先直接回答问题，再说明主要分歧；不要把样本外推为全体用户。
- clusters 按观点而不是关键词组织，每组必须引用有效 evidence_id。
- 平台比较只在同时提供 IMDb 和豆瓣时填写，并区分平台；否则为 null。
- 不编造比例、人数、评论、引文或因果关系。
- limitations 只写证据本身暴露出的限制。系统会另行补充采样偏差说明。
""".strip()


SAMPLE_LIMITATION = (
    "结果只描述本次采集的热门/高赞评论样本，不能代表平台上的全部用户。"
)


class OpinionAggregator(Protocol):
    async def aggregate(self, retrieval: RetrievalResult) -> MovieAnswer: ...


class LLMOpinionAggregator:
    """Select diverse evidence, synthesize it, and validate every citation."""

    def __init__(
        self,
        llm: BaseChatModel,
        *,
        evidence_per_platform: int,
    ) -> None:
        if evidence_per_platform < 1:
            raise ValueError("evidence_per_platform must be positive")
        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", AGGREGATION_SYSTEM_PROMPT),
                (
                    "human",
                    "用户问题：{question}\n"
                    "分析方面：{aspect}\n"
                    "目标平台：{platforms}\n"
                    "覆盖情况：{coverage}\n\n"
                    "证据（JSON）：\n{evidence}",
                ),
            ]
        )
        output = llm.with_structured_output(
            AggregationDraft,
            method="json_schema",
            strict=True,
        )
        self._chain = prompt | output
        self.evidence_per_platform = evidence_per_platform

    async def aggregate(self, retrieval: RetrievalResult) -> MovieAnswer:
        selected = select_diverse_evidence(
            retrieval.evidence,
            platforms=retrieval.platforms,
            limit_per_platform=self.evidence_per_platform,
        )
        limitations = _base_limitations(retrieval)
        if not selected:
            return MovieAnswer(
                conclusion=(
                    "当前采集到的评论中，没有找到足够直接回答该问题的证据。"
                ),
                clusters=[],
                stance_counts=retrieval.coverage.stance_counts,
                platform_comparison=None,
                evidence=[],
                coverage=retrieval.coverage,
                limitations=limitations,
            )

        value = await self._chain.ainvoke(
            {
                "question": retrieval.question,
                "aspect": retrieval.query_plan.aspect or "未单独指定",
                "platforms": ", ".join(retrieval.platforms),
                "coverage": retrieval.coverage.model_dump_json(),
                "evidence": json.dumps(
                    [
                        {
                            "evidence_id": item.evidence_id,
                            "platform": item.platform,
                            "stance": item.stance.value,
                            "subaspect": item.subaspect,
                            "quote": item.evidence_span or item.chunk_text,
                        }
                        for item in selected
                    ],
                    ensure_ascii=False,
                ),
            }
        )
        draft = AggregationDraft.model_validate(_model_data(value))
        evidence_by_id = {item.evidence_id: item for item in selected}
        clusters: list[OpinionCluster] = []
        for raw_cluster in draft.clusters:
            evidence_ids = [
                evidence_id
                for evidence_id in dict.fromkeys(raw_cluster.evidence_ids)
                if evidence_id in evidence_by_id
            ]
            if not evidence_ids:
                continue
            platforms = list(
                dict.fromkeys(
                    evidence_by_id[evidence_id].platform
                    for evidence_id in evidence_ids
                )
            )
            evidence_count = len(
                {
                    (
                        evidence_by_id[evidence_id].platform,
                        evidence_by_id[evidence_id].review_id,
                    )
                    for evidence_id in evidence_ids
                }
            )
            clusters.append(
                OpinionCluster(
                    label=raw_cluster.label,
                    summary=raw_cluster.summary,
                    stance=raw_cluster.stance,
                    platforms=platforms,
                    evidence_count=evidence_count,
                    evidence_ids=evidence_ids,
                )
            )

        for limitation in draft.limitations:
            normalized = limitation.strip()
            if normalized and normalized not in limitations:
                limitations.append(normalized)
        evidence_platforms = {item.platform for item in selected}
        comparison = (
            draft.platform_comparison
            if len(retrieval.platforms) == 2
            and set(retrieval.platforms) <= evidence_platforms
            else None
        )
        return MovieAnswer(
            conclusion=draft.conclusion,
            clusters=clusters,
            stance_counts=retrieval.coverage.stance_counts,
            platform_comparison=comparison,
            evidence=[_evidence_card(item) for item in selected],
            coverage=retrieval.coverage,
            limitations=limitations,
        )


def select_diverse_evidence(
    evidence: Sequence[RankedEvidence],
    *,
    platforms: Sequence[ReviewPlatform],
    limit_per_platform: int,
) -> list[RankedEvidence]:
    """Prefer distinct reviews and viewpoint/subaspect coverage per platform."""

    relevant = [item for item in evidence if item.relevance >= 2]
    relevant.sort(
        key=lambda item: (
            -item.relevance,
            -item.evidence_quality,
            -item.fusion_score,
            item.evidence_id,
        )
    )
    result: list[RankedEvidence] = []
    for platform in platforms:
        platform_items: list[RankedEvidence] = []
        seen_reviews: set[str] = set()
        for item in relevant:
            if item.platform == platform and item.review_id not in seen_reviews:
                platform_items.append(item)
                seen_reviews.add(item.review_id)

        selected: list[RankedEvidence] = []
        selected_ids: set[str] = set()
        seen_stances: set[str] = set()
        for item in platform_items:
            if item.stance.value not in seen_stances:
                selected.append(item)
                selected_ids.add(item.evidence_id)
                seen_stances.add(item.stance.value)
                if len(selected) == limit_per_platform:
                    break

        seen_groups = {
            (
                item.stance.value,
                (item.subaspect or "general").strip().casefold(),
            )
            for item in selected
        }
        if len(selected) < limit_per_platform:
            for item in platform_items:
                group = (
                    item.stance.value,
                    (item.subaspect or "general").strip().casefold(),
                )
                if item.evidence_id not in selected_ids and group not in seen_groups:
                    selected.append(item)
                    selected_ids.add(item.evidence_id)
                    seen_groups.add(group)
                    if len(selected) == limit_per_platform:
                        break
        if len(selected) < limit_per_platform:
            selected.extend(
                item
                for item in platform_items
                if item.evidence_id not in selected_ids
            )
            selected = selected[:limit_per_platform]
        result.extend(selected)
    return result


def _base_limitations(retrieval: RetrievalResult) -> list[str]:
    limitations = [SAMPLE_LIMITATION]
    if not retrieval.coverage.sufficient and retrieval.coverage.reason:
        limitations.append(f"证据覆盖仍不足：{retrieval.coverage.reason}")
    return limitations


def _evidence_card(item: RankedEvidence) -> EvidenceCard:
    return EvidenceCard(
        evidence_id=item.evidence_id,
        platform=item.platform,
        review_id=item.review_id,
        quote=item.evidence_span or item.chunk_text,
        full_text=item.full_text,
        rating=item.rating,
        helpful_votes=item.helpful_votes,
    )


def _model_data(value: dict[str, Any] | BaseModel) -> dict[str, Any]:
    if isinstance(value, BaseModel):
        return value.model_dump()
    return value


__all__ = [
    "LLMOpinionAggregator",
    "OpinionAggregator",
    "SAMPLE_LIMITATION",
    "select_diverse_evidence",
]
