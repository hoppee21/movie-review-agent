"""Aspect-aware LLM reranking with evidence annotations in one pass."""

from __future__ import annotations

import json
from collections.abc import Sequence
from typing import Any, Protocol

from langchain_core.language_models import BaseChatModel
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel

from app.rag.models import (
    OpinionStance,
    RankedEvidence,
    RerankJudgement,
    RerankResponse,
    RetrievalCandidate,
)


RERANK_SYSTEM_PROMPT = """
你是电影评论证据重排器。只判断候选评论是否直接回答用户问题，不总结整部电影。

逐条返回：
- relevance：0 无关，1 只间接提及，2 直接相关，3 对问题有具体明确观点。
- evidence_quality：0 无可引用内容，1 有判断但理由弱，2 有清晰观点或具体理由。
- stance：positive / negative / mixed / unclear，表示对所问方面的态度。
- subaspect：评论实际讨论的细分方面，统一使用简短中文标签；无法判断时为 null。
- evidence_span：从候选 chunk_text 原样复制的一小段直接证据；不得改写，找不到则为 null。

不要因为星级、点赞数、措辞长或平台不同而提高相关性。每个输入 chunk_id 只能返回一次。
""".strip()


class OpinionReranker(Protocol):
    async def rerank(
        self,
        *,
        question: str,
        aspect: str | None,
        candidates: Sequence[RetrievalCandidate],
    ) -> list[RankedEvidence]: ...


class LLMOpinionReranker:
    """Rerank and extract stance/subaspect/span without a second LLM pass."""

    def __init__(self, llm: BaseChatModel, *, batch_size: int) -> None:
        if batch_size < 1:
            raise ValueError("rerank batch_size must be positive")
        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", RERANK_SYSTEM_PROMPT),
                (
                    "human",
                    "用户问题：{question}\n"
                    "目标方面：{aspect}\n\n"
                    "候选评论（JSON）：\n{candidates}",
                ),
            ]
        )
        output = llm.with_structured_output(
            RerankResponse,
            method="json_schema",
            strict=True,
        )
        self._chain = prompt | output
        self.batch_size = batch_size

    async def rerank(
        self,
        *,
        question: str,
        aspect: str | None,
        candidates: Sequence[RetrievalCandidate],
    ) -> list[RankedEvidence]:
        judgements: dict[str, RerankJudgement] = {}
        for start in range(0, len(candidates), self.batch_size):
            batch = list(candidates[start : start + self.batch_size])
            raw_result = await self._chain.ainvoke(
                {
                    "question": question,
                    "aspect": aspect or "未指定，按用户问题判断",
                    "candidates": json.dumps(
                        [
                            {
                                "chunk_id": item.chunk_id,
                                "platform": item.platform,
                                "chunk_text": item.chunk_text,
                            }
                            for item in batch
                        ],
                        ensure_ascii=False,
                    ),
                }
            )
            result = RerankResponse.model_validate(_model_data(raw_result))
            allowed_ids = {item.chunk_id for item in batch}
            for judgement in result.items:
                if judgement.chunk_id in allowed_ids:
                    judgements.setdefault(judgement.chunk_id, judgement)

        ranked = [
            _to_ranked(candidate, judgements.get(candidate.chunk_id))
            for candidate in candidates
        ]
        ranked.sort(
            key=lambda item: (
                -item.relevance,
                -item.evidence_quality,
                -item.fusion_score,
                item.chunk_id,
            )
        )
        return ranked


def _to_ranked(
    candidate: RetrievalCandidate,
    judgement: RerankJudgement | None,
) -> RankedEvidence:
    if judgement is None:
        judgement = RerankJudgement(
            chunk_id=candidate.chunk_id,
            relevance=0,
            evidence_quality=0,
            stance=OpinionStance.UNCLEAR,
            subaspect=None,
            evidence_span=None,
        )
    evidence_span = judgement.evidence_span
    if evidence_span and evidence_span not in candidate.chunk_text:
        evidence_span = None
    return RankedEvidence(
        evidence_id=f"evidence:{candidate.chunk_id}",
        chunk_id=candidate.chunk_id,
        review_id=candidate.review_id,
        platform=candidate.platform,
        chunk_text=candidate.chunk_text,
        full_text=candidate.full_text,
        title=candidate.title,
        rating=candidate.rating,
        helpful_votes=candidate.helpful_votes,
        fusion_score=candidate.fusion_score,
        relevance=judgement.relevance,
        evidence_quality=judgement.evidence_quality,
        stance=judgement.stance,
        subaspect=(
            judgement.subaspect.strip()
            if judgement.subaspect and judgement.subaspect.strip()
            else None
        ),
        evidence_span=evidence_span,
    )


def _model_data(value: dict[str, Any] | BaseModel) -> dict[str, Any]:
    if isinstance(value, BaseModel):
        return value.model_dump()
    return value


__all__ = ["LLMOpinionReranker", "OpinionReranker"]
