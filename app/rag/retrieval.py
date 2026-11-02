"""Question-led hybrid retrieval, with bounded reading contexts and one follow-up."""

from __future__ import annotations

import json
from collections import Counter
from typing import Protocol, Sequence
from uuid import uuid4

import numpy as np
from langchain_core.language_models import BaseChatModel
from langchain_core.prompts import ChatPromptTemplate

from app.progress import progress
from app.rag.embedding import EmbeddingProvider
from app.rag.evidence import coverage_report, select_evidence
from app.rag.index import (
    EphemeralRagStore,
    normalize_query_vectors,
)
from app.rag.models import (
    QuerySpec,
    RankedEvidence,
    RetrievalArtifact,
    RetrievalCandidate,
    RetrievalQueryPlan,
    RetrievalResult,
    ReviewPlatform,
)
from app.rag.movie_context import MOVIE_CONTEXT_GUIDANCE
from app.rag.passages import reading_ranges
from app.rag.reranker import OpinionReranker
from app.run_log import record_event
from config import RagConfig

QUERY_SYSTEM_PROMPT = (
    """
把用户的问题改写为这部电影的评论检索查询。IMDb 用英文，豆瓣用中文。
保留用户明确指定的对象、方面、比较关系和限制；通常每个平台一条，必要时最多两条。
可用电影背景核对名称和别名；宽问题不要擅自缩成只问某位主演。
不要预设评价好坏、生成分析维度清单或编写假设评论。

首次检索：为每个目标平台生成忠于问题的中性查询。
补检索：阅读已有证据，只为阻碍回答用户问题的具体缺口生成定向查询。
例如某平台无直接证据、用户明确询问理由却只有评价，或一个必要比较缺少对应证据。
不为凑数量、预想分类、正反立场或每个偶然出现的演员补齐证据。
已有证据足以在其范围内回答时，返回 queries=[]；找不到证据不代表评论者没有该观点。
""".strip()
    + "\n\n"
    + MOVIE_CONTEXT_GUIDANCE
)


class RetrievalQueryPlanner(Protocol):
    async def plan(
        self,
        *,
        question: str,
        aspect: str | None,
        platforms: Sequence[ReviewPlatform],
        movie_context: str = "",
        evidence: Sequence[RankedEvidence] | None = None,
    ) -> RetrievalQueryPlan: ...


class LLMRetrievalQueryPlanner:
    def __init__(self, llm: BaseChatModel) -> None:
        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", QUERY_SYSTEM_PROMPT),
                (
                    "human",
                    "阶段：{stage}\n用户问题：{question}\n明确方面：{aspect}\n"
                    "目标平台：{platforms}\n电影背景：{movie_context}\n已有证据：{evidence}",
                ),
            ]
        )
        self._chain = prompt | llm.with_structured_output(
            RetrievalQueryPlan,
            method="json_schema",
            strict=True,
        )

    async def plan(
        self,
        *,
        question: str,
        aspect: str | None,
        platforms: Sequence[ReviewPlatform],
        movie_context: str = "",
        evidence: Sequence[RankedEvidence] | None = None,
    ) -> RetrievalQueryPlan:
        stage = "首次检索" if evidence is None else "补检索（可返回空查询）"
        with progress(
            "生成评论检索查询" if evidence is None else "检查问题是否需要补检索"
        ):
            value = await self._chain.ainvoke(
                {
                    "stage": stage,
                    "question": question,
                    "aspect": aspect or "以用户问题为准",
                    "platforms": ", ".join(platforms),
                    "movie_context": movie_context or "未提供电影背景",
                    "evidence": json.dumps(
                        [
                            {
                                "platform": item.platform,
                                "review_id": item.review_id,
                                "target": item.target,
                                "subaspect": item.subaspect,
                                "opinion": item.opinion,
                                "quote": item.evidence_span,
                                "reason": item.reason,
                            }
                            for item in evidence or ()
                        ],
                        ensure_ascii=False,
                    ),
                }
            )
        return RetrievalQueryPlan.model_validate(value)


class OpinionRetriever:
    def __init__(
        self,
        store: EphemeralRagStore,
        embedder: EmbeddingProvider,
        query_planner: RetrievalQueryPlanner,
        reranker: OpinionReranker,
        config: RagConfig,
    ) -> None:
        self.store, self.embedder = store, embedder
        self.query_planner, self.reranker = query_planner, reranker
        self.config = config

    async def retrieve(
        self,
        *,
        index_id: str,
        question: str,
        aspect: str | None,
        platforms: Sequence[ReviewPlatform],
        movie_context: str = "",
    ) -> RetrievalArtifact:
        platforms = list(dict.fromkeys(platforms))
        if not platforms:
            raise ValueError("At least one platform is required")
        index = self.store.get_index(index_id)
        raw_plan = await self.query_planner.plan(
            question=question,
            aspect=aspect,
            platforms=platforms,
            movie_context=movie_context,
        )
        query_plan = self._sanitize_query_plan(raw_plan, platforms, question=question)
        record_event(
            "rag.query_plan",
            index_id=index_id,
            question=question,
            movie_context=movie_context,
            raw_plan=raw_plan,
            plan=query_plan,
        )
        queries = query_plan.queries
        pool: list[RankedEvidence] = []
        read: list[RetrievalCandidate] = []
        evidence = []
        rounds = 0
        for round_number in range(1, self.config.max_retrieval_rounds + 1):
            excluded = {
                i
                for i, chunk in enumerate(index.chunks)
                if any(
                    item.review_id == chunk.review_id
                    and any(
                        r.start <= chunk.start and chunk.end <= r.end
                        for r in item.context_ranges
                    )
                    for item in read
                )
            }
            if round_number > 1:
                available = [
                    p
                    for p in platforms
                    if any(i not in excluded for i in index.platform_indices.get(p, ()))
                ]
                if not available:
                    record_event(
                        "rag.followup_decision",
                        proceed=False,
                        reason="corpus_exhausted",
                    )
                    break
                followup = await self.query_planner.plan(
                    question=question,
                    aspect=aspect,
                    platforms=available,
                    movie_context=movie_context,
                    evidence=evidence,
                )
                queries = self._sanitize_query_plan(followup, available).queries
                record_event("rag.followup_queries", queries=queries)
                record_event(
                    "rag.followup_decision",
                    proceed=bool(queries),
                    reason="question_gap" if queries else "question_answered",
                )
                if not queries:
                    break
            fusion = await self._hybrid_fusion(index, queries, excluded)
            candidates = self._candidates(index, fusion, platforms)
            _log_candidates(index_id, round_number, candidates)
            if round_number > 1 and not candidates:
                break
            pool.extend(
                await self.reranker.rerank(
                    question=question,
                    aspect=aspect,
                    candidates=candidates,
                    known_scopes=list(
                        dict.fromkeys((e.target, e.subaspect) for e in pool)
                    ),
                    movie_context=movie_context,
                )
            )
            read.extend(candidates)
            evidence = select_evidence(
                pool,
                platforms=platforms,
                limit_per_platform=self.config.final_evidence_per_platform,
                minimum_reviews=self.config.min_relevant_per_platform,
            )
            coverage = coverage_report(
                evidence,
                platforms=platforms,
                minimum=self.config.min_relevant_per_platform,
            )
            rounds = round_number
            record_event(
                "rag.retrieval_round",
                round=round_number,
                pool_count=len(pool),
                selected=evidence,
                coverage=coverage,
            )
        result = RetrievalResult(
            retrieval_id=f"retrieval-{uuid4().hex}",
            index_id=index_id,
            question=question,
            aspect=aspect,
            platforms=platforms,
            movie_context=movie_context,
            query_plan=query_plan,
            evidence=evidence,
            coverage=coverage,
            rounds=rounds,
        )
        self.store.put_retrieval(result)
        record_event("rag.retrieval_completed", result=result)
        return RetrievalArtifact(
            retrieval_id=result.retrieval_id,
            index_id=index_id,
            evidence_count=len(evidence),
            rounds=rounds,
            coverage=coverage,
        )

    def _sanitize_query_plan(self, plan, platforms, *, question=None):
        queries = []
        for platform in platforms:
            matching = [q for q in plan.queries if q.platform == platform]
            queries.extend(matching[: self.config.max_query_rewrites])
            if not matching and question is not None:
                queries.append(QuerySpec(text=question, platform=platform))
        return RetrievalQueryPlan(queries=queries)

    async def _hybrid_fusion(self, index, queries, excluded=()):
        fusion: dict[int, float] = {}
        if not index.chunks or not queries:
            return fusion
        vectors = normalize_query_vectors(
            await self.embedder.embed_queries([q.text for q in queries]),
            expected_rows=len(queries),
            expected_dimension=index.dense_vectors.shape[1],
        )
        for query, vector in zip(queries, vectors, strict=True):
            allowed = [
                i
                for i in index.platform_indices.get(query.platform, ())
                if i not in excluded
            ]
            rankings = (
                _dense_search(index, vector, allowed, len(allowed)),
                [
                    i
                    for i, _ in index.bm25.search(
                        query.text,
                        allowed_indices=allowed,
                        k=len(allowed),
                        k1=self.config.bm25_k1,
                        b=self.config.bm25_b,
                    )
                ],
            )
            for ranking in rankings:
                reviews = set()
                for rank, i in enumerate(ranking, 1):
                    review_id = index.chunks[i].review_id
                    if (
                        review_id not in reviews
                        and len(reviews) >= self.config.candidate_cap_per_platform
                    ):
                        continue
                    reviews.add(review_id)
                    fusion[i] = fusion.get(i, 0.0) + 1 / (self.config.rrf_k + rank)
        return fusion

    def _candidates(self, index, fusion, platforms):
        result = []
        for platform in platforms:
            reviews = {}
            for i in sorted(fusion, key=lambda i: (-fusion[i], i)):
                chunk = index.chunks[i]
                if chunk.platform == platform:
                    reviews.setdefault(chunk.review_id, []).append(i)
            for review_id, hits in list(reviews.items())[
                : self.config.candidate_cap_per_platform
            ]:
                chunk = index.chunks[hits[0]]
                document = index.documents[review_id]
                ranges = reading_ranges(
                    document.text,
                    [index.chunks[i] for i in hits],
                    self.config.reading_context_chars,
                )
                result.append(
                    RetrievalCandidate(
                        chunk_id=chunk.chunk_id,
                        review_id=review_id,
                        platform=platform,
                        chunk_text=chunk.text,
                        full_text=document.text,
                        context_ranges=ranges,
                        title=document.title,
                        rating=document.rating,
                        helpful_votes=document.helpful_votes,
                        fusion_score=fusion[hits[0]],
                    )
                )
        return result


def _log_candidates(index_id, round_number, candidates):
    record_event(
        "rag.retrieval_candidates",
        index_id=index_id,
        round=round_number,
        counts=dict(Counter(item.platform for item in candidates)),
        candidates=[
            item.model_dump(
                include={
                    "chunk_id",
                    "review_id",
                    "platform",
                    "fusion_score",
                    "context_ranges",
                }
            )
            for item in candidates
        ],
    )


def _dense_search(index, query_vector, allowed_indices, k):
    if k < 1 or not allowed_indices:
        return []
    allowed = np.asarray(allowed_indices, dtype=np.int64)
    scores = index.dense_vectors[allowed] @ query_vector
    order = np.argsort(-scores, kind="stable")[:k]
    return [int(allowed[position]) for position in order]


__all__ = ["LLMRetrievalQueryPlanner", "OpinionRetriever", "RetrievalQueryPlanner"]
