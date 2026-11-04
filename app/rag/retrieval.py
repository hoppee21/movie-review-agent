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
    RankedPassage,
    RetrievalArtifact,
    RetrievalCandidate,
    RetrievalQueryPlan,
    RetrievalResult,
    ReviewPlatform,
)
from app.rag.movie_context import MOVIE_CONTEXT_GUIDANCE
from app.rag.passages import reading_ranges
from app.rag.reranker import PassageReranker
from app.run_log import record_event
from config import RagConfig

QUERY_SYSTEM_PROMPT = (
    """
把用户的问题改写为这部电影的评论检索查询。IMDb 用英文，豆瓣用中文。
保留用户明确指定的对象、方面、比较关系和限制；通常每个平台一条，必要时最多两条。
首次检索先参考所附维基百科条目，从中选择与问题直接相关的事实，把它们转成评论里可能出现的检索词。
例如表演问题可用演员与角色的对应关系，场景或制作问题可用条目明确记载的场景或手法。
宽问题保留覆盖整题的中性查询；有用的具体线索可另写一条。没有相关百科资料时按用户问题规划。
evidence_focus 用一句话说明单条评论里要找什么直接评价、理由或条件，简短但直接的评价也有用。
analysis_questions 最多两个，写成需要对照评论才能回答的开放问题，简单任务可为空。
不能把百科中的动机、评价或技术细节写成预定结论，不能预设两平台存在差异。

补检索对照首轮计划、百科背景和已选评论原文。只为影响回答的具体缺口生成定向查询。
沿用首轮的重排重点，evidence_focus 留空。已有原文支持更具体的分析问题时更新 analysis_questions，
否则留空以沿用首轮问题。证据足以在其范围内回答时返回 queries=[]，即使如此也可细化分析问题。
不为凑数量、预想分类、正反立场或每个偶然出现的演员补齐证据。来源数量不代表语义覆盖完整。
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
        evidence: Sequence[RankedPassage] | None = None,
        initial_plan: RetrievalQueryPlan | None = None,
    ) -> RetrievalQueryPlan: ...


class LLMRetrievalQueryPlanner:
    def __init__(self, llm: BaseChatModel) -> None:
        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", QUERY_SYSTEM_PROMPT),
                (
                    "human",
                    "阶段：{stage}\n用户问题：{question}\n明确方面：{aspect}\n"
                    "目标平台：{platforms}\n维基百科资料：{movie_context}\n"
                    "首轮计划：{initial_plan}\n已有评论原文：{evidence}",
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
        evidence: Sequence[RankedPassage] | None = None,
        initial_plan: RetrievalQueryPlan | None = None,
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
                    "initial_plan": initial_plan.model_dump_json()
                    if initial_plan
                    else "无",
                    "evidence": json.dumps(
                        {
                            "source_counts": dict(
                                Counter(item.platform for item in evidence or ())
                            ),
                            "sources": [
                                {
                                    "source_id": item.source_id,
                                    "platform": item.platform,
                                    "review_id": item.review_id,
                                    "title": item.title,
                                    "contexts": item.contexts,
                                }
                                for item in evidence or ()
                            ],
                        },
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
        reranker: PassageReranker,
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
        rerank_query = question
        if query_plan.evidence_focus:
            rerank_query += f"\n评论检索重点：{query_plan.evidence_focus}"
        evidence: list[RankedPassage] = []
        rounds = 0
        for round_number in range(1, self.config.max_retrieval_rounds + 1):
            excluded = {
                i
                for i, chunk in enumerate(index.chunks)
                if any(
                    item.platform == chunk.platform
                    and item.review_id == chunk.review_id
                    and any(
                        r.start <= chunk.start and chunk.end <= r.end
                        for r in item.context_ranges
                    )
                    for item in evidence
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
                    initial_plan=query_plan,
                )
                queries = self._sanitize_query_plan(followup, available).queries
                if followup.analysis_questions:
                    query_plan = query_plan.model_copy(
                        update={"analysis_questions": followup.analysis_questions}
                    )
                record_event(
                    "rag.followup_queries",
                    queries=queries,
                    analysis_questions=query_plan.analysis_questions,
                )
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
            # Scores from separate rerank requests are not comparable. Re-score
            # retained sources together with the follow-up hits before selecting.
            candidates = self._merge_candidates(candidates, evidence)
            pool = await self.reranker.rerank(
                question=rerank_query,
                aspect=aspect,
                candidates=candidates,
            )
            evidence = select_evidence(
                pool,
                platforms=platforms,
                limit_per_platform=self.config.final_evidence_per_platform,
            )
            coverage = coverage_report(
                evidence,
                platforms=platforms,
                minimum=self.config.min_sources_per_platform,
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
        return plan.model_copy(update={"queries": queries})

    def _merge_candidates(self, candidates, retained):
        """Merge repeated reviews, prioritizing newly found reading windows."""
        merged = {(item.platform, item.review_id): item for item in candidates}
        for item in retained:
            key = (item.platform, item.review_id)
            if key not in merged:
                merged[key] = RetrievalCandidate.model_validate(item.model_dump())
                continue
            current = merged[key]
            if current.full_text != item.full_text:
                raise ValueError("A review changed within one retrieval run")
            ranges = reading_ranges(
                current.full_text,
                [*current.context_ranges, *item.context_ranges],
                self.config.reading_context_chars,
            )
            merged[key] = current.model_copy(update={"context_ranges": ranges})
        return list(merged.values())

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
