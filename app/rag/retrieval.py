"""Platform-balanced hybrid retrieval for transient movie-review indexes."""

from __future__ import annotations

import math
from typing import Any, Protocol, Sequence
from uuid import uuid4

import numpy as np
from langchain_core.language_models import BaseChatModel
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel

from app.progress import progress
from app.rag.embedding import EmbeddingProvider
from app.rag.evidence import coverage_report, describe_gap, select_evidence
from app.rag.movie_context import MOVIE_CONTEXT_GUIDANCE
from app.rag.index import (
    EphemeralOpinionIndex,
    EphemeralRagStore,
    normalize_query_vectors,
)
from app.rag.models import (
    CoverageGap,
    HydePassage,
    HydePlan,
    QueryPurpose,
    QuerySpec,
    RankedEvidence,
    RetrievalArtifact,
    RetrievalCandidate,
    RetrievalQueryPlan,
    RetrievalResult,
    ReviewPlatform,
)
from app.rag.reranker import OpinionReranker
from config import RagConfig


QUERY_SYSTEM_PROMPT = """
你只为一部电影的评论检索生成语义查询，不选择检索算法或数值参数。

- 豆瓣查询使用中文，IMDb 查询使用英文。
- 先用 movie_context 确认人物和角色对应，再把用户问题转成观众可能实际使用的词。
  可扩展背景支持的演员名、角色名和别称；保留不含具体人物名的宽查询，避免只检索主演。
- 保留用户真正询问的评价方面，并给出少量可辨认的细分方面。
- subaspects 只列回答问题必需的 2–4 个、互不重叠的中文评价维度，两个平台共用；狭窄问题可更少。
- 同一维度的查询应同时寻找具体理由、适用条件与反例，避免只改写“好/不好”。
- 查询要覆盖直接表述、肯定、否定和复杂/矛盾观点；不要预设结论。
- 每条查询绑定 imdb 或 douban，以及 primary/exact/subaspect/positive/negative/mixed 之一。
- 不生成电影事实检索，不回答问题，不引用并不存在的评论。
""".strip() + "\n\n" + MOVIE_CONTEXT_GUIDANCE


HYDE_SYSTEM_PROMPT = """
你为召回不足的电影评论检索生成假设性段落。它们只作为 dense retrieval 查询，绝不是证据。

- 每条段落绑定给定缺口的 gap_index；IMDb 用英文，豆瓣用中文，不能跨用平台或评价对象。
- reason 缺口要寻找评论原文明示的理由或条件；comparison 缺口要查同一对象、同一维度。
- subaspect 缺口寻找该维度的直接评价；reviews 缺口寻找更多独立评论。
- 参考已有观点，补足缺少的信息，避免重复泛泛好坏。可使用不同立场，不预设必须存在反例。
- 用 movie_context 核对缺口中的人物/角色和措辞。背景中的事实不会填平评论证据缺口；
  不把百科对表演的评价改写成假设评论，也不把角色经历写成演员表演得到认可的原因。
- 只针对给定缺口生成；有多个缺口时优先分别覆盖，不对同一个缺口反复改写。
- 不声称这些文字来自真实用户，也不要使用具体用户、评分或数量。
""".strip() + "\n\n" + MOVIE_CONTEXT_GUIDANCE


class RetrievalQueryPlanner(Protocol):
    async def plan(
        self,
        *,
        question: str,
        aspect: str | None,
        platforms: Sequence[ReviewPlatform],
        movie_context: str = "",
    ) -> RetrievalQueryPlan: ...


class HydeGenerator(Protocol):
    async def generate(
        self,
        *,
        question: str,
        aspect: str | None,
        gaps: Sequence[CoverageGap],
        evidence: Sequence[RankedEvidence],
        movie_context: str = "",
    ) -> HydePlan: ...


class LLMRetrievalQueryPlanner:
    def __init__(self, llm: BaseChatModel) -> None:
        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", QUERY_SYSTEM_PROMPT),
                (
                    "human",
                    "用户问题：{question}\n"
                    "movie_context（电影背景）：\n{movie_context}\n\n"
                    "明确方面：{aspect}\n"
                    "目标平台：{platforms}",
                ),
            ]
        )
        output = llm.with_structured_output(
            RetrievalQueryPlan,
            method="json_schema",
            strict=True,
        )
        self._chain = prompt | output

    async def plan(
        self,
        *,
        question: str,
        aspect: str | None,
        platforms: Sequence[ReviewPlatform],
        movie_context: str = "",
    ) -> RetrievalQueryPlan:
        with progress("生成评论检索查询"):
            value = await self._chain.ainvoke(
                {
                    "question": question,
                    "movie_context": movie_context or "未提供电影背景",
                    "aspect": aspect or "未单独指定，以用户问题为准",
                    "platforms": ", ".join(platforms),
                }
            )
        return RetrievalQueryPlan.model_validate(_model_data(value))


class LLMHydeGenerator:
    def __init__(self, llm: BaseChatModel) -> None:
        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", HYDE_SYSTEM_PROMPT),
                (
                    "human",
                    "用户问题：{question}\n"
                    "movie_context（电影背景）：\n{movie_context}\n\n"
                    "明确方面：{aspect}\n"
                    "具体缺口（编号必须原样使用）：\n{gaps}\n\n"
                    "已有观点（供确定检索方向）：\n{evidence}",
                ),
            ]
        )
        output = llm.with_structured_output(
            HydePlan,
            method="json_schema",
            strict=True,
        )
        self._chain = prompt | output

    async def generate(
        self,
        *,
        question: str,
        aspect: str | None,
        gaps: Sequence[CoverageGap],
        evidence: Sequence[RankedEvidence],
        movie_context: str = "",
    ) -> HydePlan:
        with progress(f"为 {len(gaps)} 项覆盖缺口生成补检索查询"):
            value = await self._chain.ainvoke(
                {
                    "question": question,
                    "movie_context": movie_context or "未提供电影背景",
                    "aspect": aspect or "未单独指定，以用户问题为准",
                    "gaps": "\n".join(f"{i}. {describe_gap(gap)}" for i, gap in enumerate(gaps, 1)),
                    "evidence": "\n".join(
                        f"{item.platform} / {item.target} / {item.subaspect}: {item.opinion}"
                        for item in evidence
                    ),
                }
            )
        return HydePlan.model_validate(_model_data(value))


class OpinionRetriever:
    """Run BM25+dense+RRF, LLM rerank, then at most one HyDE fallback."""

    def __init__(
        self,
        store: EphemeralRagStore,
        embedder: EmbeddingProvider,
        query_planner: RetrievalQueryPlanner,
        reranker: OpinionReranker,
        hyde_generator: HydeGenerator,
        config: RagConfig,
    ) -> None:
        self.store = store
        self.embedder = embedder
        self.query_planner = query_planner
        self.reranker = reranker
        self.hyde_generator = hyde_generator
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
        selected_platforms = list(dict.fromkeys(platforms))
        if not selected_platforms:
            raise ValueError("At least one platform is required")
        index = self.store.get_index(index_id)
        raw_plan = await self.query_planner.plan(
            question=question,
            aspect=aspect,
            platforms=selected_platforms,
            movie_context=movie_context,
        )
        query_plan = self._sanitize_query_plan(
            raw_plan,
            question=question,
            aspect=aspect,
            platforms=selected_platforms,
        )
        fusion = await self._hybrid_fusion(index, query_plan.queries)
        candidates = self._candidates(index, fusion, selected_platforms)
        pool = await self.reranker.rerank(
            question=question,
            aspect=aspect,
            candidates=candidates,
            subaspects=query_plan.subaspects,
            movie_context=movie_context,
        )
        seen_chunks = {item.chunk_id for item in candidates}
        evidence = select_evidence(
            pool, platforms=selected_platforms,
            limit_per_platform=self.config.final_evidence_per_platform,
            minimum_reviews=self.config.min_relevant_per_platform,
        )
        coverage = coverage_report(
            evidence,
            platforms=selected_platforms,
            subaspects=query_plan.subaspects,
            minimum=self.config.min_relevant_per_platform,
        )
        rounds = 1

        available_platforms = {chunk.platform for chunk in index.chunks if chunk.chunk_id not in seen_chunks}
        gaps = [gap for gap in coverage.gaps if gap.platform in available_platforms]
        if gaps and self.config.max_retrieval_rounds > 1:
            hyde_plan = await self.hyde_generator.generate(
                question=question,
                aspect=aspect,
                gaps=gaps,
                evidence=evidence,
                movie_context=movie_context,
            )
            passages = self._sanitize_hyde(hyde_plan.passages, gaps)
            if passages:
                fallback = await self._hyde_fusion(index, passages, gaps, seen_chunks)
                candidates = self._candidates(index, fallback, selected_platforms)
                if candidates:
                    pool.extend(await self.reranker.rerank(
                        question=question, aspect=aspect, candidates=candidates,
                        subaspects=query_plan.subaspects,
                        known_targets=list(dict.fromkeys(item.target for item in pool)),
                        movie_context=movie_context,
                    ))
                evidence = select_evidence(
                    pool, platforms=selected_platforms,
                    limit_per_platform=self.config.final_evidence_per_platform,
                    minimum_reviews=self.config.min_relevant_per_platform,
                )
                coverage = coverage_report(
                    evidence,
                    platforms=selected_platforms,
                    subaspects=query_plan.subaspects,
                    minimum=self.config.min_relevant_per_platform,
                )
                rounds = 2

        result = RetrievalResult(
            retrieval_id=f"retrieval-{uuid4().hex}",
            index_id=index_id,
            question=question,
            platforms=selected_platforms,
            movie_context=movie_context,
            query_plan=query_plan,
            evidence=evidence,
            coverage=coverage,
            rounds=rounds,
        )
        self.store.put_retrieval(result)
        return RetrievalArtifact(
            retrieval_id=result.retrieval_id,
            index_id=index_id,
            evidence_count=len(result.evidence),
            rounds=rounds,
            coverage=coverage,
        )

    def _sanitize_query_plan(
        self,
        plan: RetrievalQueryPlan,
        *,
        question: str,
        aspect: str | None,
        platforms: Sequence[ReviewPlatform],
    ) -> RetrievalQueryPlan:
        queries: list[QuerySpec] = []
        for platform in platforms:
            matching = [query for query in plan.queries if query.platform == platform]
            queries.extend(matching[: self.config.max_query_rewrites])
            if not matching:
                queries.append(
                    QuerySpec(
                        text=question,
                        platform=platform,
                        purpose=QueryPurpose.PRIMARY,
                    )
                )
        return RetrievalQueryPlan(
            aspect=aspect if aspect is not None else plan.aspect,
            subaspects=plan.subaspects or [aspect or plan.aspect or "整体评价"],
            queries=queries,
        )

    async def _hybrid_fusion(
        self,
        index: EphemeralOpinionIndex,
        queries: Sequence[QuerySpec],
    ) -> dict[int, float]:
        fusion: dict[int, float] = {}
        if not index.chunks:
            return fusion
        vectors = normalize_query_vectors(
            await self.embedder.embed_queries([query.text for query in queries]),
            expected_rows=len(queries),
            expected_dimension=index.dense_vectors.shape[1],
        )
        for query, vector in zip(queries, vectors, strict=True):
            allowed = index.platform_indices.get(query.platform, ())
            k = self._first_stage_k(len(allowed))
            _add_ranking(
                fusion,
                _dense_search(index, vector, allowed, k),
                rrf_k=self.config.rrf_k,
            )
            _add_ranking(
                fusion,
                [
                    item[0]
                    for item in index.bm25.search(
                        query.text,
                        allowed_indices=allowed,
                        k=k,
                        k1=self.config.bm25_k1,
                        b=self.config.bm25_b,
                    )
                ],
                rrf_k=self.config.rrf_k,
            )
        return fusion

    async def _hyde_fusion(
        self,
        index: EphemeralOpinionIndex,
        passages: Sequence[HydePassage],
        gaps: Sequence[CoverageGap],
        seen_chunks: set[str],
    ) -> dict[int, float]:
        fusion: dict[int, float] = {}
        vectors = normalize_query_vectors(
            await self.embedder.embed_queries([item.text for item in passages]),
            expected_rows=len(passages),
            expected_dimension=index.dense_vectors.shape[1],
        )
        for passage, vector in zip(passages, vectors, strict=True):
            platform = gaps[passage.gap_index - 1].platform
            allowed = [i for i in index.platform_indices.get(platform, ())
                       if index.chunks[i].chunk_id not in seen_chunks]
            _add_ranking(
                fusion,
                _dense_search(
                    index,
                    vector,
                    allowed,
                    self._first_stage_k(len(allowed)),
                ),
                rrf_k=self.config.rrf_k,
            )
        return fusion

    def _candidates(
        self,
        index: EphemeralOpinionIndex,
        fusion: dict[int, float],
        platforms: Sequence[ReviewPlatform],
    ) -> list[RetrievalCandidate]:
        result: list[RetrievalCandidate] = []
        for platform in platforms:
            ranked_indices = sorted(
                (
                    (chunk_index, value)
                    for chunk_index, value in fusion.items()
                    if index.chunks[chunk_index].platform == platform
                ),
                key=lambda item: (-item[1], item[0]),
            )
            for chunk_index, value in ranked_indices[: self.config.candidate_cap_per_platform]:
                chunk = index.chunks[chunk_index]
                document = index.documents[chunk.review_id]
                result.append(
                    RetrievalCandidate(
                        chunk_id=chunk.chunk_id,
                        review_id=document.review_id,
                        platform=platform,
                        chunk_text=chunk.text,
                        full_text=document.text,
                        title=document.title,
                        rating=document.rating,
                        helpful_votes=document.helpful_votes,
                        fusion_score=value,
                    )
                )
        return result

    def _sanitize_hyde(
        self,
        passages: Sequence[HydePassage],
        gaps: Sequence[CoverageGap],
    ) -> list[HydePassage]:
        result: list[HydePassage] = []
        counts: dict[str, int] = {}
        seen = set()
        for passage in passages:
            if passage.gap_index > len(gaps):
                raise ValueError("HyDE passage references an unknown coverage gap")
            platform = gaps[passage.gap_index - 1].platform
            key = (platform, passage.text.casefold())
            if key not in seen and counts.get(platform, 0) < self.config.hyde_variants:
                result.append(passage)
                seen.add(key)
                counts[platform] = counts.get(platform, 0) + 1
        return result

    def _first_stage_k(self, platform_count: int) -> int:
        if platform_count <= 0:
            return 0
        requested = max(
            self.config.first_stage_min_k,
            math.ceil(platform_count * self.config.first_stage_ratio),
        )
        return min(platform_count, self.config.first_stage_max_k, requested)


def _dense_search(
    index: EphemeralOpinionIndex,
    query_vector: np.ndarray,
    allowed_indices: Sequence[int],
    k: int,
) -> list[int]:
    if k < 1 or not allowed_indices:
        return []
    allowed = np.asarray(allowed_indices, dtype=np.int64)
    scores = index.dense_vectors[allowed] @ query_vector
    order = np.argsort(-scores, kind="stable")[:k]
    return [int(allowed[position]) for position in order]


def _add_ranking(
    fusion: dict[int, float],
    ranking: Sequence[int],
    *,
    rrf_k: int,
) -> None:
    for rank, chunk_index in enumerate(ranking, start=1):
        fusion[chunk_index] = fusion.get(chunk_index, 0.0) + 1.0 / (
            rrf_k + rank
        )


def _model_data(value: dict[str, Any] | BaseModel) -> dict[str, Any]:
    if isinstance(value, BaseModel):
        return value.model_dump()
    return value


__all__ = [
    "HydeGenerator",
    "LLMHydeGenerator",
    "LLMRetrievalQueryPlanner",
    "OpinionRetriever",
    "RetrievalQueryPlanner",
]
