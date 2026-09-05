"""Platform-balanced hybrid retrieval for transient movie-review indexes."""

from __future__ import annotations

import math
from collections import Counter
from typing import Any, Protocol, Sequence
from uuid import uuid4

import numpy as np
from langchain_core.language_models import BaseChatModel
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel

from app.rag.embedding import EmbeddingProvider
from app.rag.index import (
    EphemeralOpinionIndex,
    EphemeralRagStore,
    normalize_query_vectors,
)
from app.rag.models import (
    CoverageReport,
    HydePassage,
    HydePlan,
    OpinionStance,
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
- 保留用户真正询问的评价方面，并给出少量可辨认的细分方面。
- 查询要覆盖直接表述、肯定、否定和复杂/矛盾观点；不要预设结论。
- 每条查询绑定 imdb 或 douban，以及 primary/exact/subaspect/positive/negative/mixed 之一。
- 不生成电影事实检索，不回答问题，不引用并不存在的评论。
""".strip()


HYDE_SYSTEM_PROMPT = """
你为召回不足的电影评论检索生成假设性段落。它们只作为 dense retrieval 查询，绝不是证据。

- IMDb 用英文，豆瓣用中文。
- 同时覆盖肯定、否定和复杂观点，避免只生成单一立场。
- 紧扣用户所问方面和缺失的细分方面。
- 不声称这些文字来自真实用户，也不要使用具体用户、评分或数量。
""".strip()


class RetrievalQueryPlanner(Protocol):
    async def plan(
        self,
        *,
        question: str,
        aspect: str | None,
        platforms: Sequence[ReviewPlatform],
    ) -> RetrievalQueryPlan: ...


class HydeGenerator(Protocol):
    async def generate(
        self,
        *,
        question: str,
        aspect: str | None,
        platforms: Sequence[ReviewPlatform],
        missing_subaspects: Sequence[str],
    ) -> HydePlan: ...


class LLMRetrievalQueryPlanner:
    def __init__(self, llm: BaseChatModel) -> None:
        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", QUERY_SYSTEM_PROMPT),
                (
                    "human",
                    "用户问题：{question}\n"
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
    ) -> RetrievalQueryPlan:
        value = await self._chain.ainvoke(
            {
                "question": question,
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
                    "明确方面：{aspect}\n"
                    "需要补充的平台：{platforms}\n"
                    "尚未覆盖的细分方面：{missing_subaspects}",
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
        platforms: Sequence[ReviewPlatform],
        missing_subaspects: Sequence[str],
    ) -> HydePlan:
        value = await self._chain.ainvoke(
            {
                "question": question,
                "aspect": aspect or "未单独指定，以用户问题为准",
                "platforms": ", ".join(platforms),
                "missing_subaspects": ", ".join(missing_subaspects) or "无",
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
    ) -> RetrievalArtifact:
        selected_platforms = list(dict.fromkeys(platforms))
        if not selected_platforms:
            raise ValueError("At least one platform is required")
        index = self.store.get_index(index_id)
        raw_plan = await self.query_planner.plan(
            question=question,
            aspect=aspect,
            platforms=selected_platforms,
        )
        query_plan = self._sanitize_query_plan(
            raw_plan,
            question=question,
            aspect=aspect,
            platforms=selected_platforms,
        )
        fusion = await self._hybrid_fusion(index, query_plan.queries)
        candidates = self._candidates(index, fusion, selected_platforms)
        evidence = await self._rerank(
            question=question,
            aspect=aspect,
            candidates=candidates,
            platforms=selected_platforms,
        )
        coverage = coverage_report(
            evidence,
            platforms=selected_platforms,
            subaspects=query_plan.subaspects,
            minimum=self.config.min_relevant_per_platform,
        )
        rounds = 1

        if (
            not coverage.sufficient
            and self.config.max_retrieval_rounds > 1
            and index.chunks
        ):
            hyde_plan = await self.hyde_generator.generate(
                question=question,
                aspect=aspect,
                platforms=selected_platforms,
                missing_subaspects=coverage.missing_subaspects,
            )
            passages = self._sanitize_hyde(
                hyde_plan.passages,
                selected_platforms,
            )
            if passages:
                fallback = await self._hyde_fusion(index, passages)
                _merge_fusion(fusion, fallback)
                candidates = self._candidates(index, fusion, selected_platforms)
                evidence = await self._rerank(
                    question=question,
                    aspect=aspect,
                    candidates=candidates,
                    platforms=selected_platforms,
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
            subaspects=plan.subaspects,
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
    ) -> dict[int, float]:
        fusion: dict[int, float] = {}
        vectors = normalize_query_vectors(
            await self.embedder.embed_queries([item.text for item in passages]),
            expected_rows=len(passages),
            expected_dimension=index.dense_vectors.shape[1],
        )
        for passage, vector in zip(passages, vectors, strict=True):
            allowed = index.platform_indices.get(passage.platform, ())
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
            seen_reviews: set[str] = set()
            for chunk_index, value in ranked_indices:
                chunk = index.chunks[chunk_index]
                if chunk.review_id in seen_reviews:
                    continue
                seen_reviews.add(chunk.review_id)
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
                if len(seen_reviews) >= self.config.candidate_cap_per_platform:
                    break
        return result

    async def _rerank(
        self,
        *,
        question: str,
        aspect: str | None,
        candidates: Sequence[RetrievalCandidate],
        platforms: Sequence[ReviewPlatform],
    ) -> list[RankedEvidence]:
        ranked = await self.reranker.rerank(
            question=question,
            aspect=aspect,
            candidates=candidates,
        )
        kept: list[RankedEvidence] = []
        for platform in platforms:
            platform_items = [item for item in ranked if item.platform == platform]
            kept.extend(platform_items[: self.config.rerank_top_n_per_platform])
        return kept

    def _sanitize_hyde(
        self,
        passages: Sequence[HydePassage],
        platforms: Sequence[ReviewPlatform],
    ) -> list[HydePassage]:
        result: list[HydePassage] = []
        for platform in platforms:
            matching = [item for item in passages if item.platform == platform]
            result.extend(matching[: self.config.hyde_variants])
        return result

    def _first_stage_k(self, platform_count: int) -> int:
        if platform_count <= 0:
            return 0
        requested = max(
            self.config.first_stage_min_k,
            math.ceil(platform_count * self.config.first_stage_ratio),
        )
        return min(platform_count, self.config.first_stage_max_k, requested)


def coverage_report(
    evidence: Sequence[RankedEvidence],
    *,
    platforms: Sequence[ReviewPlatform],
    subaspects: Sequence[str],
    minimum: int,
) -> CoverageReport:
    """Measure review-level evidence coverage, separately for each platform."""

    relevant = _unique_relevant_reviews(evidence)
    by_platform = Counter(item.platform for item in relevant)
    stance_counts: dict[str, dict[str, int]] = {}
    for platform in platforms:
        counts = Counter(
            item.stance.value
            for item in relevant
            if item.platform == platform
        )
        stance_counts[platform] = {
            stance.value: counts.get(stance.value, 0)
            for stance in OpinionStance
        }

    observed = [item.subaspect for item in relevant if item.subaspect]
    covered = [
        expected
        for expected in subaspects
        if any(_same_aspect(expected, actual) for actual in observed)
    ]
    missing_subaspects = [item for item in subaspects if item not in covered]
    missing_platforms = [
        platform
        for platform in platforms
        if by_platform.get(platform, 0) < minimum
    ]
    no_subaspect_hit = bool(subaspects) and not covered
    sufficient = not missing_platforms and not no_subaspect_hit
    reasons: list[str] = []
    if missing_platforms:
        reasons.append(
            "直接相关评论不足的平台：" + ", ".join(missing_platforms)
        )
    if no_subaspect_hit:
        reasons.append(
            "没有直接证据覆盖问题拆出的细分方面："
            + ", ".join(missing_subaspects)
        )
    return CoverageReport(
        sufficient=sufficient,
        relevant_by_platform={
            platform: by_platform.get(platform, 0)
            for platform in platforms
        },
        stance_counts=stance_counts,
        covered_subaspects=covered,
        missing_subaspects=missing_subaspects,
        missing_platforms=missing_platforms,
        reason="；".join(reasons) or None,
    )


def _unique_relevant_reviews(
    evidence: Sequence[RankedEvidence],
) -> list[RankedEvidence]:
    seen: set[tuple[str, str]] = set()
    result: list[RankedEvidence] = []
    for item in sorted(
        evidence,
        key=lambda value: (
            -value.relevance,
            -value.evidence_quality,
            -value.fusion_score,
        ),
    ):
        key = (item.platform, item.review_id)
        if item.relevance >= 2 and key not in seen:
            seen.add(key)
            result.append(item)
    return result


def _same_aspect(expected: str, observed: str) -> bool:
    left = expected.strip().casefold()
    right = observed.strip().casefold()
    return bool(left and right and (left in right or right in left))


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


def _merge_fusion(
    base: dict[int, float],
    addition: dict[int, float],
) -> None:
    for chunk_index, value in addition.items():
        base[chunk_index] = base.get(chunk_index, 0.0) + value


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
    "coverage_report",
]
