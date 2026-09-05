"""Typed contracts shared by the transient movie-opinion RAG pipeline."""

from enum import StrEnum
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator


ReviewPlatform = Literal["imdb", "douban"]


class QueryPurpose(StrEnum):
    PRIMARY = "primary"
    EXACT = "exact"
    SUBASPECT = "subaspect"
    POSITIVE = "positive"
    NEGATIVE = "negative"
    MIXED = "mixed"


class OpinionStance(StrEnum):
    POSITIVE = "positive"
    NEGATIVE = "negative"
    MIXED = "mixed"
    UNCLEAR = "unclear"


class ReviewDocument(BaseModel):
    review_id: str = Field(min_length=1)
    platform: ReviewPlatform
    text: str = Field(min_length=1)
    title: str | None = None
    author: str | None = None
    date: str | None = None
    rating: float | None = None
    helpful_votes: int | None = None
    total_votes: int | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class ReviewChunk(BaseModel):
    chunk_id: str = Field(min_length=1)
    review_id: str = Field(min_length=1)
    platform: ReviewPlatform
    text: str = Field(min_length=1)
    embed_text: str = Field(min_length=1)


class BuildIndexInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    imdb_reviews_path: str = Field(
        min_length=1,
        description="Local JSON path returned for IMDb by collect_movie_reviews.",
    )
    douban_reviews_path: str = Field(
        min_length=1,
        description="Local JSON path returned for Douban by collect_movie_reviews.",
    )

    @field_validator("imdb_reviews_path", "douban_reviews_path")
    @classmethod
    def clean_paths(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("review path cannot be blank")
        return value.strip()


class IndexArtifact(BaseModel):
    index_id: str = Field(min_length=1)
    corpus_fingerprint: str = Field(min_length=1)
    review_count: int = Field(ge=0)
    chunk_count: int = Field(ge=0)
    platform_counts: dict[str, int]
    embedding_model: str = Field(min_length=1)


class QuerySpec(BaseModel):
    text: str = Field(min_length=1, max_length=500)
    platform: ReviewPlatform
    purpose: QueryPurpose

    @field_validator("text")
    @classmethod
    def clean_query(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("query text cannot be blank")
        return value.strip()


class RetrievalQueryPlan(BaseModel):
    aspect: str | None
    subaspects: list[str] = Field(max_length=6)
    queries: list[QuerySpec] = Field(min_length=1, max_length=18)

    @field_validator("aspect")
    @classmethod
    def clean_plan_aspect(cls, value: str | None) -> str | None:
        return value.strip() if value and value.strip() else None

    @field_validator("subaspects")
    @classmethod
    def unique_subaspects(cls, value: list[str]) -> list[str]:
        return _unique_text(value)

    @field_validator("queries")
    @classmethod
    def unique_queries(cls, value: list[QuerySpec]) -> list[QuerySpec]:
        seen: set[tuple[str, str]] = set()
        unique: list[QuerySpec] = []
        for item in value:
            key = (item.platform, item.text.casefold())
            if key not in seen:
                seen.add(key)
                unique.append(item)
        return unique


class HydePassage(BaseModel):
    platform: ReviewPlatform
    stance: OpinionStance
    text: str = Field(min_length=1, max_length=1500)

    @field_validator("text")
    @classmethod
    def clean_passage(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("HyDE passage cannot be blank")
        return value.strip()


class HydePlan(BaseModel):
    passages: list[HydePassage] = Field(min_length=1, max_length=6)


class RetrievalCandidate(BaseModel):
    chunk_id: str
    review_id: str
    platform: ReviewPlatform
    chunk_text: str
    full_text: str
    title: str | None = None
    rating: float | None = None
    helpful_votes: int | None = None
    fusion_score: float


class RerankJudgement(BaseModel):
    chunk_id: str
    relevance: int = Field(ge=0, le=3)
    evidence_quality: int = Field(ge=0, le=2)
    stance: OpinionStance
    subaspect: str | None = Field(max_length=80)
    evidence_span: str | None = Field(max_length=600)


class RerankResponse(BaseModel):
    items: list[RerankJudgement]


class RankedEvidence(BaseModel):
    evidence_id: str
    chunk_id: str
    review_id: str
    platform: ReviewPlatform
    chunk_text: str
    full_text: str
    title: str | None = None
    rating: float | None = None
    helpful_votes: int | None = None
    fusion_score: float
    relevance: int = Field(ge=0, le=3)
    evidence_quality: int = Field(ge=0, le=2)
    stance: OpinionStance
    subaspect: str | None
    evidence_span: str | None


class CoverageReport(BaseModel):
    sufficient: bool
    relevant_by_platform: dict[str, int]
    stance_counts: dict[str, dict[str, int]]
    covered_subaspects: list[str]
    missing_subaspects: list[str]
    missing_platforms: list[str]
    reason: str | None


class RetrievalResult(BaseModel):
    retrieval_id: str
    index_id: str
    question: str = Field(min_length=1)
    platforms: list[ReviewPlatform] = Field(min_length=1, max_length=2)
    query_plan: RetrievalQueryPlan
    evidence: list[RankedEvidence]
    coverage: CoverageReport
    rounds: int = Field(ge=1)

    @field_validator("platforms")
    @classmethod
    def unique_result_platforms(
        cls,
        value: list[ReviewPlatform],
    ) -> list[ReviewPlatform]:
        return list(dict.fromkeys(value))


class QueryOpinionsInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    index_id: str = Field(
        min_length=1,
        description="Transient index identifier returned by build_opinion_index.",
    )
    question: str = Field(
        min_length=1,
        description="Movie-opinion question produced by RequestAnalyzer.",
    )
    aspect: str | None = Field(
        description="Explicit evaluation aspect from PlanIntent, or null.",
    )
    platforms: list[ReviewPlatform] = Field(
        min_length=1,
        max_length=2,
        description="Only the platforms requested by the user.",
    )

    @field_validator("platforms")
    @classmethod
    def unique_platforms(
        cls,
        value: list[ReviewPlatform],
    ) -> list[ReviewPlatform]:
        return list(dict.fromkeys(value))

    @field_validator("index_id", "question")
    @classmethod
    def clean_required_text(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("value cannot be blank")
        return value.strip()

    @field_validator("aspect")
    @classmethod
    def clean_aspect(cls, value: str | None) -> str | None:
        return value.strip() if value and value.strip() else None


class RetrievalArtifact(BaseModel):
    retrieval_id: str
    index_id: str
    evidence_count: int = Field(ge=0)
    rounds: int = Field(ge=1)
    coverage: CoverageReport


class AggregateOpinionsInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    retrieval_id: str = Field(
        min_length=1,
        description="Transient result identifier returned by query_movie_opinions.",
    )

    @field_validator("retrieval_id")
    @classmethod
    def clean_retrieval_id(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("retrieval_id cannot be blank")
        return value.strip()


class OpinionClusterDraft(BaseModel):
    label: str = Field(min_length=1)
    summary: str = Field(min_length=1)
    stance: OpinionStance
    platforms: list[ReviewPlatform]
    evidence_ids: list[str]


class AggregationDraft(BaseModel):
    conclusion: str = Field(min_length=1)
    clusters: list[OpinionClusterDraft]
    platform_comparison: str | None
    limitations: list[str]


class EvidenceCard(BaseModel):
    evidence_id: str
    platform: ReviewPlatform
    review_id: str
    quote: str
    full_text: str
    rating: float | None = None
    helpful_votes: int | None = None


class OpinionCluster(BaseModel):
    label: str
    summary: str
    stance: OpinionStance
    platforms: list[ReviewPlatform]
    evidence_count: int = Field(
        ge=0,
        description="Distinct cited reviews in this cluster, not population size.",
    )
    evidence_ids: list[str]


class MovieAnswer(BaseModel):
    conclusion: str
    clusters: list[OpinionCluster]
    stance_counts: dict[str, dict[str, int]] = Field(
        description="Counts among retrieved reviews judged directly relevant."
    )
    platform_comparison: str | None
    evidence: list[EvidenceCard]
    coverage: CoverageReport
    limitations: list[str]


def _unique_text(values: list[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for raw_value in values:
        value = raw_value.strip()
        key = value.casefold()
        if value and key not in seen:
            seen.add(key)
            result.append(value)
    return result
