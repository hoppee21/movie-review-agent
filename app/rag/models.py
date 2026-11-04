"""Typed contracts shared by the transient movie-opinion RAG pipeline."""

from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

ReviewPlatform = Literal["imdb", "douban"]


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


class TextRange(BaseModel):
    """Half-open character offsets into the unchanged review text."""

    start: int = Field(ge=0)
    end: int = Field(gt=0)

    @model_validator(mode="after")
    def ordered(self):
        if self.end <= self.start:
            raise ValueError("Text range must have start < end")
        return self


class ReviewChunk(TextRange):
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

    @field_validator("text")
    @classmethod
    def clean_query(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("query text cannot be blank")
        return value.strip()


class RetrievalQueryPlan(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True)

    queries: list[QuerySpec] = Field(max_length=4)
    evidence_focus: str = Field(
        default_factory=str,
        max_length=300,
        description="What to look for in a single review when reranking.",
    )
    analysis_questions: list[Annotated[str, Field(min_length=1, max_length=240)]] = (
        Field(
            default_factory=list,
            max_length=2,
            description="Open questions about relationships to verify across the selected reviews.",
        )
    )

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
    context_ranges: list[TextRange] = Field(default_factory=list)

    @model_validator(mode="after")
    def ranges_within_review(self):
        if any(span.end > len(self.full_text) for span in self.context_ranges):
            raise ValueError("Reading context is outside the review")
        return self

    @property
    def contexts(self) -> list[str]:
        # Older saved evidence has no offsets; its original chunk remains auditable.
        return [self.full_text[r.start : r.end] for r in self.context_ranges] or [
            self.chunk_text
        ]

    def contains_quote(self, quote: str) -> bool:
        return (
            bool(quote)
            and quote in self.full_text
            and any(quote in text for text in self.contexts)
        )


class RankedPassage(RetrievalCandidate):
    """An original review context ranked for relevance, not an extracted opinion."""

    source_id: str = Field(min_length=1)
    rerank_score: float = Field(ge=0, le=1, allow_inf_nan=False)


class CoverageReport(BaseModel):
    """Source-count diagnostics, not a semantic guarantee of answer completeness."""

    minimum_reviews: int = Field(ge=1)
    sufficient: bool
    source_counts: dict[str, int]
    missing_platforms: list[str]
    reason: str | None


class RetrievalResult(BaseModel):
    retrieval_id: str
    index_id: str
    question: str = Field(min_length=1)
    aspect: str | None = None
    movie_context: str = ""
    platforms: list[ReviewPlatform] = Field(min_length=1, max_length=2)
    query_plan: RetrievalQueryPlan
    evidence: list[RankedPassage]
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
    movie_context: str = Field(
        default="",
        description="Shared, source-attributed movie background; never audience-review evidence.",
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


class ClaimCitation(BaseModel):
    source_id: str = Field(min_length=1, description="Input source alias, e.g. e1.")
    quote: str = Field(min_length=1, max_length=800)

    @field_validator("source_id", "quote")
    @classmethod
    def clean_text(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("citation fields cannot be blank")
        return value.strip()


class AnswerClaim(BaseModel):
    """One reviewable statement, before it is placed in the public answer."""

    kind: Literal["finding", "insight", "comparison"]
    statement: str = Field(min_length=1, max_length=600)
    reasoning: str | None = Field(max_length=1000)
    citations: list[ClaimCitation] = Field(min_length=1, max_length=24)
    countercitations: list[ClaimCitation] = Field(max_length=24)
    limitation: str | None = Field(max_length=400)


class AggregationDraft(BaseModel):
    claims: list[AnswerClaim] = Field(max_length=12)


class ClaimReview(BaseModel):
    claim_id: str
    verdict: Literal["keep", "revise", "drop"]
    reason: str = Field(min_length=1, max_length=400)
    replacement: AnswerClaim | None


class ClaimAudit(BaseModel):
    invalid_source_ids: list[str] = Field(
        description="Source aliases with no review content relevant to the question."
    )
    reviews: list[ClaimReview] = Field(max_length=12)


class OpinionInsight(BaseModel):
    claim: str = Field(min_length=1, description="证据支持的分析判断，不是情绪分类。")
    reasoning: str = Field(
        min_length=1, description="解释评价标准、分歧或条件；推断须明确标注。"
    )
    evidence_ids: list[str] = Field(min_length=1)
    counterevidence_ids: list[str]
    limitation: str = Field(
        min_length=1, description="反例、替代解释或不能推出的结论。"
    )


class EvidenceCard(BaseModel):
    citation_id: str = Field(
        description="Answer-local short citation, mapped to the persistent evidence_id."
    )
    evidence_id: str
    source_id: str
    chunk_id: str
    platform: ReviewPlatform
    review_id: str
    quote: str
    full_text: str
    context_ranges: list[TextRange] = Field(default_factory=list)
    title: str | None = None
    rating: float | None = None
    helpful_votes: int | None = None


class OpinionCluster(BaseModel):
    label: str
    summary: str
    platforms: list[ReviewPlatform]
    evidence_count: int = Field(
        ge=0,
        description="Distinct cited reviews in this cluster, not population size.",
    )
    evidence_ids: list[str]
    counterevidence_ids: list[str] = Field(default_factory=list)


class MovieAnswer(BaseModel):
    conclusion: str
    conclusion_evidence_ids: list[str] = Field(default_factory=list)
    clusters: list[OpinionCluster]
    insights: list[OpinionInsight] = Field(default_factory=list, max_length=4)
    platform_comparison: str | None
    platform_comparison_evidence_ids: list[str] = Field(default_factory=list)
    evidence: list[EvidenceCard]
    coverage: CoverageReport
    limitations: list[str]
