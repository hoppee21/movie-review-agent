"""Movie clues produced by the LLM and verified execution targets."""

from enum import StrEnum

from pydantic import BaseModel, Field


class MovieCandidate(BaseModel):
    """One unverified English-title candidate for a single movie mention."""

    english_title: str = Field(
        min_length=1,
        description="可能的标准英文片名，尚未通过 Wikidata 验证。",
    )
    release_year: int | None = Field(
        description="可能的上映年份；不确定则为 null。",
    )


class MovieMention(BaseModel):
    """One movie target mentioned by the user, with zero or more candidates."""

    query: str = Field(
        min_length=1,
        description="用户用于指代这一部电影的原始名称、ID、URL或描述。",
    )
    candidates: list[MovieCandidate] = Field(
        max_length=3,
        description="这一目标电影的英文候选；ID/URL 或无可靠候选时为空。",
    )


class MovieAnalysis(BaseModel):
    """Movie-only analysis, deliberately separate from the requested task."""

    mentions: list[MovieMention] = Field(
        max_length=3,
        description="用户明确提到的电影目标；不同电影必须分成不同 mention。",
    )
    clarification_question: str | None = Field(
        description="电影线索缺失或单个目标有歧义时的问题；否则为 null。",
    )


class MovieTargetStatus(StrEnum):
    IDENTIFIED = "identified"
    RESOLVED = "resolved"


class MovieTarget(BaseModel):
    """A stable execution target that can later be enriched by Wikidata."""

    target_id: str = Field(min_length=1)
    query: str = Field(min_length=1)
    english_title: str | None = None
    release_year: int | None = None
    status: MovieTargetStatus = MovieTargetStatus.IDENTIFIED
    wikidata_id: str | None = None
    imdb_url: str | None = None
    douban_url: str | None = None
