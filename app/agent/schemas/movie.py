"""Movie clues produced by the LLM and verified execution targets."""

from enum import StrEnum
from typing import Literal

from pydantic import BaseModel, Field


class MovieConstraint(BaseModel):
    """A number explicitly supplied by the user, with its literal source."""

    value: int = Field(ge=1, le=2100)
    quote: str = Field(
        min_length=1,
        description="用户原文或确认答复中的连续原话，必须包含这个数字的表述。",
    )


class MovieMention(BaseModel):
    """A target or reference and user constraints; hints never establish identity."""

    query: str = Field(
        min_length=1,
        description="用户用于指代这一部电影的原始名称、ID、URL或描述。",
    )
    role: Literal["target", "reference"] = Field(
        default="target",
        description="用户实际要处理的作品是 target，仅用于描述或参照的作品是 reference。",
    )
    scope: Literal["movie", "series"] = Field(
        default="movie",
        description="只有用户明确要求整个系列时才为 series；模糊系列名称仍按单部待查证。",
    )
    release_year: MovieConstraint | None = Field(
        default=None,
        description="仅用户明确给出或确认的正向年份限定；排除的年份和模型推测不能填写。",
    )
    installment: MovieConstraint | None = Field(
        default=None,
        description="仅用户明确给出或确认的正向部数限定；未指定不能默认第一部。",
    )
    search_hints: list[str] = Field(
        default_factory=list,
        max_length=3,
        description="可选的片名或别名检索提示，未经核实，不包含猜测的 ID 或年份过滤。",
    )


class MovieAnalysis(BaseModel):
    """Movie-only analysis, deliberately separate from the requested task."""

    kind: Literal["non_movie", "general_movie", "specific_movie"] = Field(
        description=(
            "general_movie 仅指无具体目标的泛请求；指定片名、系列或待定续作仍为 "
            "specific_movie；非电影为 non_movie。"
        ),
    )
    mentions: list[MovieMention] = Field(
        description="保留所有实际目标和参照作品；具体目标没有有效线索时为空。",
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
    installment: int | None = None
    search_hints: list[str] = Field(default_factory=list)
    status: MovieTargetStatus = MovieTargetStatus.IDENTIFIED
    wikidata_id: str | None = None
    imdb_url: str | None = None
    douban_url: str | None = None
