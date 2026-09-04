"""Planner intent and deterministic execution-plan contracts."""

from enum import StrEnum
from typing import Literal

from pydantic import BaseModel, Field, field_validator


Platform = Literal["imdb", "douban"]


class TaskType(StrEnum):
    RESOLVE_URLS = "resolve_urls"
    COLLECT_REVIEWS = "collect_reviews"
    OPINION_QA = "opinion_qa"
    PLATFORM_COMPARISON = "platform_comparison"
    UNSUPPORTED = "unsupported"


class Action(StrEnum):
    ASK_USER = "ask_user"
    RESOLVE_MOVIE = "resolve_movie"
    COLLECT_REVIEWS = "collect_movie_reviews"
    BUILD_INDEX = "build_opinion_index"
    QUERY_OPINIONS = "query_movie_opinions"
    AGGREGATE_OPINIONS = "aggregate_movie_opinions"


class PlanIntent(BaseModel):
    """What the user wants; it never contains model-selected tool calls."""

    task: TaskType
    platforms: list[Platform]
    aspect: str | None
    unsupported_reason: str | None

    @field_validator("platforms")
    @classmethod
    def unique_platforms(cls, value: list[Platform]) -> list[Platform]:
        return list(dict.fromkeys(value))


class PlanStep(BaseModel):
    """One whitelisted action bound to a stable movie target."""

    step_id: str = Field(min_length=1)
    action: Action
    target_id: str | None = None


class ExecutionPlan(BaseModel):
    intent: PlanIntent
    steps: list[PlanStep]
