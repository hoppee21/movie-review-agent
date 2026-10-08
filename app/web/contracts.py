"""Public UI contracts. Backend-specific state belongs in the adapter."""

from typing import Literal

from pydantic import BaseModel, Field, field_validator


PROTOCOL_VERSION = "1.0"
RunStatus = Literal["queued", "running", "waiting_for_input", "completed", "failed", "cancelled"]


class CreateSession(BaseModel):
    model: str | None = Field(default=None, max_length=200)


class CreateRun(BaseModel):
    query: str = Field(min_length=1, max_length=12000)
    request_id: str = Field(min_length=1, max_length=120)

    @field_validator("query")
    @classmethod
    def strip_query(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("请输入一个问题。")
        return value.strip()


class Reply(BaseModel):
    request_id: str = Field(min_length=1, max_length=120)
    issue_id: str = Field(min_length=1, max_length=200)
    option_id: str | None = Field(default=None, max_length=200)
    text: str | None = Field(default=None, max_length=4000)
    cancelled: bool = False


class Choice(BaseModel):
    id: str
    label: str
    description: str = ""


class Clarification(BaseModel):
    id: str
    question: str
    choices: list[Choice] = Field(default_factory=list)


class Source(BaseModel):
    id: str
    platform: str
    title: str
    quote: str
    full_text: str
    stance: str = "unclear"
    reason: str | None = None
    url: str | None = None
    rating: float | None = None


class Section(BaseModel):
    id: str
    title: str
    body: str
    citations: list[str] = Field(default_factory=list)
    counter_citations: list[str] = Field(default_factory=list)
    note: str | None = None


class Link(BaseModel):
    title: str
    url: str


class Report(BaseModel):
    title: str
    summary: str
    citations: list[str] = Field(default_factory=list)
    sections: list[Section] = Field(default_factory=list)
    sources: list[Source] = Field(default_factory=list)
    source_counts: dict[str, int] = Field(default_factory=dict)
    coverage_sufficient: bool | None = None
    gaps: list[str] = Field(default_factory=list)
    limitations: list[str] = Field(default_factory=list)
    links: list[Link] = Field(default_factory=list)


class Message(BaseModel):
    id: str
    role: Literal["user", "assistant"]
    text: str
    created_at: str
    report: Report | None = None
    clarification: Clarification | None = None


class Stage(BaseModel):
    id: str
    label: str
    status: Literal["running", "completed", "failed"]
    elapsed_seconds: float | None = None


class Run(BaseModel):
    id: str
    query: str
    status: RunStatus
    started_at: str
    finished_at: str | None = None
    stages: list[Stage] = Field(default_factory=list)
    clarification: Clarification | None = None
    error: str | None = None


class Session(BaseModel):
    protocol_version: str = PROTOCOL_VERSION
    id: str
    revision: int = 0
    title: str = "新的分析"
    model: str
    created_at: str
    updated_at: str
    messages: list[Message] = Field(default_factory=list)
    run: Run | None = None


class Outcome(BaseModel):
    """The only result the session service expects from an agent driver."""

    text: str = ""
    report: Report | None = None
    clarification: Clarification | None = None
    cancelled: bool = False
