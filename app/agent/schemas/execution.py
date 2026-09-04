"""Execution-mode, action-result, and terminal-result contracts."""

from enum import StrEnum
from typing import Any

from pydantic import BaseModel, Field

from app.agent.schemas.clarification import ClarificationIssue, StatePatch
from app.agent.schemas.plan import PlanStep, TaskType


class ExecutionMode(StrEnum):
    RESOLUTION = "resolution"
    EXECUTION = "execution"


class ActionStatus(StrEnum):
    COMPLETED = "completed"
    NEEDS_CLARIFICATION = "needs_clarification"
    NOT_IMPLEMENTED = "not_implemented"
    CANCELLED = "cancelled"


class RunStatus(StrEnum):
    COMPLETED = "completed"
    UNSUPPORTED = "unsupported"
    NOT_IMPLEMENTED = "not_implemented"
    CANCELLED = "cancelled"
    UNRESOLVED = "unresolved"


class ActionInvocation(BaseModel):
    mode: ExecutionMode
    step: PlanStep
    issue_id: str | None = None


class ActionResult(BaseModel):
    status: ActionStatus
    patch: StatePatch | None = None
    issue: ClarificationIssue | None = None
    output: dict[str, Any] = Field(default_factory=dict)


class RunResult(BaseModel):
    status: RunStatus
    task: TaskType | None = None
    data: dict[str, Any] = Field(default_factory=dict)
    reason: str | None = None
