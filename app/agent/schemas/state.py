"""Persisted LangGraph state for analysis, planning, and execution."""

from typing import Any, TypedDict

from app.agent.schemas.clarification import ClarificationIssue, ClarificationRecord
from app.agent.schemas.execution import ActionInvocation, ActionResult, RunResult
from app.agent.schemas.movie import MovieAnalysis, MovieTarget
from app.agent.schemas.plan import ExecutionPlan
from app.agent.schemas.request import RequestAnalysis


class AgentState(TypedDict, total=False):
    user_query: str
    movie_analysis: MovieAnalysis | None
    movie_targets: list[MovieTarget]
    request_analysis: RequestAnalysis | None
    clarification_history: list[ClarificationRecord]
    clarification_rounds: int
    pending_issues: list[ClarificationIssue]
    execution_plan: ExecutionPlan | None
    plan_cursor: int
    active_action: ActionInvocation | None
    action_result: ActionResult | None
    artifacts: dict[str, Any]
    final_result: RunResult | None
