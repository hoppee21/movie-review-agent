"""Pure controller helpers: state invalidation, patching, and next-node choice."""

from typing import Any, Literal

from app.agent.schemas import (
    ActionInvocation,
    ActionResult,
    ActionStatus,
    AgentState,
    ClarificationIssue,
    ClarificationRecord,
    ExecutionPlan,
    ExecutionMode,
    MovieAnalysis,
    MovieTarget,
    RequestAnalysis,
    RunResult,
    RunStatus,
    StatePatch,
    TaskType,
)


ControllerNode = Literal[
    "analyze_movie",
    "analyze_request",
    "resolve_next",
    "build_plan",
    "execute_action",
    "finalize",
]

STATE_FIELDS = set(AgentState.__annotations__)


def apply_state_patch(state: AgentState, patch: StatePatch) -> dict[str, Any]:
    """Apply invalidation cascades first, then validated writes and appends."""

    unknown = (set(patch.set_values) | set(patch.append_values)) - STATE_FIELDS
    if unknown:
        raise ValueError(f"StatePatch contains unknown fields: {sorted(unknown)}")
    if "user_query" in patch.set_values or "user_query" in patch.append_values:
        raise ValueError("Actions cannot overwrite user_query")

    updates: dict[str, Any] = {}
    invalidations = set(patch.invalidates)

    if "movie" in invalidations:
        updates.update(
            movie_analysis=None,
            movie_targets=[],
            request_analysis=None,
            execution_plan=None,
            plan_cursor=0,
            active_action=None,
            action_result=None,
            artifacts={},
            final_result=None,
        )
    if "request" in invalidations:
        updates.update(
            request_analysis=None,
            execution_plan=None,
            plan_cursor=0,
            active_action=None,
            action_result=None,
            final_result=None,
        )
    if "plan" in invalidations:
        updates.update(
            execution_plan=None,
            plan_cursor=0,
            active_action=None,
            action_result=None,
            final_result=None,
        )
    if "execution" in invalidations:
        updates.update(
            execution_plan=None,
            plan_cursor=0,
            active_action=None,
            action_result=None,
            artifacts={},
            final_result=None,
        )

    updates.update(
        {
            field: _normalize_field(field, value)
            for field, value in patch.set_values.items()
        }
    )
    for field, values in patch.append_values.items():
        current = updates.get(field, state.get(field, []))
        if not isinstance(current, list):
            raise ValueError(f"StatePatch append target is not a list: {field}")
        updates[field] = [
            *_normalize_list_field(field, current),
            *_normalize_list_field(field, values),
        ]
    return updates


def _normalize_field(field: str, value: Any) -> Any:
    """Revalidate values nested inside StatePatch dictionaries after restore."""

    model_fields = {
        "movie_analysis": MovieAnalysis,
        "request_analysis": RequestAnalysis,
        "execution_plan": ExecutionPlan,
        "active_action": ActionInvocation,
        "action_result": ActionResult,
        "final_result": RunResult,
    }
    if value is None:
        return None
    if field in model_fields:
        return model_fields[field].model_validate(value)
    if field in {"movie_targets", "pending_issues", "clarification_history"}:
        if not isinstance(value, list):
            raise ValueError(f"StatePatch field must be a list: {field}")
        return _normalize_list_field(field, value)
    return value


def _normalize_list_field(field: str, values: list[Any]) -> list[Any]:
    models = {
        "movie_targets": MovieTarget,
        "pending_issues": ClarificationIssue,
        "clarification_history": ClarificationRecord,
    }
    model = models.get(field)
    return [model.model_validate(value) for value in values] if model else list(values)


def patch_invalidates_plan(patch: StatePatch | None) -> bool:
    return bool(patch and set(patch.invalidates) & {"movie", "request", "plan", "execution"})


def next_node(state: AgentState) -> ControllerNode:
    """Derive control flow from state readiness; no return_to field is needed."""

    if state.get("final_result") is not None:
        return "finalize"
    if state.get("pending_issues"):
        return "resolve_next"
    if not state.get("movie_targets"):
        return "analyze_movie"
    if state.get("request_analysis") is None:
        return "analyze_request"
    if state.get("execution_plan") is None:
        return "build_plan"
    plan = ExecutionPlan.model_validate(state["execution_plan"])
    if state.get("plan_cursor", 0) < len(plan.steps):
        return "execute_action"
    return "finalize"


def apply_action_result(state: AgentState) -> AgentState:
    """Reduce one ActionResult into State and advance, pause, or terminate."""

    invocation = state.get("active_action")
    result = state.get("action_result")
    if invocation is None or result is None:
        raise RuntimeError("apply_result requires an invocation and result")
    invocation = ActionInvocation.model_validate(invocation)
    result = ActionResult.model_validate(result)

    updates: dict[str, Any] = {}
    if result.patch is not None:
        updates.update(apply_state_patch(state, result.patch))
    updates.update(active_action=None, action_result=None)

    if result.status == ActionStatus.COMPLETED:
        if invocation.mode == ExecutionMode.RESOLUTION:
            pending = updates.get(
                "pending_issues",
                state.get("pending_issues", []),
            )
            updates["pending_issues"] = [
                issue
                for issue in pending
                if ClarificationIssue.model_validate(issue).issue_id
                != invocation.issue_id
            ]
        elif not patch_invalidates_plan(result.patch):
            updates["plan_cursor"] = state.get("plan_cursor", 0) + 1
        return updates

    if result.status == ActionStatus.NEEDS_CLARIFICATION:
        if result.issue is None:
            raise RuntimeError("needs_clarification result requires an issue")
        pending = [
            ClarificationIssue.model_validate(issue)
            for issue in state.get("pending_issues", [])
        ]
        if invocation.mode == ExecutionMode.RESOLUTION:
            pending = [
                issue
                for issue in pending
                if issue.issue_id != invocation.issue_id
            ]
        pending.append(result.issue)
        updates["pending_issues"] = pending
        return updates

    plan = state.get("execution_plan")
    task = ExecutionPlan.model_validate(plan).intent.task if plan else None
    if result.status == ActionStatus.NOT_IMPLEMENTED:
        updates["final_result"] = RunResult(
            status=RunStatus.NOT_IMPLEMENTED,
            task=task,
            reason=f"Action is not implemented: {result.output.get('action')}",
        )
        return updates

    reason = result.output.get("reason")
    updates["final_result"] = RunResult(
        status=(
            RunStatus.UNRESOLVED
            if reason == "clarification_budget_exhausted"
            else RunStatus.CANCELLED
        ),
        task=task,
        reason=str(reason or "clarification_cancelled"),
    )
    return updates


def finalize_state(state: AgentState) -> AgentState:
    """Create a truthful terminal result without placeholder RAG answers."""

    if state.get("final_result") is not None:
        return {}
    raw_plan = state.get("execution_plan")
    if raw_plan is None:
        raise RuntimeError("finalize requires a plan or an existing result")
    plan = ExecutionPlan.model_validate(raw_plan)
    if plan.intent.task == TaskType.UNSUPPORTED:
        return {
            "final_result": RunResult(
                status=RunStatus.UNSUPPORTED,
                task=plan.intent.task,
                reason=plan.intent.unsupported_reason or "Unsupported request",
            )
        }
    return {
        "final_result": RunResult(
            status=RunStatus.COMPLETED,
            task=plan.intent.task,
            data={
                "movie_targets": [
                    MovieTarget.model_validate(target).model_dump(mode="json")
                    for target in state.get("movie_targets", [])
                ],
                "artifacts": state.get("artifacts", {}),
            },
        )
    }
