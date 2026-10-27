"""One executor shared by resolution and ordinary plan actions."""

from app.agent.action_handlers import build_default_registry
from app.agent.schemas import (
    ActionInvocation,
    ActionResult,
    ActionStatus,
    AgentState,
)
from app.progress import progress
from app.run_log import record_event
from app.tools.registry import ActionRegistry


class Executor:
    """Invoke one registered Action and normalize missing capabilities."""

    def __init__(self, registry: ActionRegistry) -> None:
        self.registry = registry

    async def execute(
        self,
        invocation: ActionInvocation,
        state: AgentState,
    ) -> ActionResult:
        invocation = ActionInvocation.model_validate(invocation)
        record_event("action.started", invocation=invocation)
        registration = self.registry.get(invocation.step.action, invocation.mode)
        if registration is None:
            result = ActionResult(
                status=ActionStatus.NOT_IMPLEMENTED,
                output={"action": invocation.step.action.value},
            )
        else:
            with progress(f"阶段 {invocation.step.action.value}"):
                result = await registration.handler(invocation, state)
        record_event(
            "action.completed",
            action=invocation.step.action,
            status=result.status,
            output=result.output,
            issue=result.issue,
        )
        return result


__all__ = ["Executor", "build_default_registry"]
