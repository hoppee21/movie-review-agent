"""One executor shared by resolution and ordinary plan actions."""

from app.agent.action_handlers import build_default_registry
from app.agent.schemas import (
    ActionInvocation,
    ActionResult,
    ActionStatus,
    AgentState,
)
from app.tools.registry import ActionRegistry
from app.progress import progress


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
        registration = self.registry.get(invocation.step.action, invocation.mode)
        if registration is None:
            return ActionResult(
                status=ActionStatus.NOT_IMPLEMENTED,
                output={"action": invocation.step.action.value},
            )
        with progress(f"阶段 {invocation.step.action.value}"):
            return await registration.handler(invocation, state)


__all__ = ["Executor", "build_default_registry"]
