"""Single registry shared by resolution and ordinary plan execution."""

from collections.abc import Awaitable, Callable, Iterable
from dataclasses import dataclass

from app.agent.schemas import (
    Action,
    ActionInvocation,
    ActionResult,
    AgentState,
    ExecutionMode,
)


ActionHandler = Callable[[ActionInvocation, AgentState], Awaitable[ActionResult]]


@dataclass(frozen=True)
class ActionRegistration:
    action: Action
    modes: frozenset[ExecutionMode]
    handler: ActionHandler


class ActionRegistry:
    """Own action handlers and enforce their allowed execution modes."""

    def __init__(self) -> None:
        self._items: dict[Action, ActionRegistration] = {}

    def register(
        self,
        action: Action,
        *,
        modes: Iterable[ExecutionMode],
        handler: ActionHandler,
    ) -> None:
        if action in self._items:
            raise ValueError(f"Action is already registered: {action}")
        allowed = frozenset(modes)
        if not allowed:
            raise ValueError(f"Action must allow at least one mode: {action}")
        self._items[action] = ActionRegistration(action, allowed, handler)

    def get(
        self,
        action: Action,
        mode: ExecutionMode,
    ) -> ActionRegistration | None:
        registration = self._items.get(action)
        if registration is not None and mode not in registration.modes:
            raise PermissionError(
                f"Action {action} is not allowed in {mode} mode"
            )
        return registration

    def __contains__(self, action: Action) -> bool:
        return action in self._items
