"""Select a bounded clarification action from issue metadata."""

from app.agent.schemas import (
    Action,
    ActionInvocation,
    ClarificationIssue,
    ExecutionMode,
    PlanStep,
)


def next_resolution_invocation(issue: ClarificationIssue) -> ActionInvocation:
    action = issue.suggested_actions[0] if issue.suggested_actions else Action.ASK_USER
    return ActionInvocation(
        mode=ExecutionMode.RESOLUTION,
        issue_id=issue.issue_id,
        step=PlanStep(
            step_id=f"resolve-{issue.issue_id}",
            action=action,
            target_id=issue.target_id,
        ),
    )
