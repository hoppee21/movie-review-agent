"""JSON-safe contracts for clarification and controlled state updates."""

from enum import StrEnum
from typing import Any, Literal

from pydantic import BaseModel, Field, model_validator

from app.agent.schemas.plan import Action


Invalidation = Literal["movie", "request", "plan", "execution"]


class PatchAuthority(StrEnum):
    SYSTEM = "system"
    USER_CONFIRMED = "user_confirmed"
    TOOL_VERIFIED = "tool_verified"


class StatePatch(BaseModel):
    """The only write format returned by Action handlers."""

    set_values: dict[str, Any] = Field(default_factory=dict)
    append_values: dict[str, list[Any]] = Field(default_factory=dict)
    invalidates: list[Invalidation] = Field(default_factory=list)
    authority: PatchAuthority


class ClarificationOption(BaseModel):
    option_id: str = Field(min_length=1)
    label: str = Field(min_length=1)
    details: dict[str, Any] = Field(default_factory=dict)
    internal_patch: StatePatch

    def public_payload(self) -> dict[str, Any]:
        """Return only fields that are safe and useful for a UI."""

        return {
            "option_id": self.option_id,
            "label": self.label,
            "details": self.details,
        }


class ClarificationIssue(BaseModel):
    issue_id: str = Field(min_length=1)
    kind: str = Field(min_length=1)
    question: str = Field(min_length=1)
    target_id: str | None = None
    options: list[ClarificationOption] = Field(default_factory=list)
    suggested_actions: list[Action] = Field(default_factory=lambda: [Action.ASK_USER])

    def public_payload(self) -> dict[str, Any]:
        return {
            "issue_id": self.issue_id,
            "kind": self.kind,
            "question": self.question,
            "options": [option.public_payload() for option in self.options],
        }


class ClarificationReply(BaseModel):
    issue_id: str = Field(min_length=1)
    option_id: str | None = None
    text: str | None = None
    cancelled: bool = False

    @model_validator(mode="after")
    def validate_content(self) -> "ClarificationReply":
        if self.option_id is not None:
            self.option_id = self.option_id.strip() or None
        if self.text is not None:
            self.text = self.text.strip() or None
        if not self.cancelled and self.option_id is None and self.text is None:
            raise ValueError("reply must contain option_id or text")
        return self


class ClarificationRecord(BaseModel):
    issue_id: str
    question: str
    option_id: str | None = None
    selected_label: str | None = None
    text: str | None = None
    cancelled: bool = False
