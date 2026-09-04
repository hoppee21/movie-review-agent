"""Base interface shared by Agent tools."""

from abc import ABC, abstractmethod
from typing import Any


class Tool(ABC):
    """Define the metadata and execution contract of an Agent tool.

    Subclasses provide the function-calling metadata as class attributes and
    implement :meth:`execute` with the tool's actual behavior.

    Attributes:
        name: Function name exposed to the model.
        description: Description shown to the model.
        parameters: JSON Schema describing the function arguments.
        strict: Whether OpenAI strict schema validation is enabled.
    """

    name: str
    description: str
    parameters: dict[str, Any]
    strict: bool = True

    @abstractmethod
    async def execute(self, params: dict[str, Any]) -> Any:
        """Execute the tool with model-generated parameters.

        Args:
            params: Arguments validated against :attr:`parameters`.

        Returns:
            The tool-specific result returned to the Agent runtime.
        """

        raise NotImplementedError

    def to_openai_schema(self) -> dict[str, Any]:
        """Return this tool's OpenAI Responses API function schema."""

        return {
            "type": "function",
            "name": self.name,
            "description": self.description,
            "parameters": self.parameters,
            "strict": self.strict,
        }
