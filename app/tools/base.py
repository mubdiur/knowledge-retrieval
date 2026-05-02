"""Base tool abstraction and registry."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any


@dataclass
class ToolSpec:
    """Declarative metadata for a tool."""
    name: str
    description: str
    parameters: dict[str, dict[str, Any]] = field(default_factory=dict)
    required_params: list[str] = field(default_factory=list)


class BaseTool(ABC):
    """Base class for all agent tools."""

    @property
    @abstractmethod
    def spec(self) -> ToolSpec:
        """Return the tool's specification (name, description, params)."""
        ...

    @abstractmethod
    async def run(self, **kwargs) -> dict[str, Any]:
        """Execute the tool with given parameters. Return dict with 'success' and 'data'."""
        ...


class ToolRegistry:
    """Global tool registry — tools register themselves here."""

    _tools: dict[str, BaseTool] = {}

    @classmethod
    def register(cls, tool: BaseTool) -> None:
        cls._tools[tool.spec.name] = tool

    @classmethod
    def get(cls, name: str) -> BaseTool | None:
        return cls._tools.get(name)

    @classmethod
    def list_specs(cls) -> list[ToolSpec]:
        return [t.spec for t in cls._tools.values()]

    @classmethod
    def describe(cls) -> str:
        """Return a formatted description for the agent's system prompt."""
        lines = ["Available Tools:", "─" * 40]
        for spec in cls.list_specs():
            params_desc = ", ".join(
                f"{k}: {v.get('type', 'any')}{' (required)' if k in spec.required_params else ''}"
                for k, v in spec.parameters.items()
            )
            lines.append(f"\n  • {spec.name}: {spec.description}")
            if params_desc:
                lines.append(f"    Args: {params_desc}")
        return "\n".join(lines)
