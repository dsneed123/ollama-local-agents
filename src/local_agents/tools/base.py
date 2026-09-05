"""Tool abstraction.

A tool is a small, typed, async callable the model can invoke. Tools declare
their parameters (rendered as JSON schema for the model) and the permission
they need. :class:`ToolSet` groups the tools available to one agent and
enforces permissions before dispatching calls.
"""

from __future__ import annotations

import inspect
import json
import logging
import time
from abc import ABC
from collections.abc import Awaitable, Callable, Iterable, Iterator, Sequence
from dataclasses import dataclass, field
from typing import Any

from local_agents.errors import (
    LocalAgentsError,
    SecurityError,
    ToolNotFoundError,
    ToolValidationError,
)
from local_agents.models.base import ToolSpec
from local_agents.security.permissions import Permission, PermissionSet

log = logging.getLogger(__name__)

#: Called before a guarded tool runs. Return ``False`` to refuse the call.
ApprovalCallback = Callable[["BaseTool", dict[str, Any]], Awaitable[bool]]

#: Permissions that trigger the approval callback when one is configured.
GUARDED_PERMISSIONS: frozenset[Permission] = frozenset({Permission.WRITE, Permission.DELETE})

JSON_TYPES = {"string", "integer", "number", "boolean", "array", "object"}


@dataclass(frozen=True)
class ToolParameter:
    """One argument a tool accepts."""

    name: str
    type: str = "string"
    description: str = ""
    required: bool = True
    default: Any = None
    enum: Sequence[Any] | None = None
    items: str | None = None  # element type for arrays

    def __post_init__(self) -> None:
        if self.type not in JSON_TYPES:
            raise ValueError(f"Unsupported parameter type {self.type!r} for {self.name!r}")

    def schema(self) -> dict[str, Any]:
        schema: dict[str, Any] = {"type": self.type}
        if self.description:
            schema["description"] = self.description
        if self.enum is not None:
            schema["enum"] = list(self.enum)
        if self.type == "array":
            schema["items"] = {"type": self.items or "string"}
        return schema


@dataclass
class ToolResult:
    """What a tool returns. ``output`` is what the model sees."""

    output: str
    success: bool = True
    error: str | None = None
    data: Any = None
    blocked: bool = False
    duration: float = 0.0

    @classmethod
    def ok(cls, output: str, data: Any = None) -> ToolResult:
        return cls(output=output, success=True, data=data)

    @classmethod
    def fail(cls, error: str, *, blocked: bool = False, hint: str | None = None) -> ToolResult:
        text = f"Error: {error}"
        if hint:
            text += f"\nHint: {hint}"
        return cls(output=text, success=False, error=error, blocked=blocked)

    def for_model(self, max_chars: int = 12_000) -> str:
        text = self.output if self.output else ("(no output)" if self.success else "Error")
        if len(text) > max_chars:
            head = text[: max_chars - 200]
            text = f"{head}\n... [truncated {len(text) - len(head):,} characters]"
        return text


class BaseTool(ABC):
    """Subclass this to create a tool.

    Set ``name``, ``description``, ``parameters`` and ``permission`` as class
    attributes and implement ``async def execute(self, <parameters>)``. It may
    return a :class:`ToolResult`, a string, or any JSON-serialisable value, and
    should raise :class:`ToolError` for expected failures.

    Intermediate base classes (no ``name``) may omit ``execute``; concrete tools
    must define it.
    """

    name: str = ""
    description: str = ""
    parameters: Sequence[ToolParameter] = ()
    permission: Permission = Permission.READ

    #: Implemented by subclasses with their own typed signature.
    execute: Callable[..., Awaitable[Any]]

    def __init_subclass__(cls, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)
        if cls.name and not inspect.iscoroutinefunction(getattr(cls, "execute", None)):
            raise TypeError(f"Tool {cls.__name__} must define `async def execute(...)`")

    def spec(self) -> ToolSpec:
        properties = {p.name: p.schema() for p in self.parameters}
        required = [p.name for p in self.parameters if p.required]
        parameters: dict[str, Any] = {"type": "object", "properties": properties}
        if required:
            parameters["required"] = required
        return ToolSpec(name=self.name, description=self.description, parameters=parameters)

    def validate(self, arguments: dict[str, Any]) -> dict[str, Any]:
        """Check required parameters and coerce simple scalar types."""
        known = {p.name: p for p in self.parameters}
        unknown = set(arguments) - set(known)
        cleaned: dict[str, Any] = {}
        for name, param in known.items():
            if name in arguments and arguments[name] is not None:
                cleaned[name] = _coerce(param, arguments[name])
            elif param.required:
                raise ToolValidationError(
                    f"Missing required argument '{name}' for tool '{self.name}'",
                    hint=f"Expected: {', '.join(known)}",
                )
            elif param.default is not None:
                cleaned[name] = param.default
        if unknown:
            log.debug("Ignoring unknown arguments for %s: %s", self.name, sorted(unknown))
        return cleaned

    async def run(self, **arguments: Any) -> ToolResult:
        """Validate, execute, and wrap the outcome in a :class:`ToolResult`."""
        started = time.perf_counter()
        try:
            cleaned = self.validate(arguments)
            outcome = await self.execute(**cleaned)
        except SecurityError as exc:
            log.warning("Tool %s blocked: %s", self.name, exc)
            result = ToolResult.fail(str(exc), blocked=True, hint=exc.hint)
        except LocalAgentsError as exc:
            result = ToolResult.fail(str(exc), hint=exc.hint)
        except (FileNotFoundError, IsADirectoryError, NotADirectoryError, PermissionError) as exc:
            result = ToolResult.fail(str(exc))
        except (ValueError, TypeError, KeyError) as exc:
            result = ToolResult.fail(f"{type(exc).__name__}: {exc}")
        else:
            result = _to_result(outcome)
        result.duration = time.perf_counter() - started
        return result

    def __repr__(self) -> str:
        return f"<Tool {self.name} ({self.permission})>"


class ToolSet:
    """The tools available to one agent, gated by a :class:`PermissionSet`."""

    def __init__(
        self,
        tools: Iterable[BaseTool] = (),
        *,
        permissions: PermissionSet | None = None,
        approval: ApprovalCallback | None = None,
    ) -> None:
        self._tools: dict[str, BaseTool] = {}
        self.permissions = permissions or PermissionSet()
        self.approval = approval
        for tool in tools:
            self.add(tool)

    def add(self, tool: BaseTool) -> None:
        if not tool.name:
            raise ValueError(f"{type(tool).__name__} must define a non-empty `name`")
        if tool.name in self._tools:
            raise ValueError(f"Duplicate tool name: {tool.name}")
        self._tools[tool.name] = tool

    def remove(self, name: str) -> None:
        self._tools.pop(name, None)

    def get(self, name: str) -> BaseTool:
        try:
            return self._tools[name]
        except KeyError:
            raise ToolNotFoundError(name, self.names()) from None

    def names(self) -> list[str]:
        return list(self._tools)

    def specs(self) -> list[ToolSpec]:
        return [t.spec() for t in self.usable()]

    def usable(self) -> list[BaseTool]:
        """Tools whose permission is granted (others are hidden from the model)."""
        return [t for t in self._tools.values() if self.permissions.allows(t.permission)]

    async def call(self, name: str, arguments: dict[str, Any] | None = None) -> ToolResult:
        try:
            tool = self.get(name)
            self.permissions.check(tool.permission, subject=f"Tool '{name}'")
        except SecurityError as exc:
            return ToolResult.fail(str(exc), blocked=True, hint=exc.hint)
        except LocalAgentsError as exc:
            return ToolResult.fail(str(exc), hint=exc.hint)
        arguments = arguments or {}
        guarded = self.approval is not None and tool.permission in GUARDED_PERMISSIONS
        if guarded and self.approval is not None and not await self.approval(tool, arguments):
            log.info("Tool %s refused by approval callback", name)
            return ToolResult.fail(f"The user declined to run '{name}'", blocked=True)
        return await tool.run(**arguments)

    def __contains__(self, name: object) -> bool:
        return name in self._tools

    def __iter__(self) -> Iterator[BaseTool]:
        return iter(self._tools.values())

    def __len__(self) -> int:
        return len(self._tools)

    def __repr__(self) -> str:
        return f"<ToolSet {self.names()}>"


def _coerce(param: ToolParameter, value: Any) -> Any:
    try:
        if param.type == "integer" and not isinstance(value, bool):
            return int(value)
        if param.type == "number" and not isinstance(value, bool):
            return float(value)
        if param.type == "boolean":
            if isinstance(value, str):
                return value.strip().lower() in ("1", "true", "yes", "y", "on")
            return bool(value)
        if param.type == "string" and not isinstance(value, str):
            return str(value)
        if param.type == "array" and isinstance(value, str):
            return [v.strip() for v in value.split(",") if v.strip()]
    except (TypeError, ValueError) as exc:
        raise ToolValidationError(
            f"Argument '{param.name}' must be of type {param.type} (got {value!r})"
        ) from exc
    if param.enum is not None and value not in param.enum:
        raise ToolValidationError(
            f"Argument '{param.name}' must be one of {list(param.enum)} (got {value!r})"
        )
    return value


def _to_result(outcome: Any) -> ToolResult:
    if isinstance(outcome, ToolResult):
        return outcome
    if outcome is None:
        return ToolResult.ok("")
    if isinstance(outcome, str):
        return ToolResult.ok(outcome)
    try:
        return ToolResult.ok(json.dumps(outcome, indent=2, default=str), data=outcome)
    except (TypeError, ValueError):
        return ToolResult.ok(str(outcome), data=outcome)


@dataclass
class ToolCallRecord:
    """A tool invocation and its result, kept for tracing."""

    name: str
    arguments: dict[str, Any]
    result: ToolResult
    metadata: dict[str, Any] = field(default_factory=dict)
