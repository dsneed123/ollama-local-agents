"""Provider-agnostic model interface.

A :class:`BaseModel` is a *bound* model: it knows which concrete model it
talks to and how. Agents only depend on this interface, so swapping Ollama for
a remote provider means implementing one class.
"""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from collections.abc import AsyncIterator, Iterable, Sequence
from dataclasses import dataclass, field
from typing import Any, Literal

Role = Literal["system", "user", "assistant", "tool"]


@dataclass
class ToolCall:
    """A request from the model to invoke a tool."""

    name: str
    arguments: dict[str, Any] = field(default_factory=dict)
    id: str | None = None

    def to_dict(self) -> dict[str, Any]:
        data: dict[str, Any] = {"function": {"name": self.name, "arguments": self.arguments}}
        if self.id:
            data["id"] = self.id
        return data


@dataclass
class Message:
    """A single chat message. ``tool_name`` is set on ``role="tool"`` messages."""

    role: Role
    content: str = ""
    tool_calls: list[ToolCall] = field(default_factory=list)
    tool_name: str | None = None
    tool_call_id: str | None = None

    @classmethod
    def system(cls, content: str) -> Message:
        return cls("system", content)

    @classmethod
    def user(cls, content: str) -> Message:
        return cls("user", content)

    @classmethod
    def assistant(cls, content: str, tool_calls: Iterable[ToolCall] | None = None) -> Message:
        return cls("assistant", content, list(tool_calls or []))

    @classmethod
    def tool(cls, name: str, content: str, call_id: str | None = None) -> Message:
        return cls("tool", content, tool_name=name, tool_call_id=call_id)

    def to_dict(self) -> dict[str, Any]:
        data: dict[str, Any] = {"role": self.role, "content": self.content}
        if self.tool_calls:
            data["tool_calls"] = [tc.to_dict() for tc in self.tool_calls]
        if self.tool_name:
            data["tool_name"] = self.tool_name
        if self.tool_call_id:
            data["tool_call_id"] = self.tool_call_id
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Message:
        calls = []
        for raw in data.get("tool_calls") or []:
            fn = raw.get("function", raw)
            args = fn.get("arguments") or {}
            if isinstance(args, str):
                try:
                    args = json.loads(args)
                except json.JSONDecodeError:
                    args = {"input": args}
            calls.append(ToolCall(name=fn.get("name", ""), arguments=args, id=raw.get("id")))
        return cls(
            role=data.get("role", "user"),
            content=data.get("content") or "",
            tool_calls=calls,
            tool_name=data.get("tool_name"),
            tool_call_id=data.get("tool_call_id"),
        )


@dataclass
class ToolSpec:
    """JSON-schema description of a tool, as sent to the model."""

    name: str
    description: str
    parameters: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }


@dataclass
class GenerationOptions:
    """Knobs that most providers support. ``None`` means "provider default"."""

    temperature: float | None = None
    context_length: int | None = None
    max_tokens: int | None = None
    timeout: float | None = None
    stop: list[str] | None = None
    seed: int | None = None
    think: bool | None = None

    def merged(self, other: GenerationOptions | None) -> GenerationOptions:
        """Return a copy where ``other``'s non-``None`` fields override ours."""
        if other is None:
            return GenerationOptions(**self.__dict__)
        merged = dict(self.__dict__)
        merged.update({k: v for k, v in other.__dict__.items() if v is not None})
        return GenerationOptions(**merged)


@dataclass
class Usage:
    prompt_tokens: int = 0
    completion_tokens: int = 0

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens


@dataclass
class GenerationResult:
    """The complete output of one model call."""

    content: str
    model: str
    tool_calls: list[ToolCall] = field(default_factory=list)
    thinking: str | None = None
    done_reason: str | None = None
    usage: Usage = field(default_factory=Usage)
    raw: dict[str, Any] | None = None

    @property
    def message(self) -> Message:
        return Message.assistant(self.content, self.tool_calls)


@dataclass
class StreamChunk:
    """One increment of a streamed response."""

    content: str = ""
    thinking: str = ""
    tool_calls: list[ToolCall] = field(default_factory=list)
    done: bool = False
    done_reason: str | None = None
    usage: Usage | None = None


@dataclass(frozen=True)
class ModelInfo:
    """What we know about an installed model."""

    name: str
    family: str = ""
    parameter_size: str = ""
    quantization: str = ""
    context_length: int | None = None
    capabilities: tuple[str, ...] = ()
    size_bytes: int = 0
    modified_at: str = ""

    @property
    def parameters_billions(self) -> float:
        """Best-effort numeric parameter count in billions (0 if unknown)."""
        text = self.parameter_size.strip().upper()
        if not text:
            return 0.0
        try:
            if text.endswith("B"):
                return float(text[:-1])
            if text.endswith("M"):
                return float(text[:-1]) / 1000.0
            return float(text)
        except ValueError:
            return 0.0

    @property
    def supports_tools(self) -> bool:
        return "tools" in self.capabilities

    @property
    def supports_thinking(self) -> bool:
        return "thinking" in self.capabilities

    @property
    def is_embedding_model(self) -> bool:
        return "embedding" in self.capabilities and "completion" not in self.capabilities

    @property
    def is_vision_model(self) -> bool:
        return "vision" in self.capabilities

    @property
    def base_name(self) -> str:
        return self.name.split(":", 1)[0]

    @property
    def size_gb(self) -> float:
        return self.size_bytes / 1_000_000_000

    def describe(self) -> str:
        parts = [self.name]
        if self.parameter_size:
            parts.append(self.parameter_size)
        if self.quantization:
            parts.append(self.quantization)
        if self.context_length:
            parts.append(f"ctx {self.context_length:,}")
        if self.capabilities:
            parts.append(",".join(self.capabilities))
        return "  ".join(parts)


class BaseModel(ABC):
    """Abstract chat model. Implementations must be safe to share across agents."""

    #: Provider identifier, e.g. ``"ollama"``.
    provider: str = "base"

    @property
    @abstractmethod
    def name(self) -> str:
        """Concrete model identifier (may be resolved lazily)."""

    @abstractmethod
    async def generate(
        self,
        messages: Sequence[Message],
        *,
        tools: Sequence[ToolSpec] | None = None,
        options: GenerationOptions | None = None,
    ) -> GenerationResult:
        """Run one non-streaming completion."""

    @abstractmethod
    def stream(
        self,
        messages: Sequence[Message],
        *,
        tools: Sequence[ToolSpec] | None = None,
        options: GenerationOptions | None = None,
    ) -> AsyncIterator[StreamChunk]:
        """Run one streaming completion, yielding :class:`StreamChunk`."""

    @abstractmethod
    async def health_check(self) -> bool:
        """Return ``True`` when the provider is reachable."""

    @abstractmethod
    async def list_models(self, *, refresh: bool = False) -> list[ModelInfo]:
        """List models available from this model's provider."""

    async def info(self) -> ModelInfo | None:
        """Return :class:`ModelInfo` for this model when available."""
        for model in await self.list_models():
            if model.name == self.name:
                return model
        return None

    async def supports_tools(self) -> bool:
        """Whether the provider advertises native tool calling for this model."""
        info = await self.info()
        return info.supports_tools if info else False

    async def close(self) -> None:  # noqa: B027 - optional hook, intentionally empty
        """Release underlying resources. Default: nothing to do."""

    def __repr__(self) -> str:
        return f"<{type(self).__name__} {self.provider}:{self.name}>"
