"""Test helpers that avoid a running Ollama.

:class:`ScriptedModel` plays back a fixed sequence of responses so agent
behaviour (tool loops, memory, streaming) can be tested deterministically.
It is part of the public API so custom agents can be unit-tested the same way.
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Iterable, Sequence
from dataclasses import dataclass, field
from typing import Any

from local_agents.models.base import (
    BaseModel,
    GenerationOptions,
    GenerationResult,
    Message,
    ModelInfo,
    StreamChunk,
    ToolCall,
    ToolSpec,
    Usage,
)


@dataclass
class ScriptedTurn:
    """One canned model reply."""

    content: str = ""
    tool_calls: list[ToolCall] = field(default_factory=list)
    thinking: str = ""

    @classmethod
    def text(cls, content: str) -> ScriptedTurn:
        return cls(content=content)

    @classmethod
    def call(cls, name: str, content: str = "", **arguments: Any) -> ScriptedTurn:
        return cls(content=content, tool_calls=[ToolCall(name=name, arguments=arguments)])


class ScriptedModel(BaseModel):
    """A :class:`BaseModel` that returns pre-written turns in order.

    Every call (``generate`` or ``stream``) consumes the next turn. The
    messages each call received are recorded in :attr:`calls` for assertions.
    When the script runs out, the last turn is repeated.
    """

    provider = "scripted"

    def __init__(
        self,
        turns: Iterable[ScriptedTurn | str] = (),
        *,
        name: str = "scripted-model",
        supports_tools: bool = True,
        chunk_size: int = 4,
        healthy: bool = True,
    ) -> None:
        self._turns: list[ScriptedTurn] = [
            ScriptedTurn.text(t) if isinstance(t, str) else t for t in turns
        ]
        self._name = name
        self._supports_tools = supports_tools
        self._chunk_size = max(1, chunk_size)
        self._healthy = healthy
        self.calls: list[dict[str, Any]] = []
        self._index = 0

    @property
    def name(self) -> str:
        return self._name

    def add(self, turn: ScriptedTurn | str) -> None:
        self._turns.append(ScriptedTurn.text(turn) if isinstance(turn, str) else turn)

    def _next(self, messages: Sequence[Message], tools: Sequence[ToolSpec] | None) -> ScriptedTurn:
        self.calls.append({"messages": list(messages), "tools": list(tools or [])})
        if not self._turns:
            return ScriptedTurn.text("")
        turn = self._turns[min(self._index, len(self._turns) - 1)]
        self._index += 1
        return turn

    async def generate(
        self,
        messages: Sequence[Message],
        *,
        tools: Sequence[ToolSpec] | None = None,
        options: GenerationOptions | None = None,
    ) -> GenerationResult:
        turn = self._next(messages, tools)
        return GenerationResult(
            content=turn.content,
            model=self._name,
            tool_calls=list(turn.tool_calls),
            thinking=turn.thinking or None,
            done_reason="stop",
            usage=Usage(prompt_tokens=len(messages), completion_tokens=len(turn.content.split())),
        )

    async def stream(
        self,
        messages: Sequence[Message],
        *,
        tools: Sequence[ToolSpec] | None = None,
        options: GenerationOptions | None = None,
    ) -> AsyncIterator[StreamChunk]:
        turn = self._next(messages, tools)
        if turn.thinking:
            yield StreamChunk(thinking=turn.thinking)
        text = turn.content
        for start in range(0, len(text), self._chunk_size):
            yield StreamChunk(content=text[start : start + self._chunk_size])
        if turn.tool_calls:
            yield StreamChunk(tool_calls=list(turn.tool_calls))
        yield StreamChunk(
            done=True,
            done_reason="stop",
            usage=Usage(prompt_tokens=len(messages), completion_tokens=len(text.split())),
        )

    async def health_check(self) -> bool:
        return self._healthy

    async def list_models(self, *, refresh: bool = False) -> list[ModelInfo]:
        return [self.model_info()]

    async def supports_tools(self) -> bool:
        return self._supports_tools

    def model_info(self) -> ModelInfo:
        return ModelInfo(
            name=self._name,
            family="scripted",
            parameter_size="1B",
            capabilities=("completion", "tools") if self._supports_tools else ("completion",),
        )

    @property
    def remaining(self) -> int:
        return max(0, len(self._turns) - self._index)
