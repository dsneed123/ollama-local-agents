"""The universal agent interface and the shared tool-calling loop.

Every agent — built-in or user-created — subclasses :class:`BaseAgent` and
inherits :meth:`run` / :meth:`stream`, which implement a model ⇄ tool loop:

1. Build the message list from the system prompt, memory, and the task.
2. Ask the model. If it returns tool calls, execute them and feed the results
   back; otherwise the reply is the final answer.
3. Stop at ``config.max_iterations``.

Models with native tool calling get JSON-schema tool specs; other models fall
back to a fenced ``tool_call`` block protocol described in the system prompt,
so every installed model can drive tools.
"""

from __future__ import annotations

import json
import logging
import re
from collections.abc import AsyncIterator, Sequence
from dataclasses import dataclass, field
from typing import Any, ClassVar, Literal, Protocol, runtime_checkable

from local_agents.memory.base import BaseMemory
from local_agents.memory.conversation import ConversationMemory
from local_agents.models.base import (
    BaseModel,
    GenerationOptions,
    Message,
    ToolCall,
    Usage,
)
from local_agents.models.selection import ModelRole
from local_agents.security.permissions import PermissionSet
from local_agents.security.workspace import Workspace
from local_agents.tools.base import BaseTool, ToolCallRecord, ToolResult, ToolSet

log = logging.getLogger(__name__)

EventType = Literal[
    "start", "thinking", "token", "message", "tool_call", "tool_result", "nudge", "final", "error"
]

TOOL_FIRST_NUDGE = (
    "You answered without using any tools. Do not ask for clarification yet: use your tools "
    "first (for example search the exact term or read the relevant files), then answer based "
    "on what you find."
)


@dataclass
class AgentConfig:
    """Runtime knobs shared by all agents."""

    max_iterations: int = 15
    temperature: float | None = None
    context_length: int | None = None
    system_prompt: str | None = None
    max_tool_output_chars: int = 12_000
    remember_tool_calls: bool = False
    think: bool | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    def generation_options(self) -> GenerationOptions:
        return GenerationOptions(
            temperature=self.temperature, context_length=self.context_length, think=self.think
        )


@dataclass
class AgentEvent:
    """Streamed progress from :meth:`BaseAgent.stream`."""

    type: EventType
    content: str = ""
    tool_call: ToolCall | None = None
    tool_result: ToolResult | None = None
    iteration: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class AgentResult:
    """Everything a run produced."""

    output: str
    agent: str
    model: str
    iterations: int = 0
    tool_calls: list[ToolCallRecord] = field(default_factory=list)
    usage: Usage = field(default_factory=Usage)
    messages: list[Message] = field(default_factory=list)
    stop_reason: str = "final"
    thinking: str = ""

    def __str__(self) -> str:
        return self.output


@dataclass
class AgentCapabilities:
    name: str
    description: str
    model: str
    model_role: ModelRole
    tools: list[str]
    permissions: list[str]
    can_delegate: bool
    max_iterations: int


@runtime_checkable
class AgentExecutor(Protocol):
    """Runs other agents by name. Implemented by :class:`AgentRuntime`."""

    async def run(self, agent: str, task: str, context: Any = None) -> AgentResult: ...

    def available_agents(self) -> list[tuple[str, str]]: ...


class BaseAgent:
    """Base class for all agents.

    Subclasses normally only set class attributes:

    ``name``, ``description``
        Identity (also set by :func:`register_agent`).
    ``system_prompt``
        Instructions for the model.
    ``model_role``
        Which kind of model to auto-select: ``general``, ``coding``,
        ``reasoning`` or ``fast``.
    ``default_tool_groups``
        Tool groups the runtime injects when the caller passes none.
    ``can_delegate``
        Whether the runtime should hand the agent a ``delegate`` tool.
    ``tool_first``
        Nudge the model once if it answers without using any tool.

    Override :meth:`custom_tools` to ship tools with the agent, and
    :meth:`build_system_prompt` to add dynamic context.

    Everything is injected through ``__init__``; there are no globals.
    """

    name: ClassVar[str] = ""
    description: ClassVar[str] = ""
    #: Menu grouping; free-form, see ``local_agents.agents.registry.CATEGORY_ORDER``.
    category: ClassVar[str] = "General"
    #: Human-friendly label for menus; defaults to ``"<Name> Agent"``.
    display_name: ClassVar[str] = ""
    system_prompt: ClassVar[str] = "You are a helpful assistant."
    model_role: ClassVar[ModelRole] = "general"
    requires_tool_calling: ClassVar[bool] = True
    default_tool_groups: ClassVar[tuple[str, ...]] = ()
    default_permissions: ClassVar[PermissionSet | None] = None
    can_delegate: ClassVar[bool] = False
    #: When True, an answer given before any tool has been used is sent back
    #: once with a reminder to use tools first (helps smaller models that
    #: skip research and ask for clarification instead).
    tool_first: ClassVar[bool] = False

    def __init__(
        self,
        model: BaseModel,
        tools: ToolSet | Sequence[BaseTool] | None = None,
        memory: BaseMemory | None = None,
        config: AgentConfig | None = None,
        executor: AgentExecutor | None = None,
        workspace: Workspace | None = None,
    ) -> None:
        self.model = model
        if isinstance(tools, ToolSet):
            self.tools = tools
        else:
            self.tools = ToolSet(tools or (), permissions=self.default_permissions)
        self.memory: BaseMemory = memory if memory is not None else ConversationMemory()
        self.config = config or AgentConfig()
        self.executor = executor
        self.workspace = workspace
        self._native_tools: bool | None = None

    @classmethod
    def custom_tools(cls, workspace: Workspace | None = None) -> list[BaseTool]:
        """Tools this agent ships with, added on top of ``default_tool_groups``."""
        return []

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #

    async def run(self, task: str, context: Any = None) -> AgentResult:
        """Run to completion and return the final :class:`AgentResult`."""
        result: AgentResult | None = None
        async for event in self.stream(task, context):
            if event.type == "final":
                result = event.metadata["result"]
        assert result is not None  # stream always yields a final event
        return result

    async def stream(self, task: str, context: Any = None) -> AsyncIterator[AgentEvent]:
        """Run the agent loop, yielding :class:`AgentEvent` as it progresses."""
        messages = await self._build_messages(task, context)
        transcript: list[Message] = []
        records: list[ToolCallRecord] = []
        usage = Usage()
        thinking_total: list[str] = []
        options = self.config.generation_options()
        native = await self._use_native_tools()
        specs = self.tools.specs() if (native and len(self.tools)) else None

        yield AgentEvent("start", metadata={"agent": self.name, "model": self.model.name})

        final_text = ""
        stop_reason = "max_iterations"
        iteration = 0
        nudged = False
        for iteration in range(1, self.config.max_iterations + 1):
            content_parts: list[str] = []
            thinking_parts: list[str] = []
            tool_calls: list[ToolCall] = []
            gate = _TokenGate() if len(self.tools) else None
            async for chunk in self.model.stream(messages, tools=specs, options=options):
                if chunk.thinking:
                    thinking_parts.append(chunk.thinking)
                    yield AgentEvent("thinking", chunk.thinking, iteration=iteration)
                if chunk.content:
                    content_parts.append(chunk.content)
                    visible = gate.feed(chunk.content) if gate else chunk.content
                    if visible:
                        yield AgentEvent("token", visible, iteration=iteration)
                if chunk.tool_calls:
                    tool_calls.extend(chunk.tool_calls)
                if chunk.done and chunk.usage:
                    usage.prompt_tokens += chunk.usage.prompt_tokens
                    usage.completion_tokens += chunk.usage.completion_tokens
            if gate:
                visible = gate.flush()
                if visible:
                    yield AgentEvent("token", visible, iteration=iteration)

            content = "".join(content_parts)
            if thinking_parts:
                thinking_total.append("".join(thinking_parts))
            if len(self.tools) and not tool_calls:
                # Some models emit tool calls as text even when native calling is
                # available, so the text protocol is always accepted as a fallback.
                parsed, content = parse_text_tool_calls(content)
                tool_calls.extend(parsed)

            assistant = Message.assistant(content, tool_calls)
            messages.append(assistant)
            transcript.append(assistant)
            if content.strip():
                yield AgentEvent("message", content, iteration=iteration)

            if not tool_calls:
                if self._should_nudge(records, nudged):
                    nudged = True
                    nudge = Message.user(TOOL_FIRST_NUDGE)
                    messages.append(nudge)
                    transcript.append(nudge)
                    yield AgentEvent("nudge", TOOL_FIRST_NUDGE, iteration=iteration)
                    continue
                final_text = content.strip()
                stop_reason = "final"
                break

            for call in tool_calls:
                yield AgentEvent("tool_call", tool_call=call, iteration=iteration)
                result = await self.tools.call(call.name, call.arguments)
                records.append(ToolCallRecord(call.name, call.arguments, result))
                yield AgentEvent(
                    "tool_result", tool_call=call, tool_result=result, iteration=iteration
                )
                tool_msg = Message.tool(
                    call.name, result.for_model(self.config.max_tool_output_chars), call.id
                )
                messages.append(tool_msg)
                transcript.append(tool_msg)
        else:
            final_text = self._max_iterations_text(transcript)
            yield AgentEvent(
                "error",
                f"Stopped after {self.config.max_iterations} iterations without a final answer.",
                iteration=iteration,
            )

        self._remember(task, transcript, final_text)
        outcome = AgentResult(
            output=final_text,
            agent=self.name,
            model=self.model.name,
            iterations=iteration,
            tool_calls=records,
            usage=usage,
            messages=transcript,
            stop_reason=stop_reason,
            thinking="\n".join(thinking_total),
        )
        yield AgentEvent("final", final_text, iteration=iteration, metadata={"result": outcome})

    def reset(self) -> None:
        """Forget conversation history."""
        self.memory.clear()

    def get_capabilities(self) -> AgentCapabilities:
        return AgentCapabilities(
            name=self.name,
            description=self.description,
            model=self.model.name,
            model_role=self.model_role,
            tools=[t.name for t in self.tools.usable()],
            permissions=self.tools.permissions.granted(),
            can_delegate=self.can_delegate and "delegate" in self.tools,
            max_iterations=self.config.max_iterations,
        )

    # ------------------------------------------------------------------ #
    # Extension points
    # ------------------------------------------------------------------ #

    def build_system_prompt(self, context: Any = None) -> str:
        """Compose the system prompt. Override to add dynamic context."""
        prompt = (self.config.system_prompt or self.system_prompt).strip()
        prompt += (
            "\n\nAlways reply in the language the user writes in (English unless they use another)."
        )
        if self.workspace is not None:
            prompt += (
                f"\n\nWorkspace: {self.workspace.root}\n"
                f"All file paths are relative to this directory and must stay inside it. "
                f"Permissions: {self.workspace.describe().split('(')[-1].rstrip(')')}."
            )
        return prompt

    def format_task(self, task: str, context: Any = None) -> str:
        """Turn the task plus optional context into the user message."""
        if context is None or context == "":
            return task
        if isinstance(context, str):
            extra = context
        else:
            try:
                extra = json.dumps(context, indent=2, default=str)
            except (TypeError, ValueError):
                extra = str(context)
        return f"{task}\n\nContext:\n{extra}"

    # ------------------------------------------------------------------ #
    # Internals
    # ------------------------------------------------------------------ #

    async def _build_messages(self, task: str, context: Any) -> list[Message]:
        system = self.build_system_prompt(context)
        if len(self.tools) and not await self._use_native_tools():
            system += "\n\n" + text_tool_protocol(self.tools)
        messages = [Message.system(system)]
        messages.extend(self.memory.get())
        messages.append(Message.user(self.format_task(task, context)))
        return messages

    async def _use_native_tools(self) -> bool:
        if self._native_tools is None:
            if not len(self.tools):
                self._native_tools = False
            else:
                try:
                    self._native_tools = await self.model.supports_tools()
                except Exception:  # provider-specific failure: fall back to text protocol
                    log.debug("Could not determine tool support for %s", self.model, exc_info=True)
                    self._native_tools = False
        return self._native_tools

    def _should_nudge(self, records: list[ToolCallRecord], nudged: bool) -> bool:
        return self.tool_first and not nudged and not records and bool(self.tools.usable())

    def _remember(self, task: str, transcript: list[Message], final_text: str) -> None:
        self.memory.add(Message.user(task))
        if self.config.remember_tool_calls:
            self.memory.extend(transcript)
        elif final_text:
            self.memory.add(Message.assistant(final_text))

    @staticmethod
    def _max_iterations_text(transcript: list[Message]) -> str:
        for message in reversed(transcript):
            if message.role == "assistant" and message.content.strip():
                return message.content.strip()
        return "I could not finish the task within the allowed number of steps."

    def __repr__(self) -> str:
        return f"<{type(self).__name__} name={self.name!r} model={self.model.name!r}>"


class _TokenGate:
    """Hold back streamed text that looks like a textual tool call.

    Lines that begin with ``{`` or a code fence are buffered until the line
    completes; if the line turns out to be a tool call (or a fence around one)
    it is swallowed, otherwise it is released verbatim. Ordinary prose streams
    through untouched, so the UI never shows raw tool-call JSON.
    """

    def __init__(self) -> None:
        self._line = ""
        self._holding = False
        self._in_fence = False
        self._fence: list[str] = []

    def feed(self, text: str) -> str:
        out: list[str] = []
        for char in text:
            if char == "\n":
                out.append(self._end_line())
            else:
                self._line += char
                if not self._holding and not self._in_fence:
                    stripped = self._line.lstrip()
                    if stripped and stripped[0] in "{`":
                        self._holding = True
                    else:
                        out.append(self._line)
                        self._line = ""
        return "".join(out)

    def _end_line(self) -> str:
        line, self._line, self._holding = self._line, "", False
        if self._in_fence:
            self._fence.append(line)
            if line.strip().startswith("```"):
                self._in_fence = False
                block = "\n".join(self._fence) + "\n"
                self._fence = []
                calls, _ = parse_text_tool_calls(block)
                return "" if calls else block
            return ""
        stripped = line.strip()
        if stripped.startswith("```"):
            self._in_fence = True
            self._fence = [line]
            return ""
        if stripped.startswith("{") and _tool_call_from_json(stripped) is not None:
            return ""
        return line + "\n"

    def flush(self) -> str:
        pending = ""
        if self._in_fence:
            pending = "\n".join(self._fence) + ("\n" if self._line else "")
            self._fence = []
            self._in_fence = False
        pending += self._line
        self._line, self._holding = "", False
        calls, _ = parse_text_tool_calls(pending)
        return "" if calls else pending


# --------------------------------------------------------------------------- #
# Text tool-call protocol (fallback for models without native tool calling)
# --------------------------------------------------------------------------- #

_TOOL_BLOCK_RE = re.compile(r"```(?:tool_call|tool|json)?\s*\n(.*?)```", re.DOTALL)
_JSON_DECODER = json.JSONDecoder()


def text_tool_protocol(tools: ToolSet) -> str:
    lines = [
        "You can use tools. To call one, reply with ONLY a fenced block like:",
        "```tool_call",
        '{"name": "<tool name>", "arguments": {"<param>": "<value>"}}',
        "```",
        "You may include several blocks to call several tools. After the results "
        "are returned you will be asked to continue. When you have the final answer, "
        "reply in plain text without any tool_call block.",
        "",
        "Available tools:",
    ]
    for tool in tools.usable():
        spec = tool.spec()
        params = ", ".join(
            f"{name}: {schema.get('type', 'string')}"
            for name, schema in spec.parameters.get("properties", {}).items()
        )
        lines.append(f"- {spec.name}({params}): {spec.description}")
    return "\n".join(lines)


def parse_text_tool_calls(content: str) -> tuple[list[ToolCall], str]:
    """Extract tool calls written as text from ``content``.

    Two shapes are recognised: fenced blocks (```tool_call ... ```) and bare
    JSON objects of the form ``{"name": ..., "arguments": {...}}``. Anything
    else is left untouched so ordinary JSON in an answer is not eaten.
    Returns the parsed calls and the content with those spans removed.
    """
    calls: list[ToolCall] = []

    def _replace_block(match: re.Match[str]) -> str:
        call = _tool_call_from_json(match.group(1).strip())
        if call is None:
            return match.group(0)
        calls.append(call)
        return ""

    remaining = _TOOL_BLOCK_RE.sub(_replace_block, content)
    remaining = _strip_bare_json_calls(remaining, calls)
    return calls, remaining.strip()


def _tool_call_from_json(text: str) -> ToolCall | None:
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        return None
    return _tool_call_from_obj(data)


def _tool_call_from_obj(data: Any) -> ToolCall | None:
    if not isinstance(data, dict) or not isinstance(data.get("name"), str):
        return None
    if "arguments" not in data and "parameters" not in data:
        return None
    args = data.get("arguments", data.get("parameters")) or {}
    if not isinstance(args, dict):
        args = {"input": args}
    return ToolCall(name=data["name"], arguments=args)


def _strip_bare_json_calls(text: str, calls: list[ToolCall]) -> str:
    """Find top-level ``{...}`` objects that are tool calls and remove them."""
    out: list[str] = []
    pos = 0
    while True:
        start = text.find("{", pos)
        if start == -1:
            out.append(text[pos:])
            break
        try:
            data, end = _JSON_DECODER.raw_decode(text, start)
        except json.JSONDecodeError:
            out.append(text[pos : start + 1])
            pos = start + 1
            continue
        call = _tool_call_from_obj(data)
        if call is None:
            out.append(text[pos:end])
        else:
            calls.append(call)
            out.append(text[pos:start])
        pos = end
    return "".join(out)
