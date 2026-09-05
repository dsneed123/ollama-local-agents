"""Interactive chat session with one agent, with streamed output."""

from __future__ import annotations

import json
from typing import Any

from local_agents.agents.base import AgentEvent, BaseAgent
from local_agents.agents.runtime import AgentRuntime
from local_agents.cli.errors import report_error
from local_agents.cli.ui import Console, truncate
from local_agents.errors import LocalAgentsError
from local_agents.models.ollama import OllamaModel

HELP = """\
Commands:
  /model [name]   Show or switch the model for this session
  /models         List installed models
  /tools          List the agent's tools
  /info           Show agent capabilities
  /think          Toggle display of model reasoning
  /reset          Clear conversation memory
  /help           Show this help
  /back           Return to the main menu"""


class AgentSession:
    """Drive one agent interactively (or for a single task)."""

    def __init__(
        self,
        runtime: AgentRuntime,
        agent: BaseAgent,
        console: Console,
        *,
        show_thinking: bool = False,
        quiet: bool = False,
    ) -> None:
        self.runtime = runtime
        self.agent = agent
        self.console = console
        self.show_thinking = show_thinking
        self.quiet = quiet
        self._debug = runtime.settings.debug

    # ------------------------------------------------------------------ #

    async def run_task(self, task: str) -> str:
        """Run one task, printing streamed output. Returns the final answer."""
        style = self.console.style
        final = ""
        streamed_text = False
        try:
            async for event in self.agent.stream(task):
                streamed_text = self._render(event, streamed_text) or streamed_text
                if event.type == "final":
                    final = event.content
        except LocalAgentsError as exc:
            self.console.print()
            report_error(self.console, exc, debug=self._debug)
            return ""
        except KeyboardInterrupt:
            self.console.print()
            self.console.warn("Cancelled")
            return ""
        if streamed_text:
            self.console.print()
        if not final and not streamed_text:
            self.console.print(style.dim("(no answer)"))
        return final

    def _render(self, event: AgentEvent, streamed_text: bool) -> bool:
        style = self.console.style
        if event.type == "start" and not self.quiet:
            self.console.print(
                style.dim(f"[{event.metadata.get('agent')} · {event.metadata.get('model')}]")
            )
        elif event.type == "thinking" and self.show_thinking:
            self.console.write(style.dim(event.content))
            return True
        elif event.type == "token":
            self.console.write(event.content)
            return True
        elif event.type == "tool_call" and event.tool_call is not None:
            if streamed_text:
                self.console.print()
            args = _format_args(event.tool_call.arguments)
            self.console.print(style.magenta("⚙ ") + f"{event.tool_call.name}({args})")
        elif event.type == "tool_result" and event.tool_result is not None:
            result = event.tool_result
            if result.blocked:
                self.console.print(
                    style.yellow("  ⛔ blocked: ") + truncate(result.error or "", 100)
                )
            elif not result.success:
                self.console.print(style.red("  ✗ ") + truncate(result.error or "failed", 100))
            elif not self.quiet:
                first = result.output.strip().splitlines()[0] if result.output.strip() else "ok"
                self.console.print(
                    style.dim(f"  ↳ {truncate(first, 100)} ({result.duration:.1f}s)")
                )
            return False
        elif event.type == "nudge":
            if streamed_text:
                self.console.print()
            self.console.print(style.dim("↻ answered without tools; asking it to research first"))
        elif event.type == "error":
            self.console.print()
            self.console.warn(event.content)
        return False

    # ------------------------------------------------------------------ #

    async def interactive(self) -> None:
        console = self.console
        style = console.style
        console.print()
        console.print(
            style.bold(f"{self.agent.name.title()} Agent")
            + style.dim(f"  —  {self.agent.description}")
        )
        try:
            info = await self.agent.model.info()
            model_name = info.name if info else self.agent.model.name
        except LocalAgentsError as exc:
            report_error(console, exc, debug=self._debug)
            return
        console.print(
            style.dim(
                f"Model: {model_name}   Tools: {', '.join(self.agent.tools.names()) or 'none'}"
            )
        )
        if self.agent.workspace is not None:
            console.print(style.dim(f"Workspace: {self.agent.workspace.describe()}"))
        console.print(style.dim("Type a task, or /help for commands. /back returns to the menu."))
        console.rule()
        while True:
            line = console.prompt(style.bold("you> "))
            if line is None:
                break
            if not line:
                continue
            if line.startswith("/"):
                if await self._command(line):
                    break
                continue
            console.write(style.bold(f"{self.agent.name}> "))
            await self.run_task(line)
            console.print()

    async def _command(self, line: str) -> bool:
        """Handle a slash command. Returns ``True`` to leave the session."""
        console = self.console
        parts = line.split(maxsplit=1)
        command, arg = parts[0].lower(), (parts[1].strip() if len(parts) > 1 else "")
        if command in ("/back", "/quit", "/exit", "/q"):
            return True
        if command == "/help":
            console.print(HELP)
        elif command == "/reset":
            self.agent.reset()
            console.success("Conversation memory cleared")
        elif command == "/think":
            self.show_thinking = not self.show_thinking
            console.info(f"Reasoning display {'on' if self.show_thinking else 'off'}")
        elif command == "/tools":
            for tool in self.agent.tools:
                usable = self.agent.tools.permissions.allows(tool.permission)
                marker = "●" if usable else "○"
                console.print(f"  {marker} {tool.name:<18} {truncate(tool.description, 60)}")
        elif command == "/info":
            caps = self.agent.get_capabilities()
            console.print(json.dumps(caps.__dict__, indent=2))
        elif command == "/models":
            for model in await self.runtime.provider.list_models():
                console.print(f"  ● {model.describe()}")
        elif command == "/model":
            await self._switch_model(arg)
        else:
            console.warn(f"Unknown command {command}. Type /help.")
        return False

    async def _switch_model(self, name: str) -> None:
        console = self.console
        if not name:
            console.info(f"Current model: {self.agent.model.name}")
            return
        if not isinstance(self.agent.model, OllamaModel):
            console.warn("This agent's model is not an Ollama model; cannot switch")
            return
        try:
            if not await self.runtime.provider.has_model(name):
                console.error(f"Model '{name}' is not installed. Use /models to see what is.")
                return
            self.agent.model.rebind(name)
            info = await self.agent.model.resolve()
            self.agent._native_tools = None  # re-evaluate tool support for the new model
            console.success(f"Switched to {info.name}")
        except LocalAgentsError as exc:
            report_error(console, exc, debug=self._debug)


def _format_args(arguments: dict[str, Any]) -> str:
    parts = []
    for key, value in arguments.items():
        text = value if isinstance(value, str) else json.dumps(value, default=str)
        parts.append(f"{key}={truncate(str(text), 60)!r}")
    return ", ".join(parts)
