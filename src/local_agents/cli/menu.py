"""Main menu, generated from the agent registry."""

from __future__ import annotations

from local_agents.agents.runtime import AgentRuntime, RuntimeStatus
from local_agents.cli.agent_manager import AgentManager
from local_agents.cli.errors import report_error
from local_agents.cli.model_manager import ModelManager
from local_agents.cli.session import AgentSession
from local_agents.cli.ui import BANNER, Console
from local_agents.errors import LocalAgentsError
from local_agents.scaffold.generator import create_agent_scaffold, suggest_agent_name

ROLE_LABELS = {"coding": "Coding", "general": "General", "reasoning": "Reasoning", "fast": "Fast"}


class MainMenu:
    def __init__(self, runtime: AgentRuntime, console: Console) -> None:
        self.runtime = runtime
        self.console = console
        self._status: RuntimeStatus | None = None

    async def run(self) -> int:
        console = self.console
        style = console.style
        console.print(style.cyan(BANNER))
        self._status = await self.runtime.status()
        self._print_status(self._status, verbose=True)
        report = self.runtime.discovery or self.runtime.discover()
        if report.errors:
            console.warn(f"{len(report.errors)} agent module(s) failed to load (see Agent Manager)")
        while True:
            entries = self._entries()
            self._print_menu(entries)
            choice = console.prompt("> ")
            if choice is None or choice.lower() in ("q", "quit", "exit"):
                console.print("Bye.")
                return 0
            action = entries.get(choice.lower())
            if action is None:
                console.warn("Invalid selection")
                continue
            kind, name = action
            try:
                if kind == "agent":
                    await self._deploy(name)
                elif kind == "agents":
                    await AgentManager(self.runtime, console).run()
                elif kind == "models":
                    await ModelManager(self.runtime, console).run()
                    self._status = await self.runtime.status()
                elif kind == "create":
                    await self._create_agent()
            except LocalAgentsError as exc:
                report_error(console, exc, debug=self.runtime.settings.debug)

    # ------------------------------------------------------------------ #

    def _entries(self) -> dict[str, tuple[str, str]]:
        entries: dict[str, tuple[str, str]] = {}
        index = 1
        enabled = {name for name, _ in self.runtime.available_agents()}
        for _category, regs in self.runtime.registry.grouped():
            for reg in regs:
                if reg.name not in enabled:
                    continue
                entries[str(index)] = ("agent", reg.name)
                entries[reg.name] = ("agent", reg.name)
                index += 1
        for key, label in (
            ("agents", "Agent Manager"),
            ("models", "Model Manager"),
            ("create", "Create Agent"),
        ):
            entries[str(index)] = (key, label)
            entries[key] = (key, label)
            index += 1
        return entries

    def _print_menu(self, entries: dict[str, tuple[str, str]]) -> None:
        console = self.console
        style = console.style
        console.print()
        console.print("Which agent would you like to deploy?")
        numbers = {name: key for key, (kind, name) in entries.items() if key.isdigit()}
        for category, regs in self.runtime.registry.grouped():
            visible = [r for r in regs if r.name in numbers]
            if not visible:
                continue
            console.print()
            console.print(style.dim(f"  {category}"))
            for reg in visible:
                console.print(
                    f"  {numbers[reg.name]:>2}  {reg.label:<20}{style.dim(reg.description)}"
                )
        console.print()
        for key, (kind, name) in entries.items():
            if key.isdigit() and kind != "agent":
                console.print(f"  {key:>2}  {name}")
        console.print("   q  Quit")
        console.print()

    def _print_status(self, status: RuntimeStatus, *, verbose: bool = False) -> None:
        console = self.console
        style = console.style
        console.print()
        if status.connected:
            version = f" (v{status.version})" if status.version else ""
            console.print(
                f"Ollama: {style.green('● Connected')}{style.dim(version)}   "
                f"{style.dim(status.host)}"
            )
            console.print(f"Models: {len(status.models)} detected")
            if status.error:
                console.warn(status.error)
            if verbose and status.models:
                console.print()
                console.print("Detected Ollama models:")
                for model in status.models:
                    console.print(f"  ● {model.name}")
                if status.recommendations:
                    console.print()
                    console.print("Recommended:")
                    for role in ("coding", "general", "reasoning", "fast"):
                        chosen = status.recommendations.get(role)
                        if chosen:
                            console.print(f"  {ROLE_LABELS[role]:<9} → {chosen.name}")
            elif not status.models:
                console.warn("No models installed. Run `ollama pull qwen3:8b` to get started.")
        else:
            console.print(f"Ollama: {style.red('○ Not available')}   {style.dim(status.host)}")
            console.print()
            console.warn("Ollama is not available.")
            console.print("Make sure Ollama is running and try again.")
            console.print(f"Detected endpoint: {status.host}")
            console.print(style.dim("Menus still work; agents need Ollama to run."))

    async def _deploy(self, name: str) -> None:
        agent = await self.runtime.create_agent(name)
        session = AgentSession(
            self.runtime, agent, self.console, show_thinking=self.runtime.settings.debug
        )
        await session.interactive()

    async def _create_agent(self) -> None:
        console = self.console
        console.print()
        console.print(console.style.bold("CREATE AGENT"))
        raw = console.prompt("Agent name (e.g. my-agent): ")
        if not raw:
            return
        name = suggest_agent_name(raw)
        description = console.prompt("One-line description: ") or f"{name} agent"
        try:
            created = create_agent_scaffold(name, description=description)
        except (LocalAgentsError, OSError, ValueError) as exc:
            console.error(str(exc))
            return
        console.success(f"Created agent '{name}':")
        for path in created:
            console.print(f"  {path}")
        report = self.runtime.registry.discover(paths=self.runtime.settings.agent_paths)
        if any(name == reg.name for reg in self.runtime.registry.list_agents()):
            console.info(f"'{name}' is now available in the menu.")
        for source, error in report.errors:
            console.warn(f"{source}: {error}")
