"""Agent Manager screen: inspect agents, toggle them, and assign models."""

from __future__ import annotations

from local_agents.agents.registry import AgentRegistration
from local_agents.agents.runtime import AgentRuntime, requirements_for
from local_agents.cli.errors import report_error
from local_agents.cli.ui import Console, truncate
from local_agents.errors import LocalAgentsError
from local_agents.models.selection import ModelSelector


class AgentManager:
    def __init__(self, runtime: AgentRuntime, console: Console) -> None:
        self.runtime = runtime
        self.console = console

    async def run(self) -> None:
        console = self.console
        while True:
            regs = self.runtime.registry.list_agents()
            self._render(regs)
            choice = console.prompt("Select: ")
            if choice is None or choice.lower() in ("b", "back", "q"):
                return
            if choice.isdigit() and 1 <= int(choice) <= len(regs):
                await self._detail(regs[int(choice) - 1])
            else:
                console.warn("Invalid selection")

    def _render(self, regs: list[AgentRegistration]) -> None:
        console = self.console
        style = console.style
        console.print()
        console.print(style.bold("AGENT MANAGER"))
        console.print()
        for index, reg in enumerate(regs, 1):
            settings = self.runtime.settings.agent(reg.name)
            state = style.green("on ") if settings.enabled else style.red("off")
            source = "built-in" if reg.source in ("builtin", "code") else truncate(reg.source, 30)
            console.print(
                f"  {index:>2}. {reg.name:<12} {state}  model={settings.model:<18} "
                f"{style.dim(source)}"
            )
        if self.runtime.discovery and self.runtime.discovery.errors:
            console.print()
            console.warn("Some agents failed to load:")
            for path, error in self.runtime.discovery.errors:
                console.print(f"    {truncate(path, 40)}: {truncate(error, 60)}")
        console.print()
        console.print("  #. Inspect / configure an agent")
        console.print("  b. Back")
        console.print()

    async def _detail(self, reg: AgentRegistration) -> None:
        console = self.console
        style = console.style
        cls = reg.cls
        settings = self.runtime.settings.agent(reg.name)
        while True:
            console.print()
            console.print(style.bold(f"{reg.name} — {reg.description}"))
            console.print(f"  Class:        {cls.__module__}.{cls.__name__}")
            console.print(f"  Enabled:      {'yes' if settings.enabled else 'no'}")
            console.print(f"  Model:        {settings.model}")
            console.print(f"  Model role:   {cls.model_role}")
            console.print(f"  Tool groups:  {', '.join(cls.default_tool_groups) or 'none'}")
            console.print(f"  Delegation:   {'yes' if cls.can_delegate else 'no'}")
            console.print(f"  Max steps:    {settings.max_iterations or 'default'}")
            try:
                models = await self.runtime.provider.list_models()
                auto = ModelSelector(models).select(requirements_for(cls))
                console.print(f"  Auto pick:    {auto.name if auto else 'n/a'}")
            except LocalAgentsError:
                console.print("  Auto pick:    (Ollama unavailable)")
            console.print()
            console.print("  e. Toggle enabled")
            console.print("  m. Set model (name or 'auto')")
            console.print("  s. Show system prompt")
            console.print("  b. Back")
            choice = console.prompt("Select: ")
            if choice is None or choice.lower() in ("b", "back"):
                return
            if choice.lower() == "e":
                settings.enabled = not settings.enabled
                console.success(f"{reg.name} {'enabled' if settings.enabled else 'disabled'}")
            elif choice.lower() == "m":
                name = console.prompt("Model name (or 'auto'): ")
                if not name:
                    continue
                if name != "auto":
                    try:
                        if not await self.runtime.provider.has_model(name):
                            console.error(f"Model '{name}' is not installed")
                            continue
                    except LocalAgentsError as exc:
                        report_error(console, exc, debug=self.runtime.settings.debug)
                        continue
                settings.model = name
                console.success(f"{reg.name} will use {name}")
            elif choice.lower() == "s":
                console.print()
                console.print(style.dim(cls.system_prompt.strip()))
            else:
                console.warn("Invalid selection")
