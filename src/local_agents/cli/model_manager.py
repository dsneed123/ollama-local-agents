"""Model Manager screen."""

from __future__ import annotations

from local_agents.agents.runtime import AgentRuntime
from local_agents.cli.errors import report_error
from local_agents.cli.ui import Console, format_size
from local_agents.errors import LocalAgentsError
from local_agents.models.base import ModelInfo
from local_agents.models.selection import ROLES, ModelSelector

ROLE_LABELS = {"general": "General", "coding": "Coding", "reasoning": "Reasoning", "fast": "Fast"}


class ModelManager:
    def __init__(self, runtime: AgentRuntime, console: Console) -> None:
        self.runtime = runtime
        self.console = console

    async def run(self) -> None:
        console = self.console
        while True:
            try:
                models = await self.runtime.provider.list_models()
            except LocalAgentsError as exc:
                report_error(console, exc, debug=self.runtime.settings.debug)
                return
            self._render(models)
            choice = console.prompt("Select: ")
            if choice is None or choice.lower() in ("b", "back", "q"):
                return
            if choice.lower() == "r":
                await self._refresh()
            elif choice.lower() == "i":
                await self._info(models)
            elif choice.isdigit() and 1 <= int(choice) <= len(models):
                self._set_default(models[int(choice) - 1])
            else:
                console.warn("Invalid selection")

    def _render(self, models: list[ModelInfo]) -> None:
        console = self.console
        style = console.style
        console.print()
        console.print(style.bold("MODEL MANAGER"))
        console.print()
        console.print(f"Installed Ollama Models  ({self.runtime.provider.host})")
        console.print()
        default = self.runtime.settings.default_model
        if not models:
            console.print("  (none installed — run `ollama pull <model>`)")
        for index, model in enumerate(models, 1):
            marker = " ★" if model.name == default else ""
            details = f"{model.parameter_size:>6}  {format_size(model.size_bytes):>9}"
            caps = ",".join(model.capabilities)
            console.print(f"  {index:>2}. {model.name:<28} {details}  {style.dim(caps)}{marker}")
        if models:
            console.print()
            console.print("Recommended:")
            recs = ModelSelector(models).recommend(ROLES)
            for role in ROLES:
                chosen = recs.get(role)
                if chosen:
                    console.print(f"  {ROLE_LABELS[role]:<10}→ {chosen.name}")
        console.print()
        console.print(
            f"  #. Set session default model{' (★ = current)' if default != 'auto' else ''}"
        )
        console.print("  a. Use automatic selection")
        console.print("  r. Refresh models")
        console.print("  i. Model information")
        console.print("  b. Back")
        console.print()

    async def _refresh(self) -> None:
        try:
            models = await self.runtime.refresh_models()
        except LocalAgentsError as exc:
            report_error(self.console, exc, debug=self.runtime.settings.debug)
            return
        self.console.success(f"Refreshed: {len(models)} models detected")

    async def _info(self, models: list[ModelInfo]) -> None:
        console = self.console
        choice = console.prompt("Model number: ")
        if not choice or not choice.isdigit() or not 1 <= int(choice) <= len(models):
            console.warn("Invalid selection")
            return
        model = models[int(choice) - 1]
        try:
            details = await self.runtime.provider.show(model.name)
        except LocalAgentsError as exc:
            report_error(console, exc, debug=self.runtime.settings.debug)
            return
        console.print()
        console.print(console.style.bold(model.name))
        console.print(f"  Family:        {model.family or '?'}")
        console.print(f"  Parameters:    {model.parameter_size or '?'}")
        console.print(f"  Quantization:  {model.quantization or '?'}")
        console.print(f"  Context:       {model.context_length or '?'}")
        console.print(f"  Size:          {format_size(model.size_bytes)}")
        console.print(f"  Capabilities:  {', '.join(model.capabilities) or '?'}")
        console.print(f"  Modified:      {model.modified_at[:19] or '?'}")
        params = details.get("parameters")
        if params:
            console.print("  Parameters (Modelfile):")
            for line in str(params).splitlines()[:12]:
                console.print(f"    {line}")
        license_text = str(details.get("license") or "").strip().splitlines()
        if license_text:
            console.print(f"  License:       {license_text[0][:70]}")
        console.print()

    def _set_default(self, model: ModelInfo) -> None:
        self.runtime.settings.default_model = model.name
        self.console.success(f"Session default model set to {model.name}")
