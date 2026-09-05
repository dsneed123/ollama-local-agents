"""Agent runtime: wires settings, provider, registry, tools, and memory together.

:class:`AgentRuntime` is the single place that knows how to turn an agent
*name* into a ready-to-run agent instance. It also implements
:class:`AgentExecutor`, so agents that receive the runtime can delegate work
to other agents through the :class:`DelegateTool`.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

import httpx

from local_agents.agents.base import AgentConfig, AgentResult, BaseAgent
from local_agents.agents.registry import AgentRegistry, DiscoveryReport, get_registry
from local_agents.config.settings import Settings, load_agent_config, load_settings
from local_agents.errors import AgentError, LocalAgentsError
from local_agents.memory.base import BaseMemory
from local_agents.memory.conversation import ConversationMemory
from local_agents.models.base import BaseModel, GenerationOptions, ModelInfo
from local_agents.models.ollama import OllamaModel, OllamaProvider
from local_agents.models.selection import ModelRecommendations, ModelRequirements, ModelSelector
from local_agents.security.commands import CommandPolicy
from local_agents.security.permissions import Permission
from local_agents.security.workspace import Workspace
from local_agents.tools.base import ApprovalCallback, BaseTool, ToolParameter, ToolResult, ToolSet
from local_agents.tools.factory import ToolKit
from local_agents.tools.shell import ConfirmCallback

log = logging.getLogger(__name__)

MAX_DELEGATION_DEPTH = 3


@dataclass
class RuntimeStatus:
    """Snapshot of provider connectivity for the CLI banner."""

    host: str
    connected: bool
    version: str | None = None
    models: list[ModelInfo] = field(default_factory=list)
    recommendations: ModelRecommendations | None = None
    error: str | None = None


class DelegateTool(BaseTool):
    """Lets one agent run another agent through the :class:`AgentExecutor`."""

    name = "delegate"
    description = (
        "Delegate a self-contained sub-task to another specialised agent and get its "
        "final answer back. Use the agent best suited to the sub-task."
    )
    permission = Permission.DELEGATE

    def __init__(
        self, executor: AgentRuntime, *, exclude: Sequence[str] = (), depth: int = 0
    ) -> None:
        self._executor = executor
        self._exclude = {n.lower() for n in exclude}
        self._depth = depth
        names = [n for n, _ in executor.available_agents() if n not in self._exclude]
        self.parameters = (
            ToolParameter("agent", "string", "Name of the agent to run", enum=names or None),
            ToolParameter("task", "string", "Clear, self-contained description of the sub-task"),
        )

    async def execute(self, agent: str, task: str) -> ToolResult:
        if agent.lower() in self._exclude:
            raise AgentError(f"Agent '{agent}' cannot delegate to itself")
        result = await self._executor.run(agent, task, depth=self._depth + 1)
        header = f"[{result.agent} via {result.model}, {result.iterations} steps]\n"
        return ToolResult.ok(header + result.output, data=result)


class AgentRuntime:
    """Create and run agents from configuration.

    Typical use::

        runtime = AgentRuntime()              # loads config, detects Ollama
        agent = await runtime.create_agent("research")
        result = await agent.run("What is ...?")
    """

    def __init__(
        self,
        settings: Settings | None = None,
        *,
        provider: OllamaProvider | None = None,
        registry: AgentRegistry | None = None,
        workspace: Workspace | None = None,
        confirm: ConfirmCallback | None = None,
        approval: ApprovalCallback | None = None,
        http_client: httpx.AsyncClient | None = None,
        ollama_transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        self.settings = settings or load_settings()
        self.provider = provider or OllamaProvider.from_settings(
            self.settings.ollama, transport=ollama_transport
        )
        self.registry = registry if registry is not None else get_registry()
        self._workspace = workspace
        self._confirm = confirm
        self._approval = approval
        self._http_client = http_client
        self._toolkit: ToolKit | None = None
        self._discovery: DiscoveryReport | None = None

    # ------------------------------------------------------------------ #
    # Wiring
    # ------------------------------------------------------------------ #

    @property
    def workspace(self) -> Workspace:
        if self._workspace is None:
            self._workspace = Workspace.from_settings(self.settings.workspace)
        return self._workspace

    @property
    def toolkit(self) -> ToolKit:
        if self._toolkit is None:
            policy = CommandPolicy.from_settings(self.settings.security)
            self._toolkit = ToolKit(
                workspace=self.workspace,
                policy=policy,
                confirm=self._confirm,
                approval=self._approval if self.settings.security.confirm_writes else None,
                http_client=self._http_client,
                allow_shell=self.settings.security.allow_shell,
                allow_network=self.settings.security.allow_network,
            )
        return self._toolkit

    def discover(self) -> DiscoveryReport:
        """Load agents from built-ins, plugins and configured directories.

        Agents that ship a ``config.yaml`` next to their module get those
        values applied unless the main configuration already overrides them.
        """
        self._discovery = self.registry.discover(paths=self.settings.agent_paths)
        for reg in self.registry.list_agents(include_hidden=True):
            if reg.config_path is not None and reg.name not in self.settings.agents:
                try:
                    self.settings.agents[reg.name] = load_agent_config(reg.config_path)
                except LocalAgentsError as exc:
                    log.warning("Ignoring %s: %s", reg.config_path, exc)
                    self._discovery.errors.append((str(reg.config_path), str(exc)))
        return self._discovery

    @property
    def discovery(self) -> DiscoveryReport | None:
        return self._discovery

    def available_agents(self) -> list[tuple[str, str]]:
        if not self.registry.discovered:
            self.discover()
        return [
            (reg.name, reg.description)
            for reg in self.registry.list_agents()
            if self.settings.agent(reg.name).enabled
        ]

    # ------------------------------------------------------------------ #
    # Provider status
    # ------------------------------------------------------------------ #

    async def status(self, *, refresh: bool = False) -> RuntimeStatus:
        """Check Ollama and discover models. Never raises."""
        host = self.provider.host
        if not await self.provider.health_check():
            return RuntimeStatus(host=host, connected=False, error="Ollama is not reachable")
        try:
            models = await self.provider.list_models(refresh=refresh)
        except LocalAgentsError as exc:
            return RuntimeStatus(host=host, connected=True, error=str(exc))
        recommendations = ModelSelector(models).recommend() if models else None
        return RuntimeStatus(
            host=host,
            connected=True,
            version=await self.provider.version(),
            models=models,
            recommendations=recommendations,
        )

    async def refresh_models(self) -> list[ModelInfo]:
        return await self.provider.list_models(refresh=True)

    # ------------------------------------------------------------------ #
    # Agent construction
    # ------------------------------------------------------------------ #

    def resolve_model(
        self, agent_cls: type[BaseAgent], model: str | BaseModel | None = None
    ) -> BaseModel:
        """Pick the model for ``agent_cls`` honouring explicit > config > auto."""
        if isinstance(model, BaseModel):
            return model
        agent_settings = self.settings.agent(agent_cls.name)
        options = GenerationOptions(
            temperature=agent_settings.temperature, context_length=agent_settings.context_length
        )
        requested = model or agent_settings.model or "auto"
        if requested == "auto" and self.settings.default_model != "auto":
            requested = self.settings.default_model
        if requested == "auto":
            return self.provider.auto(
                agent_cls.model_role,
                require_tools=agent_cls.requires_tool_calling
                and bool(agent_cls.default_tool_groups),
                options=options,
            )
        return OllamaModel(requested, provider=self.provider, options=options)

    def build_tools(
        self,
        agent_cls: type[BaseAgent],
        *,
        groups: Sequence[str] | None = None,
        extra: Sequence[BaseTool] = (),
        depth: int = 0,
    ) -> ToolSet:
        chosen = tuple(groups) if groups is not None else agent_cls.default_tool_groups
        extras: list[BaseTool] = list(extra)
        extras.extend(agent_cls.custom_tools(self.workspace))
        if agent_cls.can_delegate and depth < MAX_DELEGATION_DEPTH:
            extras.append(DelegateTool(self, exclude=(agent_cls.name,), depth=depth))
        return self.toolkit.build(chosen, permissions=agent_cls.default_permissions, extra=extras)

    def build_config(
        self, agent_cls: type[BaseAgent], config: AgentConfig | None = None
    ) -> AgentConfig:
        cfg = config or AgentConfig()
        agent_settings = self.settings.agent(agent_cls.name)
        if agent_settings.max_iterations and config is None:
            cfg.max_iterations = agent_settings.max_iterations
        if agent_settings.temperature is not None and cfg.temperature is None:
            cfg.temperature = agent_settings.temperature
        if agent_settings.context_length is not None and cfg.context_length is None:
            cfg.context_length = agent_settings.context_length
        return cfg

    async def create_agent(
        self,
        name: str,
        *,
        model: str | BaseModel | None = None,
        tools: ToolSet | Sequence[BaseTool] | None = None,
        tool_groups: Sequence[str] | None = None,
        memory: BaseMemory | None = None,
        config: AgentConfig | None = None,
        depth: int = 0,
    ) -> BaseAgent:
        """Instantiate the agent registered as ``name`` with everything injected."""
        if not self.registry.discovered:
            self.discover()
        agent_cls = self.registry.get(name)
        if not self.settings.agent(agent_cls.name).enabled:
            raise AgentError(f"Agent '{agent_cls.name}' is disabled in configuration")
        bound_model = self.resolve_model(agent_cls, model)
        toolset: ToolSet
        if isinstance(tools, ToolSet):
            toolset = tools
        elif tools is not None:
            toolset = ToolSet(
                tools,
                permissions=agent_cls.default_permissions or self.toolkit.default_permissions(),
            )
        else:
            toolset = self.build_tools(agent_cls, groups=tool_groups, depth=depth)
        return agent_cls(
            model=bound_model,
            tools=toolset,
            memory=memory if memory is not None else ConversationMemory(),
            config=self.build_config(agent_cls, config),
            executor=self,
            workspace=self.workspace if _uses_workspace(toolset) else None,
        )

    # ------------------------------------------------------------------ #
    # AgentExecutor protocol
    # ------------------------------------------------------------------ #

    async def run(
        self, agent: str, task: str, context: Any = None, *, depth: int = 0
    ) -> AgentResult:
        """Run ``agent`` on ``task`` in a fresh instance (used for delegation)."""
        if depth > MAX_DELEGATION_DEPTH:
            raise AgentError(
                f"Delegation depth exceeded ({MAX_DELEGATION_DEPTH})",
                hint="An agent is delegating in a loop.",
            )
        instance = await self.create_agent(agent, depth=depth)
        return await instance.run(task, context)

    async def close(self) -> None:
        await self.provider.close()
        if self._http_client is not None:
            await self._http_client.aclose()

    async def __aenter__(self) -> AgentRuntime:
        return self

    async def __aexit__(self, *exc: object) -> None:
        await self.close()


def requirements_for(agent_cls: type[BaseAgent]) -> ModelRequirements:
    """Model requirements implied by an agent class (used by the CLI for display)."""
    return ModelRequirements(
        role=agent_cls.model_role,
        require_tools=agent_cls.requires_tool_calling and bool(agent_cls.default_tool_groups),
    )


def _uses_workspace(toolset: ToolSet) -> bool:
    return any(hasattr(tool, "workspace") for tool in toolset)
