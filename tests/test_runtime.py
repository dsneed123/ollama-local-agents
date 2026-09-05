from __future__ import annotations

import json
from pathlib import Path

import pytest
from tests.conftest import FakeOllama

from local_agents.agents.base import AgentConfig, BaseAgent
from local_agents.agents.builtin import CodingAgent, PlanningAgent
from local_agents.agents.registry import AgentRegistry
from local_agents.agents.runtime import AgentRuntime, DelegateTool
from local_agents.config.settings import AgentSettings, Settings
from local_agents.errors import AgentError, AgentNotFoundError, ModelNotFoundError
from local_agents.models.ollama import OllamaModel
from local_agents.security.permissions import Permission
from local_agents.testing import ScriptedModel, ScriptedTurn


def _runtime(settings: Settings, fake: FakeOllama, registry: AgentRegistry, **kw) -> AgentRuntime:
    return AgentRuntime(settings, registry=registry, ollama_transport=fake.transport(), **kw)


async def test_status_connected(
    settings: Settings, fake_ollama: FakeOllama, registry: AgentRegistry
) -> None:
    async with _runtime(settings, fake_ollama, registry) as runtime:
        status = await runtime.status()
        assert status.connected and status.version == "0.32.9"
        assert next(m.name for m in status.models) == "qwen3:8b"
        assert status.recommendations is not None
        assert status.recommendations.get("coding").name == "qwen2.5-coder:14b"


async def test_status_disconnected(settings: Settings, registry: AgentRegistry) -> None:
    import httpx

    fake = FakeOllama()
    fake.fail_with = httpx.ConnectError("no")
    async with _runtime(settings, fake, registry) as runtime:
        status = await runtime.status()
        assert not status.connected and status.error


async def test_create_agent_auto_model_and_tools(
    settings: Settings, fake_ollama: FakeOllama, registry: AgentRegistry
) -> None:
    async with _runtime(settings, fake_ollama, registry) as runtime:
        agent = await runtime.create_agent("coding")
        assert isinstance(agent, CodingAgent)
        assert isinstance(agent.model, OllamaModel)
        info = await agent.model.resolve()
        assert info.name == "qwen2.5-coder:14b"
        names = agent.tools.names()
        assert "read_file" in names and "run_command" in names and "web_search" not in names
        assert (
            agent.workspace is not None
            and agent.workspace.root == Path(settings.workspace.root).resolve()
        )
        assert agent.executor is runtime
        caps = agent.get_capabilities()
        assert "execute" in caps.permissions and "delete" not in caps.permissions


async def test_create_agent_honours_config_and_explicit_model(
    settings: Settings, fake_ollama: FakeOllama, registry: AgentRegistry
) -> None:
    settings.agents["research"] = AgentSettings(
        model="llama3.1:8b", max_iterations=3, temperature=0.1
    )
    async with _runtime(settings, fake_ollama, registry) as runtime:
        agent = await runtime.create_agent("research")
        assert (await agent.model.resolve()).name == "llama3.1:8b"
        assert agent.config.max_iterations == 3 and agent.config.temperature == 0.1
        explicit = await runtime.create_agent("research", model="qwen3:8b")
        assert (await explicit.model.resolve()).name == "qwen3:8b"
        injected = await runtime.create_agent("research", model=ScriptedModel(["x"]))
        assert isinstance(injected.model, ScriptedModel)
        with pytest.raises(ModelNotFoundError):
            await (await runtime.create_agent("research", model="ghost:1b")).model.resolve()


async def test_default_model_setting_applies_to_all(
    settings: Settings, fake_ollama: FakeOllama, registry: AgentRegistry
) -> None:
    settings.default_model = "llama3.1:8b"
    async with _runtime(settings, fake_ollama, registry) as runtime:
        agent = await runtime.create_agent("coding")
        assert (await agent.model.resolve()).name == "llama3.1:8b"


async def test_disabled_and_unknown_agents(
    settings: Settings, fake_ollama: FakeOllama, registry: AgentRegistry
) -> None:
    settings.agents["web"] = AgentSettings(enabled=False)
    async with _runtime(settings, fake_ollama, registry) as runtime:
        assert "web" not in [n for n, _ in runtime.available_agents()]
        with pytest.raises(AgentError, match="disabled"):
            await runtime.create_agent("web")
        with pytest.raises(AgentNotFoundError):
            await runtime.create_agent("ghost")


async def test_security_settings_shape_tools(
    settings: Settings, fake_ollama: FakeOllama, registry: AgentRegistry
) -> None:
    settings.security.allow_shell = False
    settings.security.allow_network = False
    settings.workspace.allow_write = False
    async with _runtime(settings, fake_ollama, registry) as runtime:
        coding = await runtime.create_agent("coding")
        assert "run_command" not in coding.tools.names()
        assert "write_file" not in [t.name for t in coding.tools.usable()]
        research = await runtime.create_agent("research")
        assert "web_search" not in research.tools.names()


async def test_run_agent_end_to_end_via_fake_ollama(
    settings: Settings, fake_ollama: FakeOllama, registry: AgentRegistry
) -> None:
    (Path(settings.workspace.root) / "hello.txt").write_text("hello from disk")
    fake_ollama.queue(
        "", tool_calls=[{"function": {"name": "read_file", "arguments": {"path": "hello.txt"}}}]
    )
    fake_ollama.queue("The file says: hello from disk")
    async with _runtime(settings, fake_ollama, registry) as runtime:
        result = await runtime.run("file", "What does hello.txt say?")
        assert result.output == "The file says: hello from disk"
        assert result.tool_calls[0].name == "read_file" and result.tool_calls[0].result.success
        chat_bodies = [
            json.loads(r.content) for r in fake_ollama.requests if r.url.path == "/api/chat"
        ]
        assert chat_bodies[0]["tools"]  # native tool specs were sent
        assert chat_bodies[1]["messages"][-1]["role"] == "tool"


async def test_planning_agent_gets_delegate_tool_and_prompt(
    settings: Settings, fake_ollama: FakeOllama, registry: AgentRegistry
) -> None:
    async with _runtime(settings, fake_ollama, registry) as runtime:
        planner = await runtime.create_agent("planning")
        assert isinstance(planner, PlanningAgent)
        assert "delegate" in planner.tools
        delegate = planner.tools.get("delegate")
        assert isinstance(delegate, DelegateTool)
        assert delegate.permission == Permission.DELEGATE
        assert "planning" not in delegate.parameters[0].enum
        prompt = planner.build_system_prompt()
        assert "- research:" in prompt and "- planning:" not in prompt


async def test_delegation_runs_other_agent(
    settings: Settings, fake_ollama: FakeOllama, registry: AgentRegistry
) -> None:
    fake_ollama.queue("Sub-result: 42")  # the delegated research agent's answer
    async with _runtime(settings, fake_ollama, registry) as runtime:
        planner = await runtime.create_agent(
            "planning",
            model=ScriptedModel(
                [
                    ScriptedTurn.call("delegate", agent="research", task="find the answer"),
                    "Plan complete.",
                ]
            ),
        )
        result = await planner.run("Do the thing")
        assert result.output == "Plan complete."
        record = result.tool_calls[0]
        assert record.result.success and "Sub-result: 42" in record.result.output
        assert "[research via" in record.result.output


async def test_delegation_depth_limit(
    settings: Settings, fake_ollama: FakeOllama, registry: AgentRegistry
) -> None:
    async with _runtime(settings, fake_ollama, registry) as runtime:
        with pytest.raises(AgentError, match="depth"):
            await runtime.run("research", "x", depth=99)
        deep = await runtime.create_agent("planning", depth=3)
        assert "delegate" not in deep.tools  # no further delegation at the limit


async def test_self_delegation_is_refused(
    settings: Settings, fake_ollama: FakeOllama, registry: AgentRegistry
) -> None:
    async with _runtime(settings, fake_ollama, registry) as runtime:
        tool = DelegateTool(runtime, exclude=("planning",))
        result = await tool.run(agent="planning", task="loop")
        assert not result.success and "planning" in (result.error or "")


async def test_custom_tools_and_agent_config_yaml_from_directory(
    settings: Settings, fake_ollama: FakeOllama, tmp_path: Path
) -> None:
    agents_dir = tmp_path / "custom"
    pkg = agents_dir / "hello_agent"
    pkg.mkdir(parents=True)
    (pkg / "__init__.py").write_text("")
    (pkg / "tools.py").write_text(
        "from local_agents import BaseTool\n"
        "class Ping(BaseTool):\n    name='ping'\n    description='pong'\n"
        "    async def execute(self):\n        return 'pong'\n"
    )
    (pkg / "agent.py").write_text(
        "from local_agents import BaseAgent, register_agent\nfrom .tools import Ping\n"
        "@register_agent(name='hello', description='hi')\n"
        "class HelloAgent(BaseAgent):\n"
        "    default_tool_groups = ()\n"
        "    @classmethod\n    def custom_tools(cls, workspace=None):\n        return [Ping()]\n"
    )
    (pkg / "config.yaml").write_text("agent:\n  model: llama3.1:8b\n  max_iterations: 2\n")
    settings.agent_paths = [str(agents_dir)]
    registry = AgentRegistry()
    async with _runtime(settings, fake_ollama, registry) as runtime:
        report = runtime.discover()
        assert report.errors == []
        agent = await runtime.create_agent("hello")
        assert agent.tools.names() == ["ping"]
        assert agent.workspace is None  # no workspace tools -> no workspace context
        assert (await agent.model.resolve()).name == "llama3.1:8b"
        assert agent.config.max_iterations == 2
        assert (await agent.tools.call("ping")).output == "pong"


async def test_write_approval_wiring(
    settings: Settings, fake_ollama: FakeOllama, registry: AgentRegistry
) -> None:
    calls: list[str] = []

    async def approve(tool, arguments) -> bool:
        calls.append(tool.name)
        return False

    settings.security.confirm_writes = True
    async with _runtime(settings, fake_ollama, registry, approval=approve) as runtime:
        agent = await runtime.create_agent("file")
        result = await agent.tools.call("write_file", {"path": "x.txt", "content": "y"})
        assert result.blocked and calls == ["write_file"]
    settings.security.confirm_writes = False
    async with _runtime(settings, fake_ollama, registry, approval=approve) as runtime:
        agent = await runtime.create_agent("file")
        assert agent.tools.approval is None


async def test_explicit_tool_list_and_groups(
    settings: Settings, fake_ollama: FakeOllama, registry: AgentRegistry
) -> None:
    async with _runtime(settings, fake_ollama, registry) as runtime:
        only_web = await runtime.create_agent("coding", tool_groups=["web"])
        assert only_web.tools.names() == ["web_search", "fetch_url"]
        none = await runtime.create_agent("coding", tools=[], config=AgentConfig(max_iterations=1))
        assert len(none.tools) == 0 and none.config.max_iterations == 1


def test_agent_base_can_be_used_standalone() -> None:
    class Solo(BaseAgent):
        name = "solo"

    agent = Solo(model=ScriptedModel(["hi"]))
    assert agent.tools.names() == [] and agent.executor is None
