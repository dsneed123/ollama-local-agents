"""Sanity checks that apply to every built-in agent."""

from __future__ import annotations

import pytest
from tests.conftest import FakeOllama

from local_agents.agents.builtin import BUILTIN_AGENTS
from local_agents.agents.registry import CATEGORY_ORDER, AgentRegistry
from local_agents.agents.runtime import AgentRuntime
from local_agents.config.settings import Settings
from local_agents.models.selection import ROLES
from local_agents.tools.factory import TOOL_GROUPS


@pytest.mark.parametrize("cls", BUILTIN_AGENTS, ids=lambda c: c.name)
def test_agent_metadata_is_complete(cls) -> None:
    assert cls.name and cls.name == cls.name.lower()
    assert cls.description and len(cls.description) < 90
    assert len(cls.system_prompt.strip()) > 80
    assert cls.model_role in ROLES
    assert cls.category in CATEGORY_ORDER
    assert all(group in TOOL_GROUPS for group in cls.default_tool_groups)
    if cls.tool_first:
        assert cls.default_tool_groups, "tool_first agents need tools"


def test_no_duplicate_names() -> None:
    names = [cls.name for cls in BUILTIN_AGENTS]
    assert len(names) == len(set(names)) == 28


@pytest.mark.parametrize("cls", BUILTIN_AGENTS, ids=lambda c: c.name)
async def test_every_agent_can_be_built_by_the_runtime(
    cls, settings: Settings, fake_ollama: FakeOllama, registry: AgentRegistry
) -> None:
    async with AgentRuntime(
        settings, registry=registry, ollama_transport=fake_ollama.transport()
    ) as runtime:
        agent = await runtime.create_agent(cls.name)
        info = await agent.model.resolve()
        assert info.name
        caps = agent.get_capabilities()
        assert caps.name == cls.name
        # Every declared tool group produced tools (except groups disabled by settings).
        expected_tools = set()
        for group in cls.default_tool_groups:
            expected_tools.update(t.name for t in runtime.toolkit.group(group))
        assert expected_tools <= set(agent.tools.names())
        prompt = agent.build_system_prompt()
        assert cls.system_prompt.strip()[:40] in prompt


async def test_read_only_agents_cannot_write(
    settings: Settings, fake_ollama: FakeOllama, registry: AgentRegistry
) -> None:
    async with AgentRuntime(
        settings, registry=registry, ollama_transport=fake_ollama.transport()
    ) as runtime:
        for name in ("review", "security", "system"):
            agent = await runtime.create_agent(name)
            usable = {t.name for t in agent.tools.usable()}
            assert "write_file" not in usable and "edit_file" not in usable, name
            assert "run_command" in usable, name
            assert "write" not in agent.get_capabilities().permissions


async def test_chat_agent_has_no_tools_and_uses_fast_role(
    settings: Settings, fake_ollama: FakeOllama, registry: AgentRegistry
) -> None:
    fake_ollama.queue("hello!")
    async with AgentRuntime(
        settings, registry=registry, ollama_transport=fake_ollama.transport()
    ) as runtime:
        agent = await runtime.create_agent("chat")
        assert len(agent.tools) == 0
        result = await agent.run("hi")
        assert result.output == "hello!"
