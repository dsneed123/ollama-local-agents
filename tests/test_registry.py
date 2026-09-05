from __future__ import annotations

from pathlib import Path

import pytest

from local_agents.agents.base import BaseAgent
from local_agents.agents.registry import AgentRegistry, get_registry, register_agent
from local_agents.errors import AgentNotFoundError, AgentRegistrationError


def test_builtin_agents_discovered_in_menu_order(registry: AgentRegistry) -> None:
    names = registry.names()
    assert names[:6] == ["research", "coding", "web", "file", "data", "planning"]
    assert len(names) == 28
    assert registry.discovered
    assert registry.get("Coding").name == "coding"
    assert "research" in registry and "nope" not in registry
    assert [r.name for r in registry] == names


def test_grouped_by_category(registry: AgentRegistry) -> None:
    groups = registry.grouped()
    assert [c for c, _ in groups] == [
        "Research & Knowledge",
        "Software Development",
        "Data & Files",
        "Writing & Communication",
        "Everyday",
    ]
    by_cat = dict(groups)
    assert [r.name for r in by_cat["Research & Knowledge"]] == ["research", "web", "planning"]
    assert by_cat["Data & Files"][2].label == "SQL Agent"
    assert registry.registration("coding").label == "Coding Agent"

    class Odd(BaseAgent):
        name = "odd"
        category = "Zeta"

    registry.register(Odd)
    assert registry.grouped()[-1][0] == "Zeta"


def test_register_decorator_with_custom_registry() -> None:
    reg = AgentRegistry()

    @register_agent(name="Custom", description="A custom agent", registry=reg)
    class CustomAgent(BaseAgent):
        pass

    assert CustomAgent.name == "custom"
    assert CustomAgent.description == "A custom agent"
    assert reg.get("custom") is CustomAgent
    assert reg.registration("custom").source == "code"
    assert "custom" not in get_registry()


def test_register_infers_name_and_description() -> None:
    reg = AgentRegistry()

    class SummaryWriterAgent(BaseAgent):
        """Writes summaries.

        More detail here.
        """

    reg.register(SummaryWriterAgent)
    assert SummaryWriterAgent.name == "summary-writer"
    assert SummaryWriterAgent.description == "Writes summaries."


def test_duplicate_registration_rules() -> None:
    reg = AgentRegistry()

    class A(BaseAgent):
        name = "same"

    class B(BaseAgent):
        name = "same"

    reg.register(A)
    reg.register(A)  # idempotent for the same class
    with pytest.raises(AgentRegistrationError):
        reg.register(B)
    reg.register(B, replace=True)
    assert reg.get("same") is B
    reg.unregister("same")
    with pytest.raises(AgentNotFoundError) as info:
        reg.get("same")
    assert info.value.available == []


def test_register_rejects_non_agents() -> None:
    reg = AgentRegistry()
    with pytest.raises(AgentRegistrationError):
        reg.register(object)  # type: ignore[arg-type]


def test_hidden_agents(registry: AgentRegistry) -> None:
    class Secret(BaseAgent):
        name = "secret"

    registry.register(Secret, hidden=True)
    assert "secret" not in registry.names()
    assert "secret" in [r.name for r in registry.list_agents(include_hidden=True)]


def _write_agent(
    directory: Path, name: str, *, broken: bool = False, with_tools: bool = False
) -> Path:
    pkg = directory / name.replace("-", "_")
    pkg.mkdir(parents=True)
    tools_import = "from .tools import Extra\n" if with_tools else ""
    body = "raise RuntimeError('boom')\n" if broken else ""
    (pkg / "agent.py").write_text(
        "from local_agents import BaseAgent, register_agent\n"
        f"{tools_import}\n"
        f"{body}"
        f"@register_agent(name={name!r}, description='dir agent')\n"
        f"class DirAgent(BaseAgent):\n    system_prompt = 'x'\n"
    )
    if with_tools:
        (pkg / "tools.py").write_text("class Extra:\n    pass\n")
        (pkg / "__init__.py").write_text("")
    return pkg


def test_load_directory_agents_with_relative_imports(tmp_path: Path) -> None:
    agents_dir = tmp_path / "agents"
    _write_agent(agents_dir, "dir-one", with_tools=True)
    (agents_dir / "flat.py").write_text(
        "from local_agents import BaseAgent, register_agent\n"
        "@register_agent(name='flat')\nclass Flat(BaseAgent):\n    pass\n"
    )
    (agents_dir / "_private.py").write_text("raise RuntimeError('should not load')\n")
    (agents_dir / "dir_one" / "config.yaml").write_text("agent:\n  model: qwen3:8b\n")
    reg = AgentRegistry()
    report = reg.discover(
        builtins=False, plugins=False, paths=[agents_dir], include_default_dir=False
    )
    assert report.errors == []
    assert set(reg.names()) == {"dir-one", "flat"}
    assert reg.registration("dir-one").source.endswith("agent.py")
    assert reg.registration("dir-one").config_path == agents_dir / "dir_one" / "config.yaml"
    assert reg.registration("flat").config_path is None
    # Re-discovery is idempotent.
    reg.discover(builtins=False, plugins=False, paths=[agents_dir], include_default_dir=False)
    assert len(reg) == 2


def test_broken_agent_module_is_reported_not_fatal(tmp_path: Path) -> None:
    agents_dir = tmp_path / "agents"
    _write_agent(agents_dir, "broken", broken=True)
    _write_agent(agents_dir, "fine")
    reg = AgentRegistry()
    report = reg.discover(
        builtins=False, plugins=False, paths=[agents_dir], include_default_dir=False
    )
    assert reg.names() == ["fine"]
    assert len(report.errors) == 1 and "boom" in report.errors[0][1]


def test_default_agents_dir_in_cwd(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    _write_agent(tmp_path / "agents", "local-one")
    reg = AgentRegistry()
    reg.discover(builtins=False, plugins=False)
    assert reg.names() == ["local-one"]


def test_clear(registry: AgentRegistry) -> None:
    registry.clear()
    assert len(registry) == 0 and not registry.discovered
