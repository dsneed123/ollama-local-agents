from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

from local_agents.agents.registry import AgentRegistry
from local_agents.errors import LocalAgentsError
from local_agents.scaffold.generator import (
    class_name_for,
    create_agent_scaffold,
    module_name_for,
    suggest_agent_name,
)
from local_agents.testing import ScriptedModel, ScriptedTurn


def test_name_helpers() -> None:
    assert suggest_agent_name("  My Cool Agent!! ") == "my-cool-agent"
    assert class_name_for("my-cool-agent") == "MyCoolAgent"
    assert class_name_for("summary") == "SummaryAgent"
    assert class_name_for("3d") == "Agent3dAgent"
    assert module_name_for("my-cool-agent") == "my_cool_agent"
    assert module_name_for("class") == "agent_class"
    with pytest.raises(ValueError):
        suggest_agent_name("!!!")


def test_scaffold_creates_all_files(tmp_path: Path) -> None:
    created = create_agent_scaffold(
        "ticket-triage",
        directory=tmp_path / "agents",
        description='Triages "tickets"',
        tool_groups=("filesystem", "web"),
    )
    root = tmp_path / "agents" / "ticket_triage"
    names = sorted(p.relative_to(root).as_posix() for p in created)
    assert names == sorted(
        [
            "agent.py",
            "tools.py",
            "config.yaml",
            "README.md",
            "__init__.py",
            "tests/__init__.py",
            "tests/test_agent.py",
        ]
    )
    agent_src = (root / "agent.py").read_text()
    assert '@register_agent(name="ticket-triage", description="Triages \'tickets\'")' in agent_src
    assert "class TicketTriageAgent(BaseAgent)" in agent_src
    assert "default_tool_groups = ('filesystem', 'web')" in agent_src
    assert "model: auto" in (root / "config.yaml").read_text()


def test_scaffold_refuses_overwrite_and_validates_groups(tmp_path: Path) -> None:
    create_agent_scaffold("a", directory=tmp_path)
    with pytest.raises(LocalAgentsError, match="already exists"):
        create_agent_scaffold("a", directory=tmp_path)
    create_agent_scaffold("a", directory=tmp_path, force=True)
    with pytest.raises(LocalAgentsError, match="Unknown tool group"):
        create_agent_scaffold("b", directory=tmp_path, tool_groups=("nope",))


async def test_generated_agent_registers_and_runs(tmp_path: Path) -> None:
    create_agent_scaffold("demo-bot", directory=tmp_path / "agents", description="Demo")
    registry = AgentRegistry()
    report = registry.discover(
        builtins=False, plugins=False, paths=[tmp_path / "agents"], include_default_dir=False
    )
    assert report.errors == []
    cls = registry.get("demo-bot")
    assert cls.description == "Demo" and cls.model_role == "general"
    assert registry.registration("demo-bot").config_path is not None
    tools = cls.custom_tools()
    assert [t.name for t in tools] == ["current_time"]
    agent = cls(
        model=ScriptedModel([ScriptedTurn.call("current_time", utc=True), "Now you know."]),
        tools=tools,
    )
    result = await agent.run("time?")
    assert result.output == "Now you know."
    assert result.tool_calls[0].result.success and "UTC" in result.tool_calls[0].result.output


def test_generated_tests_pass(tmp_path: Path) -> None:
    create_agent_scaffold("checked", directory=tmp_path / "agents")
    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            "-q",
            "-p",
            "no:cacheprovider",
            str(tmp_path / "agents" / "checked" / "tests"),
        ],
        capture_output=True,
        text=True,
        cwd=tmp_path,
        timeout=120,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "3 passed" in proc.stdout
