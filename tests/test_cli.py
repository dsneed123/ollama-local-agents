from __future__ import annotations

import io
import json
from pathlib import Path

import httpx
import pytest
from tests.conftest import FakeOllama

import local_agents.cli.main as cli_main
from local_agents.agents.registry import AgentRegistry
from local_agents.agents.runtime import AgentRuntime
from local_agents.cli.errors import report_error
from local_agents.cli.menu import MainMenu
from local_agents.cli.session import AgentSession
from local_agents.cli.ui import Console, format_size, truncate
from local_agents.config.settings import Settings
from local_agents.errors import (
    ModelNotFoundError,
    NoModelsAvailableError,
    ProviderConnectionError,
    ToolError,
)
from local_agents.models.ollama import OllamaProvider
from local_agents.testing import ScriptedModel, ScriptedTurn


def _runtime(settings: Settings, fake: FakeOllama, registry: AgentRegistry) -> AgentRuntime:
    return AgentRuntime(settings, registry=registry, ollama_transport=fake.transport())


def _console(*inputs: str) -> tuple[Console, io.StringIO]:
    out = io.StringIO()
    return Console.scripted(inputs, out), out


# --------------------------------------------------------------------------- #
# Menu navigation
# --------------------------------------------------------------------------- #


async def test_menu_lists_agents_from_registry_and_quits(
    settings: Settings, fake_ollama: FakeOllama, registry: AgentRegistry
) -> None:
    console, out = _console("q")
    async with _runtime(settings, fake_ollama, registry) as runtime:
        code = await MainMenu(runtime, console).run()
    text = out.getvalue()
    assert code == 0
    assert "LOCAL AGENT STACK" in text
    assert "● Connected" in text and "Models: 5 detected" in text
    assert "Recommended:" in text and "Coding    → qwen2.5-coder:14b" in text
    assert "Which agent would you like to deploy?" in text
    assert " 1  Research Agent" in text and " 2  Web Agent" in text
    assert "Software Development" in text and "Coding Agent" in text and "SQL Agent" in text
    total = len(registry.names())
    assert f"{total + 1:>2}  Agent Manager" in text
    assert f"{total + 2:>2}  Model Manager" in text
    assert f"{total + 3:>2}  Create Agent" in text
    assert "q  Quit" in text and "Bye." in text


async def test_menu_reflects_custom_agent_without_core_changes(
    settings: Settings, fake_ollama: FakeOllama, registry: AgentRegistry
) -> None:
    from local_agents.agents.base import BaseAgent

    class ExtraAgent(BaseAgent):
        name = "extra"
        description = "An extra one"

    registry.register(ExtraAgent)
    console, out = _console("q")
    async with _runtime(settings, fake_ollama, registry) as runtime:
        await MainMenu(runtime, console).run()
    text = out.getvalue()
    total = len(registry.names())
    assert f"{total:>2}  Extra Agent" in text and f"{total + 1:>2}  Agent Manager" in text
    assert "  General\n" in text  # default category heading for the custom agent


async def test_menu_handles_ollama_down(settings: Settings, registry: AgentRegistry) -> None:
    fake = FakeOllama()
    fake.fail_with = httpx.ConnectError("refused")
    console, out = _console("q")
    async with _runtime(settings, fake, registry) as runtime:
        await MainMenu(runtime, console).run()
    text = out.getvalue()
    assert "○ Not available" in text
    assert "Ollama is not available." in text
    assert "Detected endpoint: http://ollama.test:11434" in text or "Detected endpoint:" in text


async def test_menu_invalid_choice_then_quit(
    settings: Settings, fake_ollama: FakeOllama, registry: AgentRegistry
) -> None:
    console, out = _console("zzz", "q")
    async with _runtime(settings, fake_ollama, registry) as runtime:
        await MainMenu(runtime, console).run()
    assert "Invalid selection" in out.getvalue()


async def test_menu_deploys_agent_and_streams_answer(
    settings: Settings, fake_ollama: FakeOllama, registry: AgentRegistry
) -> None:
    fake_ollama.queue("Photosynthesis converts light into chemical energy.")
    console, out = _console("1", "what is photosynthesis?", "/back", "q")
    async with _runtime(settings, fake_ollama, registry) as runtime:
        await MainMenu(runtime, console).run()
    text = out.getvalue()
    assert "Research Agent" in text
    assert "Model: qwen3:8b" in text or "Model: llama3.1:8b" in text
    assert "Photosynthesis converts light into chemical energy." in text


async def test_menu_model_manager_refresh_info_and_default(
    settings: Settings, fake_ollama: FakeOllama, registry: AgentRegistry
) -> None:
    models_key = str(len(registry.names()) + 2)
    console, out = _console(models_key, "r", "i", "1", "2", "x", "b", "q")
    async with _runtime(settings, fake_ollama, registry) as runtime:
        await MainMenu(runtime, console).run()
        assert runtime.settings.default_model == "qwen2.5-coder:14b"
    text = out.getvalue()
    assert "MODEL MANAGER" in text and "Installed Ollama Models" in text
    assert "Refreshed: 5 models detected" in text
    assert "Capabilities:  completion, tools, thinking" in text
    assert "Session default model set to qwen2.5-coder:14b" in text
    assert "Invalid selection" in text
    tags_calls = [r for r in fake_ollama.requests if r.url.path == "/api/tags"]
    assert len(tags_calls) == 2  # initial + refresh; other screens used the cache


async def test_menu_agent_manager_toggle_and_model(
    settings: Settings, fake_ollama: FakeOllama, registry: AgentRegistry
) -> None:
    agents_key = str(len(registry.names()) + 1)
    console, out = _console(
        agents_key, "2", "e", "m", "ghost", "m", "llama3.1:8b", "s", "b", "b", "q"
    )
    async with _runtime(settings, fake_ollama, registry) as runtime:
        await MainMenu(runtime, console).run()
        assert runtime.settings.agent("coding").enabled is False
        assert runtime.settings.agent("coding").model == "llama3.1:8b"
    text = out.getvalue()
    assert "AGENT MANAGER" in text
    assert "coding disabled" in text
    assert "Model 'ghost' is not installed" in text
    assert "coding will use llama3.1:8b" in text
    assert "Auto pick:    qwen2.5-coder:14b" in text
    assert "careful senior software engineer" in text


async def test_menu_create_agent(
    settings: Settings,
    fake_ollama: FakeOllama,
    registry: AgentRegistry,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.chdir(tmp_path)
    create_key = str(len(registry.names()) + 3)
    console, out = _console(create_key, "My Agent!", "Does things", "q")
    async with _runtime(settings, fake_ollama, registry) as runtime:
        await MainMenu(runtime, console).run()
        assert "my-agent" in runtime.registry
    text = out.getvalue()
    assert "Created agent 'my-agent'" in text
    assert "'my-agent' is now available in the menu." in text
    assert (tmp_path / "agents" / "my_agent" / "agent.py").is_file()


# --------------------------------------------------------------------------- #
# Session commands
# --------------------------------------------------------------------------- #


async def test_session_commands(
    settings: Settings, fake_ollama: FakeOllama, registry: AgentRegistry
) -> None:
    fake_ollama.queue("first answer")
    console, out = _console(
        "/help",
        "/tools",
        "/info",
        "/models",
        "/model",
        "/model ghost:1b",
        "/model llama3.1:8b",
        "/think",
        "/reset",
        "/bogus",
        "hello",
        "/back",
    )
    async with _runtime(settings, fake_ollama, registry) as runtime:
        agent = await runtime.create_agent("coding")
        session = AgentSession(runtime, agent, console)
        await session.interactive()
        assert agent.model.name == "llama3.1:8b"
    text = out.getvalue()
    assert "Commands:" in text
    assert "read_file" in text and '"can_delegate": false' in text
    assert "● qwen3:8b" in text
    assert "Current model: qwen2.5-coder:14b" in text
    assert "Model 'ghost:1b' is not installed" in text
    assert "Switched to llama3.1:8b" in text
    assert "Reasoning display on" in text
    assert "Conversation memory cleared" in text
    assert "Unknown command /bogus" in text
    assert "first answer" in text


async def test_session_renders_tool_events_and_blocked_calls(
    settings: Settings, fake_ollama: FakeOllama, registry: AgentRegistry
) -> None:
    console, out = _console()
    async with _runtime(settings, fake_ollama, registry) as runtime:
        model = ScriptedModel(
            [
                ScriptedTurn.call("read_file", path="../secret"),
                ScriptedTurn.call("run_command", command="rm -rf /"),
                ScriptedTurn.call("read_file", path="README.md"),
                "All done.",
            ]
        )
        (Path(settings.workspace.root) / "README.md").write_text("# hi\n")
        agent = await runtime.create_agent("coding", model=model)
        final = await AgentSession(runtime, agent, console).run_task("go")
    text = out.getvalue()
    assert final == "All done."
    assert "⚙ read_file(path='../secret')" in text
    assert "⛔ blocked" in text
    assert "run_command(command='rm -rf /')" in text
    assert "↳ README.md (lines 1-1 of 1)" in text


async def test_session_reports_model_errors_gracefully(
    settings: Settings, registry: AgentRegistry
) -> None:
    fake = FakeOllama({"models": []})
    console, out = _console()
    async with _runtime(settings, fake, registry) as runtime:
        agent = await runtime.create_agent("research")
        final = await AgentSession(runtime, agent, console).run_task("hi")
    assert final == ""
    assert "no models installed" in out.getvalue()


# --------------------------------------------------------------------------- #
# Error rendering
# --------------------------------------------------------------------------- #


def test_report_error_messages() -> None:
    console, out = _console()
    report_error(console, ProviderConnectionError("http://localhost:11434"))
    report_error(console, NoModelsAvailableError("http://x"))
    report_error(console, ModelNotFoundError("ghost:1b", ["a:1", "b:2"]))
    report_error(console, ToolError("tool broke", hint="fix it"))
    report_error(console, RuntimeError("kaboom"))
    text = out.getvalue()
    assert "✗ Could not connect to Ollama" in text and "Endpoint:\n  http://localhost:11434" in text
    assert "Start Ollama" in text
    assert "no models installed" in text and "ollama pull" in text
    assert "● a:1" in text and "ollama pull ghost:1b" in text
    assert "tool broke" in text and "Hint:\n  fix it" in text
    assert "Unexpected error: RuntimeError: kaboom" in text and "--debug" in text
    console, out = _console()
    try:
        raise RuntimeError("kaboom")
    except RuntimeError as exc:
        report_error(console, exc, debug=True)
    assert "Traceback" in out.getvalue()


# --------------------------------------------------------------------------- #
# Argument parsing and subcommands through main()
# --------------------------------------------------------------------------- #


def test_parser_flags() -> None:
    parser = cli_main.build_parser()
    args = parser.parse_args(
        ["--debug", "--host", "http://x:1", "run", "coding", "fix", "it", "--json"]
    )
    assert args.debug and args.host == "http://x:1" and args.command == "run"
    assert args.agent == "coding" and args.task == ["fix", "it"] and args.json


def test_main_config_and_version(capsys: pytest.CaptureFixture[str], tmp_path: Path) -> None:
    assert cli_main.main(["--no-color", "--workspace", str(tmp_path), "config"]) == 0
    data = json.loads(capsys.readouterr().out.split("\n#")[0])
    assert data["workspace"]["root"] == str(tmp_path)
    with pytest.raises(SystemExit) as info:
        cli_main.main(["--version"])
    assert info.value.code == 0


def test_main_create_agent_and_list(
    capsys: pytest.CaptureFixture[str], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        "local_agents.agents.runtime.OllamaProvider.from_settings",
        lambda s, transport=None: OllamaProvider(s.host, transport=FakeOllama().transport()),
    )
    assert (
        cli_main.main(
            [
                "--no-color",
                "create-agent",
                "Report Writer",
                "--description",
                "Writes reports",
                "--tools",
                "filesystem,web",
            ]
        )
        == 0
    )
    out = capsys.readouterr().out
    assert "Created agent 'report-writer'" in out
    assert (tmp_path / "agents" / "report_writer" / "tests" / "test_agent.py").is_file()
    # Re-creating without --force fails cleanly.
    assert cli_main.main(["--no-color", "create-agent", "report-writer"]) == 1
    assert "already exists" in capsys.readouterr().out
    assert cli_main.main(["--no-color", "create-agent", "x", "--tools", "teleport"]) == 1
    assert "Unknown tool group" in capsys.readouterr().out
    assert cli_main.main(["--no-color", "list"]) == 0
    out = capsys.readouterr().out
    assert "● research" in out and "● report-writer" in out


def test_main_models_and_run_with_fake_ollama(
    capsys: pytest.CaptureFixture[str], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake = FakeOllama()
    fake.queue("Hmm?")  # research is tool-first: first tool-less reply gets nudged
    fake.queue("Answer: 4")

    monkeypatch.setattr(
        "local_agents.agents.runtime.OllamaProvider.from_settings",
        lambda s, transport=None: OllamaProvider(s.host, transport=fake.transport()),
    )
    monkeypatch.chdir(tmp_path)
    assert cli_main.main(["--no-color", "models"]) == 0
    out = capsys.readouterr().out
    assert "● qwen3:8b" in out and "Coding    → qwen2.5-coder:14b" in out
    assert cli_main.main(["--no-color", "models", "--json"]) == 0
    data = json.loads(capsys.readouterr().out)
    assert data["recommended"]["coding"] == "qwen2.5-coder:14b"
    assert cli_main.main(["--no-color", "run", "research", "what", "is", "2+2", "--quiet"]) == 0
    assert capsys.readouterr().out.strip() == "Answer: 4"
    fake.queue("Hmm?")
    fake.queue("Answer: 5")
    assert cli_main.main(["--no-color", "run", "research", "again", "--json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["output"] == "Answer: 5" and payload["agent"] == "research"
    fake.queue("Hmm?")
    fake.queue("streamed")
    assert cli_main.main(["--no-color", "run", "research", "again"]) == 0
    assert "streamed" in capsys.readouterr().out


def test_main_ollama_down_is_friendly(
    capsys: pytest.CaptureFixture[str], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake = FakeOllama()
    fake.fail_with = httpx.ConnectError("refused")

    monkeypatch.setattr(
        "local_agents.agents.runtime.OllamaProvider.from_settings",
        lambda s, transport=None: OllamaProvider(s.host, transport=fake.transport()),
    )
    monkeypatch.chdir(tmp_path)
    assert cli_main.main(["--no-color", "models"]) == 1
    out = capsys.readouterr().out
    assert "✗ Could not connect to Ollama" in out and "Traceback" not in out
    assert cli_main.main(["--no-color", "run", "research", "hi", "--quiet"]) == 1
    out = capsys.readouterr().out
    assert "Could not connect to Ollama" in out and "Traceback" not in out
    assert cli_main.main(["--no-color", "--debug", "run", "research", "hi", "--quiet"]) == 1
    assert "Traceback" in capsys.readouterr().out


def test_main_bad_config_path(capsys: pytest.CaptureFixture[str], tmp_path: Path) -> None:
    assert cli_main.main(["--no-color", "--config", str(tmp_path / "missing.yaml"), "list"]) == 1
    assert "Config file not found" in capsys.readouterr().out


def test_ui_helpers() -> None:
    assert format_size(512) == "512 B"
    assert format_size(5_300_000_000) == "4.9 GB"
    assert truncate("a" * 100, 10) == "a" * 9 + "…"
    console, _ = _console("y", "", "n")
    assert console.confirm("ok?") is True
    assert console.confirm("ok?", default=True) is True
    assert console.confirm("ok?") is False
    assert console.prompt("x") is None  # EOF
    console.style.enabled = True
    assert "\x1b[1m" in console.style.bold("t")
