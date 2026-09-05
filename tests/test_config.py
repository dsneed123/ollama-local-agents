from __future__ import annotations

from pathlib import Path

import pytest

from local_agents.config.settings import (
    AgentSettings,
    Settings,
    find_config_file,
    load_agent_config,
    load_settings,
    normalize_host,
)
from local_agents.errors import ConfigError


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("", "http://localhost:11434"),
        ("localhost", "http://localhost:11434"),
        ("0.0.0.0", "http://localhost:11434"),
        ("0.0.0.0:11434", "http://localhost:11434"),
        ("127.0.0.1:9999", "http://127.0.0.1:9999"),
        ("http://gpu-box:11434/", "http://gpu-box:11434"),
        ("https://ollama.example.com", "https://ollama.example.com:11434"),
        ("[::1]:11434", "http://[::1]:11434"),
    ],
)
def test_normalize_host(raw: str, expected: str) -> None:
    assert normalize_host(raw) == expected


def test_defaults_need_no_config(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    settings = load_settings(env={})
    assert settings.ollama.host == "http://localhost:11434"
    assert settings.workspace.allow_delete is False
    assert settings.default_model == "auto"
    assert settings.config_path is None


def test_yaml_file_is_loaded(tmp_path: Path) -> None:
    cfg = tmp_path / "local-agents.yaml"
    cfg.write_text(
        """
ollama:
  host: http://gpu:11434
  timeout: 30
agents:
  research:
    model: qwen3:8b
  coding:
    enabled: false
workspace:
  allow_delete: true
agent_paths: ["./my-agents"]
"""
    )
    settings = load_settings(cfg, env={})
    assert settings.ollama.host == "http://gpu:11434"
    assert settings.ollama.timeout == 30
    assert settings.agent("research").model == "qwen3:8b"
    assert settings.agent("coding").enabled is False
    assert settings.workspace.allow_delete is True
    assert settings.agent_paths == ["./my-agents"]
    assert settings.config_path == cfg


def test_env_overrides_file(tmp_path: Path) -> None:
    cfg = tmp_path / "c.yaml"
    cfg.write_text("ollama:\n  host: http://from-file:11434\n")
    settings = load_settings(
        cfg,
        env={
            "OLLAMA_HOST": "0.0.0.0:12345",
            "LOCAL_AGENTS_WORKSPACE": str(tmp_path),
            "LOCAL_AGENTS_DEBUG": "true",
            "LOCAL_AGENTS_MODEL": "llama3.1:8b",
            "LOCAL_AGENTS_PATH": "/a:/b",
        },
    )
    assert settings.ollama.host == "http://localhost:12345"
    assert settings.workspace.resolved_root() == tmp_path.resolve()
    assert settings.debug is True
    assert settings.default_model == "llama3.1:8b"
    assert settings.agent_paths == ["/a", "/b"]


def test_overrides_win_over_everything(tmp_path: Path) -> None:
    settings = load_settings(
        None,
        env={"OLLAMA_HOST": "http://env:11434"},
        overrides={"ollama": {"host": "http://cli:1"}},
    )
    assert settings.ollama.host == "http://cli:1"


def test_env_config_path(tmp_path: Path) -> None:
    cfg = tmp_path / "custom.yaml"
    cfg.write_text("default_model: qwen3:8b\n")
    settings = load_settings(env={"LOCAL_AGENTS_CONFIG": str(cfg)})
    assert settings.default_model == "qwen3:8b"


def test_missing_explicit_config_is_an_error(tmp_path: Path) -> None:
    with pytest.raises(ConfigError):
        find_config_file(tmp_path / "nope.yaml")


def test_unknown_keys_are_reported(tmp_path: Path) -> None:
    cfg = tmp_path / "c.yaml"
    cfg.write_text("ollama:\n  hostname: x\n")
    with pytest.raises(ConfigError, match="Unknown option"):
        load_settings(cfg, env={})


def test_invalid_yaml(tmp_path: Path) -> None:
    cfg = tmp_path / "c.yaml"
    cfg.write_text("ollama: [unclosed\n")
    with pytest.raises(ConfigError, match="Invalid YAML"):
        load_settings(cfg, env={})


def test_top_level_must_be_mapping(tmp_path: Path) -> None:
    cfg = tmp_path / "c.yaml"
    cfg.write_text("- just\n- a list\n")
    with pytest.raises(ConfigError):
        load_settings(cfg, env={})


def test_to_dict_roundtrip_shape() -> None:
    settings = Settings(agents={"x": AgentSettings(model="m")})
    data = settings.to_dict()
    assert data["ollama"]["host"] == "http://localhost:11434"
    assert data["agents"]["x"]["model"] == "m"


def test_load_agent_config(tmp_path: Path) -> None:
    cfg = tmp_path / "config.yaml"
    cfg.write_text("agent:\n  model: qwen3:8b\n  max_iterations: 3\n  name: ignored\n")
    agent_cfg = load_agent_config(cfg)
    assert agent_cfg.model == "qwen3:8b"
    assert agent_cfg.max_iterations == 3
    flat = tmp_path / "flat.yaml"
    flat.write_text("model: llama3.1:8b\n")
    assert load_agent_config(flat).model == "llama3.1:8b"
