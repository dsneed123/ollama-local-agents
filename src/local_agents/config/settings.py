"""Typed settings with layered resolution.

Resolution order (later layers win):

1. Built-in defaults (everything works with zero configuration).
2. A YAML config file, found via ``--config``, ``LOCAL_AGENTS_CONFIG``, or
   the first of :data:`DEFAULT_CONFIG_FILENAMES` in the current directory or
   ``~/.config/local-agents/``.
3. Environment variables (``OLLAMA_HOST``, ``LOCAL_AGENTS_WORKSPACE``, ...).
4. Explicit overrides passed by the caller (CLI flags).

Settings are plain dataclasses so they stay dependency-free and trivially
testable.
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any

import yaml

from local_agents.errors import ConfigError

DEFAULT_OLLAMA_HOST = "http://localhost:11434"
DEFAULT_CONFIG_FILENAMES = ("local-agents.yaml", "local-agents.yml", ".local-agents.yaml")

ENV_CONFIG = "LOCAL_AGENTS_CONFIG"
ENV_OLLAMA_HOST = "OLLAMA_HOST"
ENV_WORKSPACE = "LOCAL_AGENTS_WORKSPACE"
ENV_DEBUG = "LOCAL_AGENTS_DEBUG"
ENV_AGENT_PATH = "LOCAL_AGENTS_PATH"
ENV_DEFAULT_MODEL = "LOCAL_AGENTS_MODEL"


def normalize_host(raw: str) -> str:
    """Turn any of the forms Ollama accepts for ``OLLAMA_HOST`` into a base URL.

    Ollama itself accepts ``0.0.0.0``, ``127.0.0.1:11434``, ``http://host:port``
    and ``https://host``. We normalise to ``scheme://host:port`` with no trailing
    slash so it can be used directly as an HTTP base URL.
    """
    value = raw.strip().rstrip("/")
    if not value:
        return DEFAULT_OLLAMA_HOST
    if "://" not in value:
        value = "http://" + value
    scheme, _, rest = value.partition("://")
    host, _, path = rest.partition("/")
    hostname, port = _split_host_port(host)
    if hostname in ("0.0.0.0", ""):
        hostname = "localhost"
    host = f"{hostname}:{port or 11434}"
    url = f"{scheme}://{host}"
    if path:
        url = f"{url}/{path}"
    return url


def _split_host_port(host: str) -> tuple[str, str]:
    """Split ``host[:port]`` (IPv6 in brackets supported)."""
    if host.startswith("["):
        end = host.find("]")
        if end != -1:
            return host[: end + 1], host[end + 2 :] if host[end + 1 : end + 2] == ":" else ""
    if host.count(":") == 1:
        name, _, port = host.partition(":")
        return name, port
    return host, ""


@dataclass
class OllamaSettings:
    """Connection settings for the Ollama provider."""

    host: str = DEFAULT_OLLAMA_HOST
    auto_detect: bool = True
    timeout: float = 120.0
    connect_timeout: float = 5.0
    keep_alive: str | None = None

    def __post_init__(self) -> None:
        self.host = normalize_host(self.host)


@dataclass
class AgentSettings:
    """Per-agent overrides. ``model="auto"`` triggers automatic selection."""

    enabled: bool = True
    model: str = "auto"
    temperature: float | None = None
    max_iterations: int | None = None
    context_length: int | None = None


@dataclass
class WorkspaceSettings:
    """Filesystem boundary applied to file and shell tools."""

    root: str = "."
    allow_write: bool = True
    allow_delete: bool = False
    max_file_bytes: int = 2_000_000

    def resolved_root(self) -> Path:
        return Path(self.root).expanduser().resolve()


@dataclass
class SecuritySettings:
    """Guard rails for shell execution and network access."""

    allow_shell: bool = True
    shell_timeout: float = 60.0
    confirm_shell: bool = True
    confirm_writes: bool = False
    allow_network: bool = True
    blocked_commands: list[str] = field(default_factory=list)
    allowed_commands: list[str] = field(default_factory=list)


@dataclass
class Settings:
    """Top-level settings object."""

    ollama: OllamaSettings = field(default_factory=OllamaSettings)
    workspace: WorkspaceSettings = field(default_factory=WorkspaceSettings)
    security: SecuritySettings = field(default_factory=SecuritySettings)
    agents: dict[str, AgentSettings] = field(default_factory=dict)
    agent_paths: list[str] = field(default_factory=list)
    default_model: str = "auto"
    debug: bool = False
    config_path: Path | None = None

    def agent(self, name: str) -> AgentSettings:
        """Return settings for ``name``, creating defaults on first access."""
        return self.agents.setdefault(name, AgentSettings())

    def to_dict(self) -> dict[str, Any]:
        """Serialise to a plain dict (useful for ``--show-config``)."""
        return {
            "ollama": _dataclass_to_dict(self.ollama),
            "workspace": _dataclass_to_dict(self.workspace),
            "security": _dataclass_to_dict(self.security),
            "agents": {k: _dataclass_to_dict(v) for k, v in self.agents.items()},
            "agent_paths": list(self.agent_paths),
            "default_model": self.default_model,
            "debug": self.debug,
        }


# --------------------------------------------------------------------------- #
# Loading
# --------------------------------------------------------------------------- #


def find_config_file(explicit: str | os.PathLike[str] | None = None) -> Path | None:
    """Locate a config file, or return ``None`` when running with defaults."""
    if explicit:
        path = Path(explicit).expanduser()
        if not path.is_file():
            raise ConfigError(
                f"Config file not found: {path}",
                hint="Check the --config path or unset LOCAL_AGENTS_CONFIG.",
            )
        return path
    env = os.environ.get(ENV_CONFIG)
    if env:
        return find_config_file(env)
    search_dirs = [Path.cwd(), Path.home() / ".config" / "local-agents"]
    for directory in search_dirs:
        for name in DEFAULT_CONFIG_FILENAMES:
            candidate = directory / name
            if candidate.is_file():
                return candidate
        candidate = directory / "config.yaml"
        if directory != Path.cwd() and candidate.is_file():
            return candidate
    return None


def load_settings(
    config_path: str | os.PathLike[str] | None = None,
    *,
    env: dict[str, str] | None = None,
    overrides: dict[str, Any] | None = None,
) -> Settings:
    """Build a :class:`Settings` from defaults, file, environment, and overrides.

    ``env`` defaults to ``os.environ``; tests pass an explicit dict.
    ``overrides`` uses the same nested shape as the YAML file, e.g.
    ``{"ollama": {"host": "..."}, "debug": True}``.
    """
    environ: Mapping[str, str] = os.environ if env is None else env
    data: dict[str, Any] = {}

    path = find_config_file(config_path or environ.get(ENV_CONFIG))
    if path is not None:
        data = _read_yaml(path)

    _apply_env(data, environ)
    if overrides:
        _deep_merge(data, overrides)

    settings = _build_settings(data)
    settings.config_path = path
    return settings


def _read_yaml(path: Path) -> dict[str, Any]:
    try:
        with path.open("r", encoding="utf-8") as fh:
            loaded = yaml.safe_load(fh) or {}
    except yaml.YAMLError as exc:
        raise ConfigError(f"Invalid YAML in {path}: {exc}") from exc
    except OSError as exc:
        raise ConfigError(f"Could not read config file {path}: {exc}") from exc
    if not isinstance(loaded, dict):
        raise ConfigError(f"Config file {path} must contain a mapping at the top level")
    return loaded


def _apply_env(data: dict[str, Any], env: Mapping[str, str]) -> None:
    if host := env.get(ENV_OLLAMA_HOST):
        data.setdefault("ollama", {})["host"] = host
    if workspace := env.get(ENV_WORKSPACE):
        data.setdefault("workspace", {})["root"] = workspace
    if model := env.get(ENV_DEFAULT_MODEL):
        data["default_model"] = model
    if debug := env.get(ENV_DEBUG):
        data["debug"] = debug.strip().lower() in ("1", "true", "yes", "on")
    if paths := env.get(ENV_AGENT_PATH):
        extra = [p for p in paths.split(os.pathsep) if p]
        data["agent_paths"] = [*data.get("agent_paths", []), *extra]


def _deep_merge(base: dict[str, Any], update: dict[str, Any]) -> None:
    for key, value in update.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            _deep_merge(base[key], value)
        elif value is not None:
            base[key] = value


def _build_settings(data: dict[str, Any]) -> Settings:
    try:
        ollama = _build(OllamaSettings, data.get("ollama") or {}, "ollama")
        workspace = _build(WorkspaceSettings, data.get("workspace") or {}, "workspace")
        security = _build(SecuritySettings, data.get("security") or {}, "security")
        agents_raw = data.get("agents") or {}
        if not isinstance(agents_raw, dict):
            raise ConfigError("`agents` must be a mapping of agent name -> settings")
        agents = {
            str(name): _build(AgentSettings, cfg or {}, f"agents.{name}")
            for name, cfg in agents_raw.items()
        }
        agent_paths = data.get("agent_paths") or []
        if isinstance(agent_paths, str):
            agent_paths = [agent_paths]
        return Settings(
            ollama=ollama,
            workspace=workspace,
            security=security,
            agents=agents,
            agent_paths=[str(p) for p in agent_paths],
            default_model=str(data.get("default_model", "auto")),
            debug=bool(data.get("debug", False)),
        )
    except TypeError as exc:
        raise ConfigError(f"Invalid configuration: {exc}") from exc


def _build(cls: type[Any], raw: Any, section: str) -> Any:
    if not isinstance(raw, dict):
        raise ConfigError(f"`{section}` must be a mapping")
    allowed = {f.name for f in fields(cls)}
    unknown = set(raw) - allowed
    if unknown:
        raise ConfigError(
            f"Unknown option(s) in `{section}`: {', '.join(sorted(unknown))}",
            hint=f"Valid options: {', '.join(sorted(allowed))}",
        )
    return cls(**raw)


def _dataclass_to_dict(obj: Any) -> dict[str, Any]:
    return {f.name: getattr(obj, f.name) for f in fields(obj)}


def load_agent_config(path: str | os.PathLike[str]) -> AgentSettings:
    """Load a per-agent ``config.yaml`` (as generated by ``create-agent``).

    Accepts either the bare :class:`AgentSettings` keys at the top level or an
    ``agent:`` mapping containing them. Unknown keys such as ``name`` and
    ``description`` are ignored so the file can double as documentation.
    """
    raw = _read_yaml(Path(path))
    nested = raw.get("agent")
    section: dict[str, Any] = nested if isinstance(nested, dict) else raw
    allowed = {f.name for f in fields(AgentSettings)}
    data = {k: v for k, v in section.items() if k in allowed}
    try:
        return AgentSettings(**data)
    except TypeError as exc:
        raise ConfigError(f"Invalid agent config {path}: {exc}") from exc
