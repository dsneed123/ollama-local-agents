"""Agent registry and discovery.

Agents register themselves with the :func:`register_agent` decorator. The CLI
menu, the runtime and the delegation tool all read from the registry, so adding
an agent never requires editing core code.

Discovery sources, in order:

1. Built-in reference agents (``local_agents.agents.builtin``).
2. Python entry points in the ``local_agents.agents`` group (pip-installed plugins).
3. Directories of user agents: ``./agents/<name>/agent.py`` or ``./agents/<name>.py``.
"""

from __future__ import annotations

import importlib
import importlib.machinery
import importlib.util
import logging
import os
import sys
from collections.abc import Callable, Iterable, Iterator
from dataclasses import dataclass, field
from importlib.metadata import entry_points
from pathlib import Path
from typing import TypeVar

from local_agents.agents.base import BaseAgent
from local_agents.errors import AgentNotFoundError, AgentRegistrationError

log = logging.getLogger(__name__)

ENTRY_POINT_GROUP = "local_agents.agents"
DEFAULT_AGENT_DIR = "agents"

A = TypeVar("A", bound=type[BaseAgent])


@dataclass
class AgentRegistration:
    name: str
    description: str
    cls: type[BaseAgent]
    source: str = "code"
    order: int = 0
    hidden: bool = False
    config_path: Path | None = None


@dataclass
class DiscoveryReport:
    """What :meth:`AgentRegistry.discover` found (and failed to load)."""

    loaded: list[str] = field(default_factory=list)
    errors: list[tuple[str, str]] = field(default_factory=list)


class AgentRegistry:
    """Name -> agent class mapping with pluggable discovery."""

    def __init__(self) -> None:
        self._agents: dict[str, AgentRegistration] = {}
        self._counter = 0
        self._discovered = False
        self._loaded_paths: set[Path] = set()

    # ------------------------------------------------------------------ #
    # Registration
    # ------------------------------------------------------------------ #

    def register(
        self,
        cls: type[BaseAgent],
        *,
        name: str | None = None,
        description: str | None = None,
        source: str = "code",
        replace: bool = False,
        hidden: bool = False,
    ) -> type[BaseAgent]:
        if not isinstance(cls, type) or not issubclass(cls, BaseAgent):
            raise AgentRegistrationError(f"{cls!r} is not a BaseAgent subclass")
        agent_name = (name or cls.name or _default_name(cls)).strip().lower()
        if not agent_name:
            raise AgentRegistrationError(f"Agent {cls.__name__} needs a name")
        if agent_name in self._agents and not replace:
            existing = self._agents[agent_name].cls
            if existing is cls:
                return cls
            raise AgentRegistrationError(
                f"An agent named '{agent_name}' is already registered ({existing.__name__})",
                hint="Pass replace=True or choose another name.",
            )
        cls.name = agent_name
        cls.description = description or cls.description or _doc_summary(cls)
        self._counter += 1
        self._agents[agent_name] = AgentRegistration(
            name=agent_name,
            description=cls.description,
            cls=cls,
            source=source,
            order=self._counter,
            hidden=hidden,
        )
        log.debug("Registered agent %s (%s)", agent_name, source)
        return cls

    def unregister(self, name: str) -> None:
        self._agents.pop(name.lower(), None)

    def clear(self) -> None:
        self._agents.clear()
        self._discovered = False
        self._loaded_paths.clear()

    # ------------------------------------------------------------------ #
    # Lookup
    # ------------------------------------------------------------------ #

    def get(self, name: str) -> type[BaseAgent]:
        try:
            return self._agents[name.lower()].cls
        except KeyError:
            raise AgentNotFoundError(name, self.names()) from None

    def registration(self, name: str) -> AgentRegistration:
        try:
            return self._agents[name.lower()]
        except KeyError:
            raise AgentNotFoundError(name, self.names()) from None

    def list_agents(self, *, include_hidden: bool = False) -> list[AgentRegistration]:
        regs = sorted(self._agents.values(), key=lambda r: r.order)
        return [r for r in regs if include_hidden or not r.hidden]

    def names(self) -> list[str]:
        return [r.name for r in self.list_agents()]

    def __contains__(self, name: object) -> bool:
        return isinstance(name, str) and name.lower() in self._agents

    def __iter__(self) -> Iterator[AgentRegistration]:
        return iter(self.list_agents())

    def __len__(self) -> int:
        return len(self._agents)

    # ------------------------------------------------------------------ #
    # Discovery
    # ------------------------------------------------------------------ #

    @property
    def discovered(self) -> bool:
        return self._discovered

    def discover(
        self,
        *,
        builtins: bool = True,
        plugins: bool = True,
        paths: Iterable[str | os.PathLike[str]] = (),
        include_default_dir: bool = True,
    ) -> DiscoveryReport:
        """Load agents from all sources. Safe to call repeatedly."""
        report = DiscoveryReport()
        if builtins:
            self._load_builtins(report)
        if plugins and not self._discovered:
            self._load_entry_points(report)
        search: list[Path] = [Path(p).expanduser() for p in paths]
        if include_default_dir:
            search.append(Path.cwd() / DEFAULT_AGENT_DIR)
        for directory in search:
            if directory.is_dir():
                self.load_directory(directory, report)
        self._discovered = True
        return report

    def load_directory(
        self, directory: Path, report: DiscoveryReport | None = None
    ) -> DiscoveryReport:
        """Import every ``<dir>/<name>/agent.py`` and ``<dir>/<name>.py``."""
        report = report or DiscoveryReport()
        directory = directory.resolve()
        candidates: list[Path] = []
        for child in sorted(directory.iterdir()):
            if child.name.startswith((".", "_")):
                continue
            if child.is_dir() and (child / "agent.py").is_file():
                candidates.append(child / "agent.py")
            elif child.is_file() and child.suffix == ".py":
                candidates.append(child)
        for module_path in candidates:
            if module_path in self._loaded_paths:
                continue
            try:
                self.load_file(module_path)
            except Exception as exc:
                log.warning("Could not load agent module %s: %s", module_path, exc)
                report.errors.append((str(module_path), f"{type(exc).__name__}: {exc}"))
            else:
                report.loaded.append(str(module_path))
                self._loaded_paths.add(module_path)
        return report

    def load_file(self, path: Path) -> None:
        """Import a single agent module from ``path``.

        Modules are imported under ``local_agents_user.<name>`` so relative
        imports of sibling files (``from .tools import ...``) work when the
        agent lives in its own directory.
        """
        path = path.resolve()
        _ensure_namespace_package("local_agents_user")
        if path.name == "agent.py":
            package_dir = path.parent
            package_name = f"local_agents_user.{_module_safe(package_dir.name)}"
            _ensure_package(package_name, package_dir)
            module_name = f"{package_name}.agent"
        else:
            module_name = f"local_agents_user.{_module_safe(path.stem)}"
        spec = importlib.util.spec_from_file_location(module_name, path)
        if spec is None or spec.loader is None:
            raise ImportError(f"Cannot import {path}")
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        before = set(self._agents)
        with _temporary_registry(self):
            spec.loader.exec_module(module)
        config_path = path.parent / "config.yaml"
        for name in set(self._agents) - before:
            self._agents[name].source = str(path)
            if path.name == "agent.py" and config_path.is_file():
                self._agents[name].config_path = config_path

    def _load_builtins(self, report: DiscoveryReport) -> None:
        """Register the reference agents (idempotent, works for any registry instance)."""
        module = importlib.import_module("local_agents.agents.builtin")
        for cls in module.BUILTIN_AGENTS:
            if cls.name not in self._agents:
                self.register(cls, source="builtin")
        report.loaded.append("local_agents.agents.builtin")

    def _load_entry_points(self, report: DiscoveryReport) -> None:
        for ep in entry_points(group=ENTRY_POINT_GROUP):
            try:
                with _temporary_registry(self):
                    ep.load()
                report.loaded.append(f"entry-point:{ep.name}")
            except Exception as exc:
                log.warning("Could not load agent plugin %s: %s", ep.name, exc)
                report.errors.append((f"entry-point:{ep.name}", f"{type(exc).__name__}: {exc}"))


# --------------------------------------------------------------------------- #
# Module-level default registry + decorator
# --------------------------------------------------------------------------- #

_default_registry = AgentRegistry()
_active_registry: AgentRegistry | None = None


def get_registry() -> AgentRegistry:
    """The registry :func:`register_agent` writes to (default unless overridden)."""
    return _active_registry if _active_registry is not None else _default_registry


def register_agent(
    name: str | None = None,
    description: str | None = None,
    *,
    registry: AgentRegistry | None = None,
    hidden: bool = False,
    replace: bool = False,
) -> Callable[[A], A]:
    """Class decorator that registers a :class:`BaseAgent` subclass.

    Usage::

        @register_agent(name="research", description="Research and synthesize information")
        class ResearchAgent(BaseAgent):
            ...
    """

    def decorator(cls: A) -> A:
        target = registry if registry is not None else get_registry()
        target.register(cls, name=name, description=description, hidden=hidden, replace=replace)
        return cls

    return decorator


class _temporary_registry:
    """Route ``register_agent`` calls to ``registry`` while importing a module."""

    def __init__(self, registry: AgentRegistry) -> None:
        self.registry = registry
        self.previous: AgentRegistry | None = None

    def __enter__(self) -> None:
        global _active_registry
        self.previous = _active_registry
        _active_registry = self.registry

    def __exit__(self, *exc: object) -> None:
        global _active_registry
        _active_registry = self.previous


def _default_name(cls: type[BaseAgent]) -> str:
    name = cls.__name__
    if name.endswith("Agent"):
        name = name[: -len("Agent")]
    return "".join(f"-{c.lower()}" if c.isupper() else c for c in name).lstrip("-")


def _module_safe(text: str) -> str:
    safe = "".join(c if c.isalnum() or c == "_" else "_" for c in text)
    return safe if safe and not safe[0].isdigit() else f"_{safe}"


def _ensure_namespace_package(name: str) -> None:
    if name in sys.modules:
        return
    spec = importlib.machinery.ModuleSpec(name, None, is_package=True)
    module = importlib.util.module_from_spec(spec)
    module.__path__ = []
    sys.modules[name] = module


def _ensure_package(name: str, directory: Path) -> None:
    """Register ``directory`` as package ``name`` so ``from .tools import X`` works.

    An ``__init__.py`` is executed when present; otherwise an empty package
    module is synthesised.
    """
    if name in sys.modules:
        return
    init = directory / "__init__.py"
    if init.is_file():
        spec = importlib.util.spec_from_file_location(
            name, init, submodule_search_locations=[str(directory)]
        )
        if spec is None or spec.loader is None:
            raise ImportError(f"Cannot create package spec for {directory}")
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
        return
    spec = importlib.machinery.ModuleSpec(name, None, is_package=True)
    module = importlib.util.module_from_spec(spec)
    module.__path__ = [str(directory)]
    sys.modules[name] = module


def _doc_summary(cls: type) -> str:
    doc = (cls.__doc__ or "").strip()
    return doc.splitlines()[0] if doc else ""
