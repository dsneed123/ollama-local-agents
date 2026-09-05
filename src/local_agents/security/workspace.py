"""Workspace boundary enforcement.

Every filesystem tool resolves paths through a :class:`Workspace`, which
guarantees that reads and writes stay inside the configured root (symlinks
included) and that destructive operations are opt-in.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from local_agents.config.settings import WorkspaceSettings
from local_agents.errors import PermissionDeniedError, WorkspaceViolationError

DEFAULT_IGNORED_DIRS = frozenset(
    {
        ".git",
        ".hg",
        ".svn",
        "__pycache__",
        ".venv",
        "venv",
        "node_modules",
        ".mypy_cache",
        ".pytest_cache",
        ".ruff_cache",
        ".tox",
        "dist",
        "build",
        ".idea",
        ".vscode",
    }
)


@dataclass
class Workspace:
    """A directory the agent may operate in.

    Attributes:
        root: Absolute directory all paths are confined to.
        allow_write: Whether files may be created or modified.
        allow_delete: Whether files may be removed (off by default).
        max_file_bytes: Largest file a tool will read in one go.
        ignored_dirs: Directory names skipped during searches.
    """

    root: Path
    allow_write: bool = True
    allow_delete: bool = False
    max_file_bytes: int = 2_000_000
    ignored_dirs: frozenset[str] = field(default_factory=lambda: DEFAULT_IGNORED_DIRS)

    def __post_init__(self) -> None:
        self.root = Path(self.root).expanduser().resolve()
        if not self.root.is_dir():
            raise ValueError(f"Workspace root does not exist or is not a directory: {self.root}")

    @classmethod
    def from_settings(cls, settings: WorkspaceSettings) -> Workspace:
        return cls(
            root=settings.resolved_root(),
            allow_write=settings.allow_write,
            allow_delete=settings.allow_delete,
            max_file_bytes=settings.max_file_bytes,
        )

    @classmethod
    def cwd(cls, **kwargs: object) -> Workspace:
        return cls(root=Path.cwd(), **kwargs)  # type: ignore[arg-type]

    # ------------------------------------------------------------------ #

    def resolve(self, path: str | os.PathLike[str], *, must_exist: bool = False) -> Path:
        """Return an absolute path inside the workspace or raise.

        Relative paths are interpreted relative to the root. Symlinks are
        followed so a link pointing outside the root is rejected.
        """
        raw = Path(path).expanduser()
        candidate = raw if raw.is_absolute() else self.root / raw
        # Resolve the deepest existing ancestor so brand-new files are validated too.
        resolved = self._resolve_lenient(candidate)
        if not self.contains(resolved):
            raise WorkspaceViolationError(str(path), str(self.root))
        if must_exist and not resolved.exists():
            raise FileNotFoundError(f"No such file or directory: {self.relative(resolved)}")
        return resolved

    def contains(self, path: Path) -> bool:
        try:
            path.relative_to(self.root)
        except ValueError:
            return False
        return True

    def relative(self, path: Path) -> str:
        try:
            return str(path.relative_to(self.root)) or "."
        except ValueError:
            return str(path)

    def check_write(self, path: str | os.PathLike[str]) -> Path:
        if not self.allow_write:
            raise PermissionDeniedError(
                "Writing is disabled for this workspace",
                hint="Set `workspace.allow_write: true` or pass allow_write=True.",
            )
        return self.resolve(path)

    def check_delete(self, path: str | os.PathLike[str]) -> Path:
        if not self.allow_delete:
            raise PermissionDeniedError(
                "Deleting is disabled for this workspace (default)",
                hint="Set `workspace.allow_delete: true` to opt in.",
            )
        return self.resolve(path, must_exist=True)

    def is_ignored(self, path: Path) -> bool:
        return any(part in self.ignored_dirs for part in path.parts)

    def describe(self) -> str:
        flags = ["read"]
        if self.allow_write:
            flags.append("write")
        if self.allow_delete:
            flags.append("delete")
        return f"{self.root} ({', '.join(flags)})"

    # ------------------------------------------------------------------ #

    @staticmethod
    def _resolve_lenient(path: Path) -> Path:
        """``Path.resolve()`` that tolerates non-existent tails."""
        try:
            return path.resolve(strict=True)
        except FileNotFoundError:
            pass
        except OSError:
            pass
        existing = path
        tail: list[str] = []
        while not existing.exists() and existing != existing.parent:
            tail.append(existing.name)
            existing = existing.parent
        base = existing.resolve()
        for part in reversed(tail):
            base = base / part
        return base
