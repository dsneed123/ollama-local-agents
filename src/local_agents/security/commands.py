"""Shell command policy.

Blocks obviously destructive commands before they reach a shell. This is a
guard rail against *accidents* by an autonomous agent, not a sandbox: agents
that need real isolation should run inside a container.

Two layers:

* ``blocked_patterns`` — regexes that always block (``rm -rf``, ``mkfs`` ...).
* ``allowed_prefixes`` — when non-empty, only commands starting with one of
  these are permitted (allow-list mode).
"""

from __future__ import annotations

import re
import shlex
from dataclasses import dataclass, field
from re import Pattern

from local_agents.config.settings import SecuritySettings
from local_agents.errors import CommandBlockedError

DEFAULT_BLOCKED_PATTERNS: tuple[tuple[str, str], ...] = (
    (r"\brm\s+(-[a-zA-Z]*[rRf][a-zA-Z]*\s+)+", "recursive/forced rm"),
    (r"\brm\s+.*\s-[a-zA-Z]*[rRf]", "recursive/forced rm"),
    (r"\brm\s+(/|~|\*|\.)(\s|$)", "rm of a root, home, or wildcard target"),
    (r"\brmdir\b", "directory removal"),
    (r"\bmkfs(\.\w+)?\b", "filesystem format"),
    (r"\bdd\s+.*\bof=", "raw disk write with dd"),
    (r">\s*/dev/(sd|nvme|hd|disk)", "writing to a block device"),
    (r"\b(shutdown|reboot|halt|poweroff)\b", "system power command"),
    (r"\bsystemctl\s+(stop|disable|mask|poweroff|reboot)\b", "service shutdown"),
    (r"\bkill(all)?\s+-9\b", "forced process kill"),
    (r":\(\)\s*\{\s*:\|:\s*&\s*\}\s*;", "fork bomb"),
    (r"\bchmod\s+(-R\s+)?[0-7]*777\b", "world-writable chmod"),
    (r"\bchown\s+-R\b", "recursive chown"),
    (r"\bsudo\b|\bsu\b\s", "privilege escalation"),
    (r"\bgit\s+push\s+.*(-f\b|--force)", "git force push"),
    (r"\bgit\s+reset\s+--hard", "git hard reset"),
    (r"\bgit\s+clean\s+-[a-zA-Z]*[fd]", "git clean"),
    (r"\bgit\s+checkout\s+--\s+\.", "git checkout discarding changes"),
    (r"\bgit\s+branch\s+-D\b", "git branch force delete"),
    (r"\b(curl|wget)\b.*\|\s*(ba|z|k)?sh\b", "piping a download into a shell"),
    (r"\b(truncate|shred)\b", "file destruction"),
    (r"\bfind\b.*\s-delete\b", "find -delete"),
    (r"\bxargs\s+rm\b", "xargs rm"),
    (r">\s*/etc/", "writing to /etc"),
    (r"\bcrontab\s+-r\b", "crontab removal"),
    (r"\b(docker|podman)\s+(system\s+prune|rm\s+-f|rmi\s+-f)", "container destruction"),
)


@dataclass
class CommandPolicy:
    """Decide whether a shell command may run."""

    blocked_patterns: list[tuple[Pattern[str], str]] = field(default_factory=list)
    allowed_prefixes: list[str] = field(default_factory=list)
    timeout: float = 60.0
    require_confirmation: bool = True

    def __post_init__(self) -> None:
        if not self.blocked_patterns:
            self.blocked_patterns = [
                (re.compile(p, re.IGNORECASE), why) for p, why in DEFAULT_BLOCKED_PATTERNS
            ]

    @classmethod
    def from_settings(cls, settings: SecuritySettings) -> CommandPolicy:
        policy = cls(timeout=settings.shell_timeout, require_confirmation=settings.confirm_shell)
        for extra in settings.blocked_commands:
            policy.block(extra, f"blocked by configuration: {extra}")
        policy.allowed_prefixes = list(settings.allowed_commands)
        return policy

    @classmethod
    def permissive(cls) -> CommandPolicy:
        """Default block-list only, no confirmation prompt. For scripts/tests."""
        return cls(require_confirmation=False)

    def block(self, pattern: str, reason: str) -> None:
        self.blocked_patterns.append((re.compile(pattern, re.IGNORECASE), reason))

    def allow_only(self, *prefixes: str) -> None:
        self.allowed_prefixes = list(prefixes)

    def check(self, command: str) -> None:
        """Raise :class:`CommandBlockedError` if ``command`` is not permitted."""
        stripped = command.strip()
        if not stripped:
            raise CommandBlockedError(command, "empty command")
        for pattern, reason in self.blocked_patterns:
            if pattern.search(stripped):
                raise CommandBlockedError(command, reason)
        if self.allowed_prefixes:
            for segment in _split_segments(stripped):
                if not any(segment.startswith(p) for p in self.allowed_prefixes):
                    program = segment.split()[0] if segment.split() else segment
                    raise CommandBlockedError(command, f"'{program}' is not in the allow-list")

    def is_allowed(self, command: str) -> bool:
        try:
            self.check(command)
        except CommandBlockedError:
            return False
        return True


def _split_segments(command: str) -> list[str]:
    """Split ``a && b | c ; d`` into its component commands (best effort)."""
    try:
        tokens = shlex.split(command, posix=True)
    except ValueError:
        return [command]
    segments: list[list[str]] = [[]]
    for token in tokens:
        if token in ("&&", "||", "|", ";", "&"):
            segments.append([])
        else:
            segments[-1].append(token)
    return [" ".join(seg) for seg in segments if seg]
