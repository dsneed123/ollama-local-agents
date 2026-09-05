"""Terminal presentation helpers and a testable console abstraction."""

from __future__ import annotations

import os
import sys
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from typing import TextIO

BANNER = """\
╔══════════════════════════════════════════════╗
║              LOCAL AGENT STACK               ║
║        Local AI • Modular • Extensible       ║
╚══════════════════════════════════════════════╝"""


def supports_color(stream: TextIO | None = None) -> bool:
    if os.environ.get("NO_COLOR"):
        return False
    if os.environ.get("FORCE_COLOR"):
        return True
    stream = stream or sys.stdout
    return hasattr(stream, "isatty") and stream.isatty()


class Style:
    """ANSI styling that degrades to plain text when colour is off."""

    def __init__(self, enabled: bool) -> None:
        self.enabled = enabled

    def _wrap(self, code: str, text: str) -> str:
        return f"\x1b[{code}m{text}\x1b[0m" if self.enabled else text

    def bold(self, text: str) -> str:
        return self._wrap("1", text)

    def dim(self, text: str) -> str:
        return self._wrap("2", text)

    def green(self, text: str) -> str:
        return self._wrap("32", text)

    def red(self, text: str) -> str:
        return self._wrap("31", text)

    def yellow(self, text: str) -> str:
        return self._wrap("33", text)

    def cyan(self, text: str) -> str:
        return self._wrap("36", text)

    def magenta(self, text: str) -> str:
        return self._wrap("35", text)


@dataclass
class Console:
    """Minimal I/O surface used by every screen.

    ``read`` returns ``None`` on EOF so loops can exit cleanly. Tests supply
    scripted ``read`` functions and capture ``out``.
    """

    out: TextIO = field(default_factory=lambda: sys.stdout)
    read: Callable[[str], str | None] = field(default=None)  # type: ignore[assignment]
    color: bool | None = None
    style: Style = field(init=False)

    def __post_init__(self) -> None:
        if self.read is None:
            self.read = _stdin_reader
        self.style = Style(supports_color(self.out) if self.color is None else self.color)

    @classmethod
    def scripted(cls, inputs: Iterable[str], out: TextIO) -> Console:
        iterator = iter(inputs)

        def _read(prompt: str) -> str | None:
            out.write(prompt)
            try:
                value = next(iterator)
            except StopIteration:
                return None
            out.write(value + "\n")
            return value

        return cls(out=out, read=_read, color=False)

    # ------------------------------------------------------------------ #

    def write(self, text: str = "") -> None:
        self.out.write(text)
        self.out.flush()

    def print(self, text: str = "") -> None:
        self.out.write(text + "\n")
        self.out.flush()

    def rule(self, width: int = 48) -> None:
        self.print(self.style.dim("─" * width))

    def info(self, text: str) -> None:
        self.print(self.style.cyan("i ") + text)

    def success(self, text: str) -> None:
        self.print(self.style.green("✓ ") + text)

    def warn(self, text: str) -> None:
        self.print(self.style.yellow("⚠ ") + text)

    def error(self, text: str) -> None:
        self.print(self.style.red("✗ ") + text)

    def prompt(self, text: str = "> ") -> str | None:
        value = self.read(text)
        return value.strip() if value is not None else None

    def confirm(self, question: str, *, default: bool = False) -> bool:
        suffix = " [Y/n] " if default else " [y/N] "
        answer = self.prompt(question + suffix)
        if answer is None or answer == "":
            return default
        return answer.lower() in ("y", "yes")


def _stdin_reader(prompt: str) -> str | None:
    try:
        return input(prompt)
    except EOFError:
        return None


def format_size(size_bytes: int) -> str:
    value = float(size_bytes)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if value < 1024 or unit == "TB":
            return f"{value:.1f} {unit}" if unit != "B" else f"{int(value)} B"
        value /= 1024
    return f"{size_bytes} B"


def truncate(text: str, width: int = 80) -> str:
    text = text.replace("\n", " ")
    return text if len(text) <= width else text[: width - 1] + "…"
