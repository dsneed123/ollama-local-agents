"""Text utilities: regex testing, match statistics, and tailing files."""

from __future__ import annotations

import re
from collections import Counter
from pathlib import Path

from local_agents.errors import ToolError
from local_agents.security.permissions import Permission
from local_agents.security.workspace import Workspace
from local_agents.tools.base import BaseTool, ToolParameter, ToolResult

MAX_MATCHES = 200


def _compile(pattern: str, ignore_case: bool = False, multiline: bool = False) -> re.Pattern[str]:
    flags = 0
    if ignore_case:
        flags |= re.IGNORECASE
    if multiline:
        flags |= re.MULTILINE
    try:
        return re.compile(pattern, flags)
    except re.error as exc:
        raise ToolError(
            f"Invalid regular expression: {exc}", hint=f"Pattern was: {pattern}"
        ) from exc


class RegexTestTool(BaseTool):
    name = "test_regex"
    description = (
        "Test a regular expression against sample text. Returns every match with its "
        "position and capture groups, and optionally the result of a substitution."
    )
    parameters = (
        ToolParameter("pattern", "string", "Python regular expression"),
        ToolParameter("text", "string", "Sample text to match against"),
        ToolParameter(
            "replacement", "string", "If given, also show re.sub() output", required=False
        ),
        ToolParameter("ignore_case", "boolean", "Case-insensitive", required=False, default=False),
        ToolParameter(
            "multiline",
            "boolean",
            "^ and $ match at line boundaries",
            required=False,
            default=False,
        ),
    )
    permission = Permission.READ

    async def execute(
        self,
        pattern: str,
        text: str,
        replacement: str | None = None,
        ignore_case: bool = False,
        multiline: bool = False,
    ) -> ToolResult:
        regex = _compile(pattern, ignore_case, multiline)
        matches = list(regex.finditer(text))
        lines = [f"{len(matches)} match(es) for /{pattern}/"]
        for index, match in enumerate(matches[:MAX_MATCHES], 1):
            line = f"{index}. {match.group(0)!r} at {match.start()}-{match.end()}"
            if match.groups():
                line += f"  groups={match.groups()!r}"
            if match.groupdict():
                line += f"  named={match.groupdict()!r}"
            lines.append(line)
        if len(matches) > MAX_MATCHES:
            lines.append(f"... ({len(matches) - MAX_MATCHES} more)")
        if replacement is not None:
            try:
                lines.append("Substitution result:\n" + regex.sub(replacement, text))
            except re.error as exc:
                raise ToolError(f"Invalid replacement: {exc}") from exc
        data = [
            {"match": m.group(0), "start": m.start(), "end": m.end(), "groups": list(m.groups())}
            for m in matches[:MAX_MATCHES]
        ]
        return ToolResult.ok("\n".join(lines), data=data)


class CountMatchesTool(BaseTool):
    """Frequency table of regex matches in a file (great for logs)."""

    name = "count_matches"
    description = (
        "Count how often each distinct match of a regular expression occurs in a file and "
        "return a frequency table. Use a capture group to count only part of the match, "
        "e.g. 'ERROR (\\w+)' to count error types."
    )
    parameters = (
        ToolParameter("path", "string", "File to scan"),
        ToolParameter("pattern", "string", "Regular expression; group 1 is counted if present"),
        ToolParameter(
            "top", "integer", "How many distinct values to show", required=False, default=20
        ),
        ToolParameter("ignore_case", "boolean", "Case-insensitive", required=False, default=False),
    )
    permission = Permission.READ

    def __init__(self, workspace: Workspace) -> None:
        self.workspace = workspace

    async def execute(
        self, path: str, pattern: str, top: int = 20, ignore_case: bool = False
    ) -> ToolResult:
        regex = _compile(pattern, ignore_case)
        target = self.workspace.resolve(path, must_exist=True)
        counter: Counter[str] = Counter()
        total_lines = 0
        matching_lines = 0
        with target.open("r", encoding="utf-8", errors="replace") as fh:
            for line in fh:
                total_lines += 1
                found = False
                for match in regex.finditer(line):
                    found = True
                    counter[match.group(1) if match.groups() else match.group(0)] += 1
                matching_lines += found
        total = sum(counter.values())
        lines = [
            f"{self.workspace.relative(target)}: {total} matches on {matching_lines} of "
            f"{total_lines} lines, {len(counter)} distinct values",
        ]
        for value, count in counter.most_common(top):
            share = 100 * count / total if total else 0
            lines.append(f"  {count:>7}  {share:5.1f}%  {value}")
        return ToolResult.ok(
            "\n".join(lines),
            data={"total": total, "lines": total_lines, "top": counter.most_common(top)},
        )


class TailFileTool(BaseTool):
    name = "tail_file"
    description = "Return the last N lines of a file (like `tail -n`). Useful for logs."
    parameters = (
        ToolParameter("path", "string", "File to read"),
        ToolParameter(
            "lines", "integer", "Number of lines from the end", required=False, default=50
        ),
    )
    permission = Permission.READ

    def __init__(self, workspace: Workspace) -> None:
        self.workspace = workspace

    async def execute(self, path: str, lines: int = 50) -> ToolResult:
        target = self.workspace.resolve(path, must_exist=True)
        if target.is_dir():
            raise IsADirectoryError(f"{path} is a directory")
        tail = _tail(target, max(1, lines))
        header = f"{self.workspace.relative(target)} (last {len(tail)} lines)\n"
        return ToolResult.ok(header + "".join(tail).rstrip("\n"), data=tail)


def _tail(path: Path, count: int, chunk: int = 65536) -> list[str]:
    """Read the last ``count`` lines without loading the whole file."""
    with path.open("rb") as fh:
        fh.seek(0, 2)
        remaining = fh.tell()
        buffer = b""
        while remaining > 0 and buffer.count(b"\n") <= count:
            step = min(chunk, remaining)
            remaining -= step
            fh.seek(remaining)
            buffer = fh.read(step) + buffer
    text = buffer.decode("utf-8", errors="replace")
    return text.splitlines(keepends=True)[-count:]


def text_tools(workspace: Workspace) -> list[BaseTool]:
    return [RegexTestTool(), CountMatchesTool(workspace), TailFileTool(workspace)]
