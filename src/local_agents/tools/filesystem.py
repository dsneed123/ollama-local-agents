"""Filesystem tools. All paths are validated through a :class:`Workspace`."""

from __future__ import annotations

import fnmatch
import re
import shutil
from pathlib import Path

from local_agents.errors import ToolError
from local_agents.security.permissions import Permission
from local_agents.security.workspace import Workspace
from local_agents.tools.base import BaseTool, ToolParameter, ToolResult

TEXT_SAMPLE_BYTES = 4096


class _WorkspaceTool(BaseTool):
    """Base for tools that operate inside a workspace."""

    def __init__(self, workspace: Workspace) -> None:
        self.workspace = workspace

    def _read_text(self, path: Path) -> str:
        size = path.stat().st_size
        if size > self.workspace.max_file_bytes:
            raise ToolError(
                f"File is too large to read ({size:,} bytes > {self.workspace.max_file_bytes:,})",
                hint="Use read_file with start_line/end_line to read part of it.",
            )
        data = path.read_bytes()
        if _looks_binary(data[:TEXT_SAMPLE_BYTES]):
            raise ToolError(f"{self.workspace.relative(path)} looks like a binary file")
        return data.decode("utf-8", errors="replace")


class ReadFileTool(_WorkspaceTool):
    name = "read_file"
    description = (
        "Read a text file from the workspace. Optionally read only a line range. "
        "Output is numbered by line."
    )
    parameters = (
        ToolParameter("path", "string", "Path relative to the workspace root"),
        ToolParameter("start_line", "integer", "First line to read (1-based)", required=False),
        ToolParameter("end_line", "integer", "Last line to read (inclusive)", required=False),
    )
    permission = Permission.READ

    async def execute(
        self, path: str, start_line: int | None = None, end_line: int | None = None
    ) -> ToolResult:
        target = self.workspace.resolve(path, must_exist=True)
        if target.is_dir():
            raise IsADirectoryError(f"{path} is a directory; use list_directory")
        lines = self._read_text(target).splitlines()
        start = max(1, start_line or 1)
        end = min(len(lines), end_line or len(lines))
        if start > len(lines):
            return ToolResult.ok(f"(file has only {len(lines)} lines)")
        width = len(str(end))
        body = "\n".join(f"{i:>{width}}| {lines[i - 1]}" for i in range(start, end + 1))
        header = f"{self.workspace.relative(target)} (lines {start}-{end} of {len(lines)})\n"
        return ToolResult.ok(header + body, data={"lines": lines[start - 1 : end]})


class WriteFileTool(_WorkspaceTool):
    name = "write_file"
    description = (
        "Create or overwrite a text file in the workspace with the given content. "
        "Parent directories are created."
    )
    parameters = (
        ToolParameter("path", "string", "Path relative to the workspace root"),
        ToolParameter("content", "string", "Full file content"),
    )
    permission = Permission.WRITE

    async def execute(self, path: str, content: str) -> ToolResult:
        target = self.workspace.check_write(path)
        if target.is_dir():
            raise IsADirectoryError(f"{path} is a directory")
        existed = target.exists()
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")
        verb = "Updated" if existed else "Created"
        return ToolResult.ok(
            f"{verb} {self.workspace.relative(target)} ({len(content.splitlines())} lines)"
        )


class EditFileTool(_WorkspaceTool):
    name = "edit_file"
    description = (
        "Replace an exact text snippet in a file with new text. The snippet must "
        "appear exactly once. Use this for targeted edits instead of rewriting files."
    )
    parameters = (
        ToolParameter("path", "string", "Path relative to the workspace root"),
        ToolParameter("old_text", "string", "Exact text to replace (must be unique)"),
        ToolParameter("new_text", "string", "Replacement text"),
    )
    permission = Permission.WRITE

    async def execute(self, path: str, old_text: str, new_text: str) -> ToolResult:
        target = self.workspace.check_write(path)
        if not target.is_file():
            raise FileNotFoundError(f"No such file: {path}")
        original = self._read_text(target)
        count = original.count(old_text)
        if count == 0:
            raise ToolError(
                f"old_text not found in {path}",
                hint="Read the file first and copy the snippet exactly, including whitespace.",
            )
        if count > 1:
            raise ToolError(
                f"old_text appears {count} times in {path}; include more context to make it unique"
            )
        updated = original.replace(old_text, new_text, 1)
        target.write_text(updated, encoding="utf-8")
        delta = len(new_text.splitlines()) - len(old_text.splitlines())
        return ToolResult.ok(f"Edited {self.workspace.relative(target)} ({delta:+d} lines)")


class ListDirectoryTool(_WorkspaceTool):
    name = "list_directory"
    description = "List files and directories at a path in the workspace."
    parameters = (
        ToolParameter(
            "path",
            "string",
            "Directory path (default: workspace root)",
            required=False,
            default=".",
        ),
        ToolParameter(
            "recursive", "boolean", "Recurse into subdirectories", required=False, default=False
        ),
        ToolParameter(
            "max_entries", "integer", "Maximum entries to return", required=False, default=200
        ),
    )
    permission = Permission.READ

    async def execute(
        self, path: str = ".", recursive: bool = False, max_entries: int = 200
    ) -> ToolResult:
        target = self.workspace.resolve(path, must_exist=True)
        if not target.is_dir():
            raise NotADirectoryError(f"{path} is not a directory")
        entries: list[str] = []
        iterator = target.rglob("*") if recursive else target.iterdir()
        truncated = False
        for item in sorted(iterator, key=lambda p: (not p.is_dir(), str(p).lower())):
            if self.workspace.is_ignored(item.relative_to(target)):
                continue
            if len(entries) >= max_entries:
                truncated = True
                break
            rel = item.relative_to(target)
            if item.is_dir():
                entries.append(f"{rel}/")
            else:
                try:
                    entries.append(f"{rel}  ({_human_size(item.stat().st_size)})")
                except OSError:
                    entries.append(str(rel))
        header = f"{self.workspace.relative(target)}/ — {len(entries)} entries"
        if truncated:
            header += f" (truncated at {max_entries})"
        return ToolResult.ok(header + "\n" + "\n".join(entries), data=entries)


class FindFilesTool(_WorkspaceTool):
    name = "find_files"
    description = "Find files whose name matches a glob pattern, e.g. '*.py' or 'test_*.py'."
    parameters = (
        ToolParameter("pattern", "string", "Glob pattern matched against file names"),
        ToolParameter(
            "path",
            "string",
            "Directory to search (default: workspace root)",
            required=False,
            default=".",
        ),
        ToolParameter("max_results", "integer", "Maximum matches", required=False, default=200),
    )
    permission = Permission.READ

    async def execute(self, pattern: str, path: str = ".", max_results: int = 200) -> ToolResult:
        root = self.workspace.resolve(path, must_exist=True)
        matches: list[str] = []
        for item in root.rglob("*"):
            rel = item.relative_to(root)
            if self.workspace.is_ignored(rel):
                continue
            if item.is_file() and (
                fnmatch.fnmatch(item.name, pattern) or fnmatch.fnmatch(str(rel), pattern)
            ):
                matches.append(self.workspace.relative(item))
                if len(matches) >= max_results:
                    break
        if not matches:
            return ToolResult.ok(
                f"No files matching {pattern!r} under {self.workspace.relative(root)}"
            )
        return ToolResult.ok("\n".join(sorted(matches)), data=matches)


class SearchTextTool(_WorkspaceTool):
    name = "search_text"
    description = (
        "Search file contents for a regular expression (like grep -rn). "
        "Returns matching lines with file names and line numbers."
    )
    parameters = (
        ToolParameter("pattern", "string", "Regular expression to search for"),
        ToolParameter(
            "path",
            "string",
            "Directory or file to search (default: workspace root)",
            required=False,
            default=".",
        ),
        ToolParameter(
            "glob", "string", "Only search files matching this glob, e.g. '*.py'", required=False
        ),
        ToolParameter(
            "max_results", "integer", "Maximum matching lines", required=False, default=100
        ),
        ToolParameter(
            "ignore_case", "boolean", "Case-insensitive search", required=False, default=False
        ),
    )
    permission = Permission.READ

    async def execute(
        self,
        pattern: str,
        path: str = ".",
        glob: str | None = None,
        max_results: int = 100,
        ignore_case: bool = False,
    ) -> ToolResult:
        try:
            regex = re.compile(pattern, re.IGNORECASE if ignore_case else 0)
        except re.error as exc:
            raise ToolError(f"Invalid regular expression: {exc}") from exc
        root = self.workspace.resolve(path, must_exist=True)
        files = [root] if root.is_file() else sorted(p for p in root.rglob("*") if p.is_file())
        hits: list[str] = []
        scanned = 0
        for file in files:
            if root.is_dir() and self.workspace.is_ignored(file.relative_to(root)):
                continue
            if glob and not fnmatch.fnmatch(file.name, glob):
                continue
            try:
                if file.stat().st_size > self.workspace.max_file_bytes:
                    continue
                data = file.read_bytes()
            except OSError:
                continue
            if _looks_binary(data[:TEXT_SAMPLE_BYTES]):
                continue
            scanned += 1
            for number, line in enumerate(data.decode("utf-8", "replace").splitlines(), 1):
                if regex.search(line):
                    hits.append(f"{self.workspace.relative(file)}:{number}: {line.strip()[:300]}")
                    if len(hits) >= max_results:
                        break
            if len(hits) >= max_results:
                break
        if not hits:
            return ToolResult.ok(f"No matches for /{pattern}/ in {scanned} files")
        suffix = f"\n... (stopped at {max_results} results)" if len(hits) >= max_results else ""
        return ToolResult.ok("\n".join(hits) + suffix, data=hits)


class MoveFileTool(_WorkspaceTool):
    name = "move_file"
    description = (
        "Move or rename a file or directory inside the workspace. "
        "Refuses to overwrite an existing destination."
    )
    parameters = (
        ToolParameter("source", "string", "Existing path"),
        ToolParameter("destination", "string", "New path"),
    )
    permission = Permission.WRITE

    async def execute(self, source: str, destination: str) -> ToolResult:
        src = self.workspace.resolve(source, must_exist=True)
        dst = self.workspace.check_write(destination)
        if dst.exists():
            raise ToolError(f"Destination already exists: {destination}")
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(src), str(dst))
        return ToolResult.ok(
            f"Moved {self.workspace.relative(src)} -> {self.workspace.relative(dst)}"
        )


class CreateDirectoryTool(_WorkspaceTool):
    name = "create_directory"
    description = "Create a directory (and parents) inside the workspace."
    parameters = (ToolParameter("path", "string", "Directory path to create"),)
    permission = Permission.WRITE

    async def execute(self, path: str) -> ToolResult:
        target = self.workspace.check_write(path)
        target.mkdir(parents=True, exist_ok=True)
        return ToolResult.ok(f"Created directory {self.workspace.relative(target)}")


class DeleteFileTool(_WorkspaceTool):
    name = "delete_file"
    description = "Delete a single file. Disabled unless the workspace allows deletion."
    parameters = (ToolParameter("path", "string", "File to delete"),)
    permission = Permission.DELETE

    async def execute(self, path: str) -> ToolResult:
        target = self.workspace.check_delete(path)
        if target.is_dir():
            raise ToolError("Refusing to delete a directory; delete files individually")
        target.unlink()
        return ToolResult.ok(f"Deleted {self.workspace.relative(target)}")


def filesystem_tools(workspace: Workspace) -> list[BaseTool]:
    """Standard filesystem tool set for ``workspace``.

    ``delete_file`` is only included when the workspace allows deletion.
    """
    tools: list[BaseTool] = [
        ReadFileTool(workspace),
        ListDirectoryTool(workspace),
        FindFilesTool(workspace),
        SearchTextTool(workspace),
        WriteFileTool(workspace),
        EditFileTool(workspace),
        CreateDirectoryTool(workspace),
        MoveFileTool(workspace),
    ]
    if workspace.allow_delete:
        tools.append(DeleteFileTool(workspace))
    return tools


def _looks_binary(sample: bytes) -> bool:
    if not sample:
        return False
    if b"\x00" in sample:
        return True
    text_chars = bytes(range(32, 127)) + b"\n\r\t\b\f\x1b"
    non_text = sum(byte not in text_chars for byte in sample if byte < 128)
    return non_text / max(len(sample), 1) > 0.3


def _human_size(size: int) -> str:
    value = float(size)
    for unit in ("B", "KB", "MB", "GB"):
        if value < 1024 or unit == "GB":
            return f"{value:.0f}{unit}" if unit == "B" else f"{value:.1f}{unit}"
        value /= 1024
    return f"{size}B"
