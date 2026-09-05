from __future__ import annotations

from pathlib import Path

import pytest

from local_agents.security.workspace import Workspace
from local_agents.tools.base import ToolSet
from local_agents.tools.filesystem import (
    CreateDirectoryTool,
    DeleteFileTool,
    EditFileTool,
    FindFilesTool,
    ListDirectoryTool,
    MoveFileTool,
    ReadFileTool,
    SearchTextTool,
    WriteFileTool,
    filesystem_tools,
)


async def test_read_file(workspace: Workspace) -> None:
    result = await ReadFileTool(workspace).run(path="src/app.py")
    assert result.success
    assert "1| def hello():" in result.output
    ranged = await ReadFileTool(workspace).run(path="src/app.py", start_line=2, end_line=2)
    assert "return 'hi'" in ranged.output and "def hello" not in ranged.output
    beyond = await ReadFileTool(workspace).run(path="src/app.py", start_line=99)
    assert "only 2 lines" in beyond.output


async def test_read_errors(workspace: Workspace) -> None:
    missing = await ReadFileTool(workspace).run(path="nope.txt")
    assert not missing.success and "No such file" in (missing.error or "")
    directory = await ReadFileTool(workspace).run(path="src")
    assert not directory.success
    outside = await ReadFileTool(workspace).run(path="../etc/passwd")
    assert outside.blocked
    (workspace.root / "bin.dat").write_bytes(b"\x00\x01\x02binary")
    binary = await ReadFileTool(workspace).run(path="bin.dat")
    assert not binary.success and "binary" in (binary.error or "")


async def test_read_respects_size_limit(tmp_path: Path) -> None:
    ws = Workspace(root=tmp_path, max_file_bytes=10)
    (tmp_path / "big.txt").write_text("x" * 100)
    result = await ReadFileTool(ws).run(path="big.txt")
    assert not result.success and "too large" in (result.error or "")


async def test_write_and_edit(workspace: Workspace) -> None:
    write = await WriteFileTool(workspace).run(path="notes/todo.md", content="- one\n- two\n")
    assert write.success and "Created" in write.output
    assert (workspace.root / "notes" / "todo.md").read_text() == "- one\n- two\n"
    again = await WriteFileTool(workspace).run(path="notes/todo.md", content="x\n")
    assert "Updated" in again.output

    edit = await EditFileTool(workspace).run(path="src/app.py", old_text="'hi'", new_text="'hello'")
    assert edit.success
    assert "return 'hello'" in (workspace.root / "src" / "app.py").read_text()
    missing = await EditFileTool(workspace).run(path="src/app.py", old_text="zzz", new_text="y")
    assert not missing.success and "not found" in (missing.error or "")
    (workspace.root / "dup.txt").write_text("a a")
    ambiguous = await EditFileTool(workspace).run(path="dup.txt", old_text="a", new_text="b")
    assert not ambiguous.success and "2 times" in (ambiguous.error or "")


async def test_write_blocked_when_read_only(tmp_path: Path) -> None:
    ws = Workspace(root=tmp_path, allow_write=False)
    result = await WriteFileTool(ws).run(path="a.txt", content="x")
    assert result.blocked and not (tmp_path / "a.txt").exists()


async def test_list_find_search(workspace: Workspace) -> None:
    (workspace.root / "node_modules").mkdir()
    (workspace.root / "node_modules" / "x.js").write_text("ignored")
    listing = await ListDirectoryTool(workspace).run()
    assert "src/" in listing.output and "README.md" in listing.output
    assert "node_modules" not in listing.output
    recursive = await ListDirectoryTool(workspace).run(path=".", recursive=True, max_entries=2)
    assert "truncated" in recursive.output

    found = await FindFilesTool(workspace).run(pattern="*.py")
    assert found.data == ["src/app.py"]
    none = await FindFilesTool(workspace).run(pattern="*.rs")
    assert "No files" in none.output

    hits = await SearchTextTool(workspace).run(pattern="hello", glob="*.py")
    assert hits.data == ["src/app.py:1: def hello():"]
    insensitive = await SearchTextTool(workspace).run(pattern="HELLO", ignore_case=True, path="src")
    assert insensitive.success and len(insensitive.data) == 1
    bad = await SearchTextTool(workspace).run(pattern="(")
    assert not bad.success and "Invalid regular expression" in (bad.error or "")
    empty = await SearchTextTool(workspace).run(pattern="nothing-here")
    assert "No matches" in empty.output


async def test_move_and_mkdir(workspace: Workspace) -> None:
    mk = await CreateDirectoryTool(workspace).run(path="docs/guides")
    assert mk.success and (workspace.root / "docs" / "guides").is_dir()
    mv = await MoveFileTool(workspace).run(source="README.md", destination="docs/README.md")
    assert mv.success and (workspace.root / "docs" / "README.md").exists()
    clash = await MoveFileTool(workspace).run(source="src/app.py", destination="docs/README.md")
    assert not clash.success and "already exists" in (clash.error or "")


async def test_delete_requires_opt_in(tmp_path: Path) -> None:
    (tmp_path / "a.txt").write_text("x")
    ws = Workspace(root=tmp_path, allow_delete=False)
    assert "delete_file" not in {t.name for t in filesystem_tools(ws)}
    blocked = await DeleteFileTool(ws).run(path="a.txt")
    assert blocked.blocked and (tmp_path / "a.txt").exists()
    ws = Workspace(root=tmp_path, allow_delete=True)
    assert "delete_file" in {t.name for t in filesystem_tools(ws)}
    ok = await DeleteFileTool(ws).run(path="a.txt")
    assert ok.success and not (tmp_path / "a.txt").exists()
    (tmp_path / "d").mkdir()
    refused = await DeleteFileTool(ws).run(path="d")
    assert not refused.success


async def test_toolset_permission_gate(workspace: Workspace) -> None:
    from local_agents.security.permissions import PermissionSet

    tools = ToolSet(filesystem_tools(workspace), permissions=PermissionSet.read_only())
    assert "write_file" not in [t.name for t in tools.usable()]
    result = await tools.call("write_file", {"path": "x", "content": "y"})
    assert result.blocked and "write" in (result.error or "")
    ok = await tools.call("read_file", {"path": "README.md"})
    assert ok.success


async def test_toolset_validation_and_unknown(workspace: Workspace) -> None:
    tools = ToolSet(filesystem_tools(workspace))
    missing = await tools.call("read_file", {})
    assert not missing.success and "Missing required argument" in (missing.error or "")
    unknown = await tools.call("teleport", {})
    assert not unknown.success and "not available" in (unknown.error or "")
    coerced = await tools.call(
        "read_file", {"path": "README.md", "start_line": "1", "end_line": "1"}
    )
    assert coerced.success and "# Demo" in coerced.output
    with pytest.raises(ValueError):
        tools.add(ReadFileTool(workspace))


async def test_toolset_approval_hook(workspace: Workspace) -> None:
    from local_agents.security.permissions import PermissionSet

    seen: list[str] = []

    async def approve(tool, arguments) -> bool:
        seen.append(tool.name)
        return tool.name != "write_file"

    tools = ToolSet(filesystem_tools(workspace), permissions=PermissionSet.all(), approval=approve)
    denied = await tools.call("write_file", {"path": "x.txt", "content": "y"})
    assert denied.blocked and not (workspace.root / "x.txt").exists()
    allowed = await tools.call("create_directory", {"path": "made"})
    assert allowed.success
    read = await tools.call("read_file", {"path": "README.md"})
    assert read.success and "read_file" not in seen  # reads are never gated


def test_tool_spec_shape(workspace: Workspace) -> None:
    spec = ReadFileTool(workspace).spec().to_dict()
    assert spec["type"] == "function"
    assert spec["function"]["name"] == "read_file"
    assert spec["function"]["parameters"]["required"] == ["path"]
    assert spec["function"]["parameters"]["properties"]["start_line"]["type"] == "integer"


def test_tool_subclass_must_define_execute() -> None:
    from local_agents.tools.base import BaseTool

    with pytest.raises(TypeError):

        class Broken(BaseTool):
            name = "broken"
