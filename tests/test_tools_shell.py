from __future__ import annotations

import sys

from local_agents.security.commands import CommandPolicy
from local_agents.security.workspace import Workspace
from local_agents.tools.shell import ShellTool


async def test_runs_command_in_workspace(workspace: Workspace) -> None:
    tool = ShellTool(workspace, CommandPolicy.permissive())
    result = await tool.run(command=f'{sys.executable} -c "import os; print(os.getcwd())"')
    assert result.success
    assert str(workspace.root) in result.output
    assert result.data["returncode"] == 0


async def test_nonzero_exit_and_stderr(workspace: Workspace) -> None:
    tool = ShellTool(workspace, CommandPolicy.permissive())
    result = await tool.run(
        command=f"{sys.executable} -c \"import sys; sys.stderr.write('bad'); sys.exit(3)\""
    )
    assert not result.success
    assert "[stderr]" in result.output and "bad" in result.output
    assert "exit code 3" in result.output


async def test_timeout(workspace: Workspace) -> None:
    tool = ShellTool(workspace, CommandPolicy.permissive())
    result = await tool.run(command=f'{sys.executable} -c "import time; time.sleep(5)"', timeout=1)
    assert not result.success and "timed out" in (result.error or "")


async def test_blocked_command_never_runs(workspace: Workspace) -> None:
    tool = ShellTool(workspace, CommandPolicy.permissive())
    result = await tool.run(command="rm -rf src")
    assert result.blocked
    assert (workspace.root / "src" / "app.py").exists()


async def test_confirmation_required_without_handler(workspace: Workspace) -> None:
    tool = ShellTool(workspace, CommandPolicy())  # require_confirmation=True by default
    result = await tool.run(command="echo hi")
    assert result.blocked and "confirmation" in (result.error or "")


async def test_confirmation_handler_decides(workspace: Workspace) -> None:
    asked: list[str] = []

    async def deny(command: str) -> bool:
        asked.append(command)
        return False

    async def allow(command: str) -> bool:
        return True

    denied = await ShellTool(workspace, CommandPolicy(), confirm=deny).run(command="echo hi")
    assert denied.blocked and asked == ["echo hi"]
    allowed = await ShellTool(workspace, CommandPolicy(), confirm=allow).run(command="echo hi")
    assert allowed.success and "hi" in allowed.output


async def test_output_truncation(workspace: Workspace) -> None:
    tool = ShellTool(workspace, CommandPolicy.permissive())
    result = await tool.run(command=f"{sys.executable} -c \"print('x' * 50000)\"")
    assert result.success and "truncated" in result.output
