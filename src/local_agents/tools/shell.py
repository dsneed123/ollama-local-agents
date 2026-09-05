"""Shell execution tool with policy checks, confirmation, and timeouts."""

from __future__ import annotations

import asyncio
import os
from collections.abc import Awaitable, Callable

from local_agents.errors import CommandBlockedError, ToolError
from local_agents.security.commands import CommandPolicy
from local_agents.security.permissions import Permission
from local_agents.security.workspace import Workspace
from local_agents.tools.base import BaseTool, ToolParameter, ToolResult

ConfirmCallback = Callable[[str], Awaitable[bool]]

MAX_OUTPUT_CHARS = 20_000


class ShellTool(BaseTool):
    """Run a shell command inside the workspace.

    Safety layers, in order:

    1. :class:`CommandPolicy` blocks destructive commands outright.
    2. If the policy requires confirmation, ``confirm`` is awaited with the
       command. With no callback the command is refused (never silently run).
    3. The process runs with ``cwd`` = workspace root and a hard timeout.
    """

    name = "run_command"
    description = (
        "Run a shell command in the workspace directory and return its output. "
        "Use for running tests, linters, git status, or inspecting the environment. "
        "Destructive commands are blocked."
    )
    parameters = (
        ToolParameter("command", "string", "The shell command to run"),
        ToolParameter("timeout", "integer", "Seconds before the command is killed", required=False),
    )
    permission = Permission.EXECUTE

    def __init__(
        self,
        workspace: Workspace,
        policy: CommandPolicy | None = None,
        *,
        confirm: ConfirmCallback | None = None,
        env: dict[str, str] | None = None,
    ) -> None:
        self.workspace = workspace
        self.policy = policy or CommandPolicy.permissive()
        self.confirm = confirm
        self.env = env

    async def execute(self, command: str, timeout: int | None = None) -> ToolResult:
        self.policy.check(command)
        if self.policy.require_confirmation:
            if self.confirm is None:
                raise CommandBlockedError(
                    command,
                    "command execution requires confirmation but no confirmation handler is set",
                )
            if not await self.confirm(command):
                return ToolResult.fail("Command was not approved by the user", blocked=True)

        limit = float(timeout) if timeout else self.policy.timeout
        env = dict(os.environ if self.env is None else self.env)
        env.setdefault("PYTHONUNBUFFERED", "1")
        try:
            process = await asyncio.create_subprocess_shell(
                command,
                cwd=str(self.workspace.root),
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                stdin=asyncio.subprocess.DEVNULL,
                env=env,
            )
        except OSError as exc:
            raise ToolError(f"Could not start command: {exc}") from exc
        try:
            stdout, stderr = await asyncio.wait_for(process.communicate(), timeout=limit)
        except asyncio.TimeoutError:
            process.kill()
            await process.wait()
            return ToolResult.fail(f"Command timed out after {limit:g}s: {command}")

        out = stdout.decode("utf-8", "replace")
        err = stderr.decode("utf-8", "replace")
        code = process.returncode or 0
        parts = []
        if out.strip():
            parts.append(out.rstrip())
        if err.strip():
            parts.append("[stderr]\n" + err.rstrip())
        text = "\n".join(parts) if parts else "(no output)"
        if len(text) > MAX_OUTPUT_CHARS:
            text = text[:MAX_OUTPUT_CHARS] + f"\n... [truncated, {len(text):,} chars total]"
        text = f"$ {command}\n{text}\n[exit code {code}]"
        return ToolResult(
            output=text,
            success=code == 0,
            error=None if code == 0 else f"exit code {code}",
            data={"stdout": out, "stderr": err, "returncode": code},
        )


def shell_tools(
    workspace: Workspace,
    policy: CommandPolicy | None = None,
    *,
    confirm: ConfirmCallback | None = None,
) -> list[BaseTool]:
    return [ShellTool(workspace, policy, confirm=confirm)]
