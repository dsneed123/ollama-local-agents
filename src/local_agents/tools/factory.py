"""Assemble tool groups from settings so agents receive tools by injection."""

from __future__ import annotations

from dataclasses import dataclass, field

import httpx

from local_agents.security.commands import CommandPolicy
from local_agents.security.permissions import PermissionSet
from local_agents.security.workspace import Workspace
from local_agents.tools.base import ApprovalCallback, BaseTool, ToolSet
from local_agents.tools.calculator import calculator_tools
from local_agents.tools.convert import convert_tools
from local_agents.tools.data import data_tools
from local_agents.tools.filesystem import filesystem_tools
from local_agents.tools.shell import ConfirmCallback, shell_tools
from local_agents.tools.sqlite import sqlite_tools
from local_agents.tools.text import text_tools
from local_agents.tools.web import SearchBackend, http_tools, web_tools

TOOL_GROUPS: tuple[str, ...] = (
    "filesystem",
    "shell",
    "web",
    "http",
    "data",
    "sqlite",
    "text",
    "convert",
    "calculator",
)


@dataclass
class ToolKit:
    """Everything needed to build tool sets for agents in one session."""

    workspace: Workspace
    policy: CommandPolicy = field(default_factory=CommandPolicy.permissive)
    confirm: ConfirmCallback | None = None
    approval: ApprovalCallback | None = None
    http_client: httpx.AsyncClient | None = None
    search_backend: SearchBackend | None = None
    allow_shell: bool = True
    allow_network: bool = True

    def group(self, name: str) -> list[BaseTool]:
        if name == "filesystem":
            return filesystem_tools(self.workspace)
        if name == "shell":
            return (
                shell_tools(self.workspace, self.policy, confirm=self.confirm)
                if self.allow_shell
                else []
            )
        if name == "web":
            return (
                web_tools(self.http_client, backend=self.search_backend)
                if self.allow_network
                else []
            )
        if name == "http":
            return http_tools(self.http_client) if self.allow_network else []
        if name == "data":
            return data_tools(self.workspace)
        if name == "sqlite":
            return sqlite_tools(self.workspace)
        if name == "text":
            return text_tools(self.workspace)
        if name == "convert":
            return convert_tools(self.workspace)
        if name == "calculator":
            return calculator_tools()
        raise ValueError(f"Unknown tool group {name!r}; valid groups: {', '.join(TOOL_GROUPS)}")

    def build(
        self,
        groups: tuple[str, ...] | list[str],
        *,
        permissions: PermissionSet | None = None,
        extra: list[BaseTool] | None = None,
    ) -> ToolSet:
        tools: list[BaseTool] = []
        for name in groups:
            tools.extend(self.group(name))
        tools.extend(extra or [])
        perms = permissions or self.default_permissions()
        return ToolSet(tools, permissions=perms, approval=self.approval)

    def default_permissions(self) -> PermissionSet:
        return PermissionSet(
            read=True,
            write=self.workspace.allow_write,
            delete=self.workspace.allow_delete,
            execute=self.allow_shell,
            network=self.allow_network,
            delegate=True,
        )
