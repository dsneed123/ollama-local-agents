"""Reusable tools. Import individual tools or use :class:`ToolKit` to build sets."""

from local_agents.tools.base import (
    ApprovalCallback,
    BaseTool,
    ToolCallRecord,
    ToolParameter,
    ToolResult,
    ToolSet,
)
from local_agents.tools.data import DataQueryTool, DataStatsTool, InspectDataTool, data_tools
from local_agents.tools.factory import TOOL_GROUPS, ToolKit
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
from local_agents.tools.shell import ShellTool, shell_tools
from local_agents.tools.web import (
    DuckDuckGoBackend,
    SearchBackend,
    SearchResult,
    WebFetchTool,
    WebSearchTool,
    html_to_text,
    web_tools,
)

__all__ = [
    "TOOL_GROUPS",
    "ApprovalCallback",
    "BaseTool",
    "CreateDirectoryTool",
    "DataQueryTool",
    "DataStatsTool",
    "DeleteFileTool",
    "DuckDuckGoBackend",
    "EditFileTool",
    "FindFilesTool",
    "InspectDataTool",
    "ListDirectoryTool",
    "MoveFileTool",
    "ReadFileTool",
    "SearchBackend",
    "SearchResult",
    "SearchTextTool",
    "ShellTool",
    "ToolCallRecord",
    "ToolKit",
    "ToolParameter",
    "ToolResult",
    "ToolSet",
    "WebFetchTool",
    "WebSearchTool",
    "WriteFileTool",
    "data_tools",
    "filesystem_tools",
    "html_to_text",
    "shell_tools",
    "web_tools",
]
