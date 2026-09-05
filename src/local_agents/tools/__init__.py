"""Reusable tools. Import individual tools or use :class:`ToolKit` to build sets."""

from local_agents.tools.base import (
    ApprovalCallback,
    BaseTool,
    ToolCallRecord,
    ToolParameter,
    ToolResult,
    ToolSet,
)
from local_agents.tools.calculator import CalculatorTool, calculator_tools, evaluate
from local_agents.tools.convert import ConvertFileTool, ValidateFileTool, convert_tools
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
from local_agents.tools.sqlite import SQLiteQueryTool, SQLiteSchemaTool, sqlite_tools
from local_agents.tools.text import CountMatchesTool, RegexTestTool, TailFileTool, text_tools
from local_agents.tools.web import (
    DuckDuckGoBackend,
    HttpRequestTool,
    SearchBackend,
    SearchResult,
    WebFetchTool,
    WebSearchTool,
    html_to_text,
    http_tools,
    web_tools,
)

__all__ = [
    "TOOL_GROUPS",
    "ApprovalCallback",
    "BaseTool",
    "CalculatorTool",
    "ConvertFileTool",
    "CountMatchesTool",
    "CreateDirectoryTool",
    "DataQueryTool",
    "DataStatsTool",
    "DeleteFileTool",
    "DuckDuckGoBackend",
    "EditFileTool",
    "FindFilesTool",
    "HttpRequestTool",
    "InspectDataTool",
    "ListDirectoryTool",
    "MoveFileTool",
    "ReadFileTool",
    "RegexTestTool",
    "SQLiteQueryTool",
    "SQLiteSchemaTool",
    "SearchBackend",
    "SearchResult",
    "SearchTextTool",
    "ShellTool",
    "TailFileTool",
    "ToolCallRecord",
    "ToolKit",
    "ToolParameter",
    "ToolResult",
    "ToolSet",
    "ValidateFileTool",
    "WebFetchTool",
    "WebSearchTool",
    "WriteFileTool",
    "calculator_tools",
    "convert_tools",
    "data_tools",
    "evaluate",
    "filesystem_tools",
    "html_to_text",
    "http_tools",
    "shell_tools",
    "sqlite_tools",
    "text_tools",
    "web_tools",
]
