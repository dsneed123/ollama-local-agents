"""Built-in reference agents. Importing this package registers them.

Import order defines registration order within each category, so import
sorting is disabled for this file.
"""

# ruff: noqa: I001

from local_agents.agents.builtin.research import ResearchAgent
from local_agents.agents.builtin.coding import CodingAgent
from local_agents.agents.builtin.web import WebAgent
from local_agents.agents.builtin.file import FileAgent
from local_agents.agents.builtin.data import DataAgent
from local_agents.agents.builtin.planning import PlanningAgent
from local_agents.agents.builtin.development import (
    ApiAgent,
    CodeReviewAgent,
    DebugAgent,
    DocsAgent,
    GitAgent,
    RegexAgent,
    SecurityAuditAgent,
    TestAgent,
)
from local_agents.agents.builtin.data_files import (
    ConvertAgent,
    LogAnalysisAgent,
    NotesAgent,
    SqlAgent,
    SystemAgent,
)
from local_agents.agents.builtin.writing import (
    ApplicationAgent,
    MeetingNotesAgent,
    SummarizerAgent,
    TranslationAgent,
    WritingAgent,
)
from local_agents.agents.builtin.everyday import (
    BrainstormAgent,
    ChatAgent,
    MathAgent,
    TutorAgent,
)

#: Registration (and menu) order.
BUILTIN_AGENTS = (
    ResearchAgent,
    CodingAgent,
    WebAgent,
    FileAgent,
    DataAgent,
    PlanningAgent,
    CodeReviewAgent,
    TestAgent,
    DebugAgent,
    DocsAgent,
    GitAgent,
    SecurityAuditAgent,
    RegexAgent,
    ApiAgent,
    SqlAgent,
    LogAnalysisAgent,
    ConvertAgent,
    NotesAgent,
    SystemAgent,
    WritingAgent,
    SummarizerAgent,
    TranslationAgent,
    MeetingNotesAgent,
    ApplicationAgent,
    ChatAgent,
    TutorAgent,
    MathAgent,
    BrainstormAgent,
)

__all__ = [
    "BUILTIN_AGENTS",
    "ApiAgent",
    "ApplicationAgent",
    "BrainstormAgent",
    "ChatAgent",
    "CodeReviewAgent",
    "CodingAgent",
    "ConvertAgent",
    "DataAgent",
    "DebugAgent",
    "DocsAgent",
    "FileAgent",
    "GitAgent",
    "LogAnalysisAgent",
    "MathAgent",
    "MeetingNotesAgent",
    "NotesAgent",
    "PlanningAgent",
    "RegexAgent",
    "ResearchAgent",
    "SecurityAuditAgent",
    "SqlAgent",
    "SummarizerAgent",
    "SystemAgent",
    "TestAgent",
    "TranslationAgent",
    "TutorAgent",
    "WebAgent",
    "WritingAgent",
]
