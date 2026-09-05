"""Built-in reference agents. Importing this package registers them.

Import order defines menu order, so import sorting is disabled for this file.
"""

# ruff: noqa: I001

from local_agents.agents.builtin.research import ResearchAgent
from local_agents.agents.builtin.coding import CodingAgent
from local_agents.agents.builtin.web import WebAgent
from local_agents.agents.builtin.file import FileAgent
from local_agents.agents.builtin.data import DataAgent
from local_agents.agents.builtin.planning import PlanningAgent

#: Menu order.
BUILTIN_AGENTS = (ResearchAgent, CodingAgent, WebAgent, FileAgent, DataAgent, PlanningAgent)

__all__ = [
    "BUILTIN_AGENTS",
    "CodingAgent",
    "DataAgent",
    "FileAgent",
    "PlanningAgent",
    "ResearchAgent",
    "WebAgent",
]
