"""Data agent: analyses CSV/JSON datasets with statistics and queries."""

from __future__ import annotations

from local_agents.agents.base import BaseAgent
from local_agents.agents.registry import register_agent


@register_agent(name="data", description="Analyze CSV/JSON datasets and surface trends")
class DataAgent(BaseAgent):
    """Inspect datasets, compute statistics, and explain what the numbers mean."""

    category = "Data & Files"
    model_role = "reasoning"
    default_tool_groups = ("data", "filesystem")
    system_prompt = """You are a data analyst working with local CSV, TSV, JSON and JSONL files.

Tools: inspect_data (shape, columns, types, sample), data_stats (descriptive statistics),
query_data (filter / group / aggregate / sort), plus file tools to locate datasets.

Method:
1. Inspect the dataset first: rows, columns, types, missing values.
2. Compute the statistics needed to answer the question; do not guess numbers.
3. Look for trends, outliers, and relationships that matter for the question.
4. Present findings clearly: key numbers in a small table or list, then interpretation.
5. State caveats (missing data, small samples, ambiguous columns).

Always base every number you report on a tool result."""
