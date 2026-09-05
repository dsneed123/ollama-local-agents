"""Web agent: searches, reads, compares and summarises online sources."""

from __future__ import annotations

from local_agents.agents.base import BaseAgent
from local_agents.agents.registry import register_agent


@register_agent(name="web", description="Search the web, compare sources, and summarize")
class WebAgent(BaseAgent):
    """Retrieve and analyse web content."""

    model_role = "general"
    default_tool_groups = ("web",)
    tool_first = True
    system_prompt = """You are a web research assistant with search and fetch tools.

- Always web_search before answering, even for terms you do not recognise; never ask
  for clarification before searching. Then fetch_url the most relevant 1-3 results.
- Compare what different sources say; note agreement and disagreement.
- Prefer recent, authoritative sources; mention publication dates when visible.
- Summarise concisely with a short source list (title + URL).
- If a page cannot be fetched, say so and try an alternative source.
- Never fabricate URLs or quotes."""
