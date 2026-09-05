"""Planning agent: decomposes large tasks and can delegate to other agents."""

from __future__ import annotations

from typing import Any

from local_agents.agents.base import BaseAgent
from local_agents.agents.registry import register_agent


@register_agent(name="planning", description="Break big tasks into plans and delegate to agents")
class PlanningAgent(BaseAgent):
    """Create structured execution plans and optionally execute them via delegation."""

    model_role = "reasoning"
    default_tool_groups = ("filesystem",)
    can_delegate = True
    system_prompt = """You are a planning agent. You turn large, vague goals into concrete, \
ordered plans.

For every request:
1. Clarify the goal in one sentence and list assumptions.
2. Break the work into 3-8 subtasks. For each give: id, title, what "done" looks like,
   dependencies, and which specialist agent should do it (if any).
3. Order the subtasks and flag risks or open questions.

You may delegate subtasks with the `delegate` tool when the user asks you to execute the plan,
or when a subtask is small, self-contained and clearly needs another agent's tools
(e.g. research for facts, data for analysis, coding for code changes). Give delegated
agents a complete, standalone task description with all necessary context. Integrate
their results into your final answer and credit which agent produced what.

If the user only wants a plan, do not delegate — just deliver the plan in this format:

Goal: ...
Assumptions: ...
Plan:
  1. [agent] Title — done when ...
  2. ...
Risks / open questions: ..."""

    def build_system_prompt(self, context: Any = None) -> str:
        prompt = super().build_system_prompt(context)
        if self.executor is not None and "delegate" in self.tools:
            agents = self.executor.available_agents()
            listing = "\n".join(f"- {name}: {desc}" for name, desc in agents if name != self.name)
            if listing:
                prompt += "\n\nAgents you can delegate to:\n" + listing
        return prompt
