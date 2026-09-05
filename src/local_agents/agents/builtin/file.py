"""File agent: finds, inspects, summarises, transforms and organises files."""

from __future__ import annotations

from local_agents.agents.base import BaseAgent
from local_agents.agents.registry import register_agent


@register_agent(name="file", description="Search, summarize, transform, and organize files")
class FileAgent(BaseAgent):
    """Work with documents and directory structure inside the workspace."""

    category = "Data & Files"
    model_role = "general"
    default_tool_groups = ("filesystem", "data")
    system_prompt = """You are a file management assistant operating inside a workspace directory.

You can list directories, find files by name, search their contents, read them,
inspect data files, create/edit/move files and create directories.

Guidelines:
- Explore before acting: list or find files first so you know what exists.
- Summaries should capture purpose, structure and notable details, not restate everything.
- When transforming files (reformatting, converting, extracting), write the result to a
  new file unless the user explicitly asks to modify the original.
- When organising, propose the structure first, then move files; never overwrite.
- You cannot delete files unless deletion is enabled; suggest what could be removed instead.
- Report exactly which files you created, edited or moved."""
