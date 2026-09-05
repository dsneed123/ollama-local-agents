"""Generate a ready-to-run agent package."""

from __future__ import annotations

import keyword
import re
from pathlib import Path

from local_agents.errors import LocalAgentsError
from local_agents.scaffold import templates
from local_agents.tools.factory import TOOL_GROUPS

_NAME_RE = re.compile(r"[^a-z0-9]+")


def suggest_agent_name(raw: str) -> str:
    """Normalise user input into a valid agent name (``my-agent``)."""
    name = _NAME_RE.sub("-", raw.strip().lower()).strip("-")
    if not name:
        raise ValueError(f"Cannot derive an agent name from {raw!r}")
    return name


def class_name_for(name: str) -> str:
    parts = [p for p in re.split(r"[-_\s]+", name) if p]
    camel = "".join(p[:1].upper() + p[1:] for p in parts)
    if camel[:1].isdigit():
        camel = "Agent" + camel
    return camel if camel.endswith("Agent") else camel + "Agent"


def module_name_for(name: str) -> str:
    module = re.sub(r"[^a-z0-9_]", "_", name.lower())
    if module[:1].isdigit() or keyword.iskeyword(module):
        module = f"agent_{module}"
    return module


def create_agent_scaffold(
    name: str,
    *,
    directory: Path | str = "agents",
    description: str | None = None,
    tool_groups: tuple[str, ...] = ("filesystem",),
    model_role: str = "general",
    force: bool = False,
) -> list[Path]:
    """Create ``<directory>/<module>/`` with agent, tools, config, README and tests.

    Returns the created file paths (``agent.py`` first).
    """
    name = suggest_agent_name(name)
    for group in tool_groups:
        if group not in TOOL_GROUPS:
            raise LocalAgentsError(
                f"Unknown tool group {group!r}", hint=f"Valid groups: {', '.join(TOOL_GROUPS)}"
            )
    description = (description or f"{name.replace('-', ' ').title()} agent").replace('"', "'")
    class_name = class_name_for(name)
    module = module_name_for(name)
    root = Path(directory) / module
    if root.exists() and any(root.iterdir()) and not force:
        raise LocalAgentsError(
            f"{root} already exists", hint="Choose another name or pass --force to overwrite."
        )
    context = {
        "name": name,
        "title": name.replace("-", " ").title(),
        "description": description,
        "class_name": class_name,
        "tool_class_name": "CurrentTimeTool",
        "tool_groups": tuple(tool_groups),
        "model_role": model_role,
        "agent_dir": str(root),
    }
    files = {
        root / "agent.py": templates.AGENT_PY,
        root / "tools.py": templates.TOOLS_PY,
        root / "config.yaml": templates.CONFIG_YAML,
        root / "README.md": templates.README_MD,
        root / "__init__.py": templates.INIT_PY,
        root / "tests" / "__init__.py": "",
        root / "tests" / "test_agent.py": templates.TEST_PY,
    }
    created: list[Path] = []
    for path, template in files.items():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(template.format(**context), encoding="utf-8")
        created.append(path)
    return created
