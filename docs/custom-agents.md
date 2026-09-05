# Writing custom agents

This guide covers everything `local-agents create-agent` generates and how to go
further: custom tools, per-agent configuration, delegation, and testing.

## 1. Scaffold

```bash
local-agents create-agent ticket-triage --description "Triages support tickets" --tools filesystem,web
```

```text
agents/ticket_triage/
├── agent.py
├── tools.py
├── config.yaml
├── README.md
└── tests/test_agent.py
```

`local-agents` scans `./agents/` on startup (plus `agent_paths` from config and
`LOCAL_AGENTS_PATH`). Each `agents/<name>/agent.py` is imported as a package, so
`from .tools import ...` works.

## 2. The agent class

```python
from local_agents import BaseAgent, BaseTool, Workspace, register_agent
from .tools import LookupTicketTool

@register_agent(name="ticket-triage", description="Triages support tickets")
class TicketTriageAgent(BaseAgent):
    model_role = "general"                 # general | coding | reasoning | fast
    default_tool_groups = ("filesystem", "web")
    can_delegate = False
    system_prompt = """You triage support tickets. ..."""

    @classmethod
    def custom_tools(cls, workspace: Workspace | None = None) -> list[BaseTool]:
        return [LookupTicketTool()]
```

Class attributes the runtime reads:

| Attribute | Purpose |
| --- | --- |
| `name`, `description` | Identity; set by the decorator |
| `system_prompt` | Model instructions |
| `model_role` | Drives automatic model selection |
| `requires_tool_calling` | When `True` (default) and the agent has tools, prefer models with native tool calling |
| `default_tool_groups` | Any of `filesystem`, `shell`, `web`, `http`, `data`, `sqlite`, `text`, `convert`, `calculator`, `forms` |
| `default_permissions` | A `PermissionSet` overriding the workspace-derived defaults |
| `can_delegate` | Adds a `delegate` tool wired to the `AgentRuntime` |
| `tool_first` | Nudge the model once if it answers without using any tool |
| `category`, `display_name` | Menu grouping and label |

Hooks you can override:

- `custom_tools(workspace)` – tools shipped with the agent.
- `build_system_prompt(context)` – add dynamic context to the prompt.
- `format_task(task, context)` – control how the user message is composed.

## 3. Per-agent configuration

`config.yaml` next to `agent.py` is applied automatically unless the main
configuration already has an entry for the agent:

```yaml
agent:
  enabled: true
  model: auto            # or qwen3:8b
  temperature: 0.3
  max_iterations: 12
  context_length: 16384
```

## 4. Custom tools

```python
from local_agents import BaseTool, ToolParameter, Permission

class LookupTicketTool(BaseTool):
    name = "lookup_ticket"
    description = "Fetch a ticket by id from the local ticket store."
    parameters = (ToolParameter("ticket_id", "string", "Ticket identifier"),)
    permission = Permission.READ

    async def execute(self, ticket_id: str) -> dict:
        return {"id": ticket_id, "status": "open"}      # any JSON-serialisable value
```

Rules:

- Declare every parameter; unknown arguments from the model are ignored,
  missing required ones become an error the model sees.
- Raise `local_agents.ToolError` (with an optional `hint`) for expected failures.
- Pick the narrowest `Permission`. `WRITE`/`DELETE` tools are subject to the
  optional approval prompt; `EXECUTE` tools should go through `CommandPolicy`.
- Tools that touch files should take a `Workspace` and call `workspace.resolve()`.

## 5. Delegation

```python
class ReleaseManagerAgent(BaseAgent):
    can_delegate = True
```

The runtime injects a `delegate(agent, task)` tool listing every other enabled
agent. Delegated runs use fresh agent instances and the same configuration;
depth is capped at 3 to prevent loops. Programmatically:

```python
result = await runtime.run("coding", "Bump the version in pyproject.toml")
```

## 6. Testing without Ollama

```python
from local_agents.testing import ScriptedModel, ScriptedTurn

model = ScriptedModel([
    ScriptedTurn.call("lookup_ticket", ticket_id="42"),   # first reply: a tool call
    "Ticket 42 is open.",                                  # second reply: the answer
])
agent = TicketTriageAgent(model=model, tools=TicketTriageAgent.custom_tools())
result = await agent.run("What is the status of ticket 42?")
assert result.output == "Ticket 42 is open."
assert result.tool_calls[0].name == "lookup_ticket"
```

`ScriptedModel` records every call in `model.calls` so you can assert on the
messages and tool specs the agent sent.

## 7. Packaging agents as plugins

Add an entry point to your package's `pyproject.toml`:

```toml
[project.entry-points."local_agents.agents"]
my_pack = "my_pack.agents"     # module that registers agents on import
```

Installed plugins are discovered automatically.
