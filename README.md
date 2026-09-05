# local-agents

A modular, extensible local AI agent platform built on [Ollama](https://ollama.com).

Clone it, have Ollama running, type `local-agents`, and deploy an agent against the
models you already have installed. No API keys, no configuration, no cloud.

```text
╔══════════════════════════════════════════════╗
║              LOCAL AGENT STACK               ║
║        Local AI • Modular • Extensible       ║
╚══════════════════════════════════════════════╝

Ollama: ● Connected (v0.32.9)   http://localhost:11434
Models: 7 detected

Detected Ollama models:
  ● qwen3:8b
  ● qwen2.5-coder:32b
  ● llama3.1:8b
  ...

Recommended:
  Coding    → qwen2.5-coder:32b
  General   → qwen2.5:14b

Which agent would you like to deploy?

   1  Research Agent    Research questions and synthesize findings
   2  Coding Agent      Inspect, edit, and test code in the workspace
   3  Web Agent         Search the web, compare sources, and summarize
   4  File Agent        Search, summarize, transform, and organize files
   5  Data Agent        Analyze CSV/JSON datasets and surface trends
   6  Planning Agent    Break big tasks into plans and delegate to agents
   7  Agent Manager
   8  Model Manager
   9  Create Agent
   q  Quit
```

The CLI is one interface on top of a small, typed Python framework. Everything the
CLI does is available from Python, and every piece (agents, tools, models, memory)
can be used on its own.

## Contents

- [Quick start](#quick-start)
- [Ollama setup](#ollama-setup)
- [Available agents](#available-agents)
- [Using the CLI](#using-the-cli)
- [Creating an agent](#creating-an-agent)
- [Python API](#python-api)
- [Creating custom tools](#creating-custom-tools)
- [Creating custom models](#creating-custom-models)
- [Configuration](#configuration)
- [Security model](#security-model)
- [Architecture](#architecture)
- [Development](#development)

## Quick start

```bash
git clone https://github.com/dsneed123/local-agents.git
cd local-agents
python3 -m venv .venv && source .venv/bin/activate
pip install -e .

local-agents
```

Or, without installing the package, use the startup script at the repository root:

```bash
python main.py            # same as `local-agents`; all arguments pass through
python main.py list
```

Requirements: Python 3.10+ and a running Ollama with at least one model pulled
(for example `ollama pull qwen3:8b`).

Non-interactive use:

```bash
local-agents list                                  # registered agents
local-agents models                                # installed models + recommendations
local-agents run research "Compare SQLite and DuckDB for analytics"
local-agents run coding "Add type hints to utils.py" --workspace ./myproject
local-agents run data "Which region has the highest revenue in sales.csv?" --json
local-agents create-agent my-agent                 # scaffold a custom agent
local-agents --debug                               # full tracebacks and logging
```

## Ollama setup

Nothing to configure. On startup local-agents:

1. Detects the Ollama endpoint (`--host`, then `OLLAMA_HOST`, then `http://localhost:11434`).
2. Checks that Ollama is reachable.
3. Queries the installed models once per session (refresh from the Model Manager).
4. Picks a sensible model for each agent based on what is installed.

Model selection is heuristic and never assumes a model exists:

| Agent role | Preference |
| --- | --- |
| `coding` | Coder-tuned models (`*coder*`, `codestral`, `starcoder`, ...) |
| `general` | Larger general-purpose instruct models with tool calling |
| `reasoning` | Thinking/reasoning models, still honouring tool-calling needs |
| `fast` | The smallest capable model |

If no specialised model exists the selector falls back to the best installed model.
Models without native tool calling still work: agents switch to a text-based
tool-call protocol automatically. Embedding-only models are never selected.

If Ollama is not running you get a clear message, not a traceback:

```text
✗ Could not connect to Ollama

Endpoint:
  http://localhost:11434

Possible fix:
  Start Ollama (`ollama serve`) and try again.
```

## Available agents

| Agent | Model role | Tools | What it does |
| --- | --- | --- | --- |
| **Research** | general | web, filesystem | Breaks a question into sub-questions, gathers evidence, cross-checks sources, and synthesises a cited answer. |
| **Coding** | coding | filesystem, shell | Inspects a repository, searches and reads code, makes targeted edits, runs tests, and explains changes. Confined to the workspace; destructive commands are blocked. |
| **Web** | general | web | Searches the web, fetches pages as readable text, compares sources, and summarises. |
| **File** | general | filesystem, data | Finds, inspects, summarises, transforms and organises files. Never deletes unless enabled. |
| **Data** | reasoning | data, filesystem | Inspects CSV/TSV/JSON/JSONL, computes statistics, filters/groups/aggregates, and explains trends. |
| **Planning** | reasoning | filesystem, delegate | Turns large goals into ordered plans and can delegate sub-tasks to the other agents. |

Every agent follows the same loop: build the prompt from memory and task, ask the
model, execute any tool calls, feed results back, repeat until the model answers.

## Using the CLI

Inside an agent session:

```text
/model [name]   Show or switch the model for this session
/models         List installed models
/tools          List the agent's tools
/info           Show agent capabilities
/think          Toggle display of model reasoning
/reset          Clear conversation memory
/back           Return to the main menu
```

**Model Manager** lists installed models with size and capabilities, refreshes the
list from Ollama, shows details for a model, and sets a session-wide default.

**Agent Manager** shows each agent's status, assigned model and tools, lets you
enable/disable agents, and pin a specific model per agent.

When the coding agent wants to run a shell command you are asked to approve it.
For scripted runs pass `--auto-approve` to `run` (use with care).

## Creating an agent

```bash
local-agents create-agent my-agent --description "Triages support tickets" --tools filesystem,web
```

generates:

```text
agents/my_agent/
├── agent.py        # the agent class (registered with @register_agent)
├── tools.py        # a working custom tool to build on
├── config.yaml     # per-agent defaults: model, temperature, max_iterations
├── README.md
└── tests/
    └── test_agent.py   # runs without Ollama using ScriptedModel
```

Run `local-agents` from the same directory and the new agent appears in the menu.
Agents are discovered from `./agents/`, from any directory listed in
`agent_paths` (or `LOCAL_AGENTS_PATH`), and from installed packages exposing the
`local_agents.agents` entry point.

A minimal agent is just a class:

```python
from local_agents import BaseAgent, register_agent

@register_agent(name="reviewer", description="Reviews code for bugs")
class ReviewerAgent(BaseAgent):
    model_role = "coding"                      # which model to auto-select
    default_tool_groups = ("filesystem",)      # filesystem | shell | web | data
    system_prompt = "You are a careful code reviewer. ..."
```

## Python API

```python
import asyncio
from local_agents import AgentRuntime

async def main() -> None:
    async with AgentRuntime() as runtime:          # loads config, detects Ollama
        agent = await runtime.create_agent("research")
        result = await agent.run("Summarise the trade-offs of SQLite for embedded apps")
        print(result.output)
        print(result.model, result.iterations, [c.name for c in result.tool_calls])

asyncio.run(main())
```

Or wire an agent by hand with no runtime at all:

```python
from local_agents import BaseAgent, OllamaModel, Workspace, register_agent
from local_agents.tools import filesystem_tools

@register_agent(name="summariser", description="Summarises files")
class Summariser(BaseAgent):
    system_prompt = "Summarise the requested file in three bullet points."

workspace = Workspace(root="./docs", allow_write=False)
agent = Summariser(
    model=OllamaModel.auto("fast"),            # resolved lazily on first use
    tools=filesystem_tools(workspace),
)
result = await agent.run("Summarise README.md")
```

Streaming:

```python
async for event in agent.stream("Explain this repo"):
    if event.type == "token":
        print(event.content, end="", flush=True)
    elif event.type == "tool_call":
        print(f"\n[calling {event.tool_call.name}]")
```

Event types: `start`, `thinking`, `token`, `message`, `tool_call`, `tool_result`,
`error`, `final` (carries the `AgentResult` in `event.metadata["result"]`).

Agent-to-agent calls go through the `AgentExecutor` interface, which
`AgentRuntime` implements:

```python
result = await runtime.run("data", "Describe sales.csv")   # fresh agent, same config
```

Agents with `can_delegate = True` receive a `delegate` tool that calls the executor.
Delegation depth is capped to prevent loops.

## Creating custom tools

```python
from local_agents import BaseTool, ToolParameter, Permission

class WordCountTool(BaseTool):
    name = "word_count"
    description = "Count the words in a piece of text."
    parameters = (
        ToolParameter("text", "string", "The text to count"),
        ToolParameter("unique", "boolean", "Count unique words only", required=False, default=False),
    )
    permission = Permission.READ      # READ | WRITE | DELETE | EXECUTE | NETWORK | DELEGATE

    async def execute(self, text: str, unique: bool = False) -> str:
        words = text.split()
        return str(len(set(words)) if unique else len(words))
```

Tools are plain objects: pass them to any agent (`tools=[WordCountTool()]`) or return
them from an agent's `custom_tools()` classmethod so the runtime injects them.
Arguments are validated and coerced against the declared parameters, errors become
structured results the model can react to, and the tool's `permission` is checked
against the agent's `PermissionSet` before every call.

## Creating custom models

Implement `BaseModel` to add a provider. Agents only depend on this interface.

```python
from local_agents import BaseModel, GenerationResult, ModelInfo, StreamChunk

class MyProviderModel(BaseModel):
    provider = "my-provider"

    @property
    def name(self) -> str: ...

    async def generate(self, messages, *, tools=None, options=None) -> GenerationResult: ...

    async def stream(self, messages, *, tools=None, options=None):
        yield StreamChunk(content="...")

    async def health_check(self) -> bool: ...

    async def list_models(self, *, refresh=False) -> list[ModelInfo]: ...
```

`local_agents.testing.ScriptedModel` is a ready-made implementation that plays back
canned replies, used throughout the test suite and in generated agent tests.

## Configuration

Configuration is optional. When present, local-agents reads `local-agents.yaml` from
the current directory or `~/.config/local-agents/`, or the file given by `--config` /
`LOCAL_AGENTS_CONFIG`. See [`local-agents.example.yaml`](local-agents.example.yaml).

```yaml
ollama:
  host: http://localhost:11434
  timeout: 120

workspace:
  root: .
  allow_write: true
  allow_delete: false

security:
  allow_shell: true
  confirm_shell: true
  confirm_writes: false
  blocked_commands: []        # extra regexes to block

agents:
  coding:
    model: auto               # or a specific model, e.g. qwen2.5-coder:14b
    max_iterations: 20
  web:
    enabled: false

agent_paths: ["./agents"]
default_model: auto
```

Environment variables override the file: `OLLAMA_HOST`, `LOCAL_AGENTS_WORKSPACE`,
`LOCAL_AGENTS_MODEL`, `LOCAL_AGENTS_PATH`, `LOCAL_AGENTS_DEBUG`, `LOCAL_AGENTS_CONFIG`.
CLI flags override everything. `local-agents config` prints the effective result.

## Security model

Agents run local code, so the defaults are conservative:

- **Workspace boundary.** All file tools resolve paths through a `Workspace`. Paths
  (including symlink targets) outside the root are rejected. The coding agent is told
  its workspace explicitly.
- **No deletion by default.** `delete_file` is only offered when `allow_delete` is on.
- **Command policy.** A block-list rejects destructive shell commands (`rm -rf`, `mkfs`,
  `git push --force`, `git reset --hard`, `curl | sh`, `sudo`, ...). An optional
  allow-list restricts commands to known prefixes.
- **Confirmation.** Shell commands require approval unless `confirm_shell` is off or
  `--auto-approve` is passed. Writes can be gated the same way with `confirm_writes`.
- **Permissions.** Every tool declares a permission; the agent's `PermissionSet` decides
  which tools the model can even see.
- **Timeouts** on model calls and shell commands; blocked actions are reported to both
  the user and the model, never silently executed.

This is a guard rail against accidents, not a sandbox. Run untrusted workloads in a container.

## Architecture

```text
                     ┌───────────────────┐
                     │   Terminal CLI    │  local_agents.cli
                     └─────────┬─────────┘
                               │
                     ┌─────────▼─────────┐
                     │   Agent Runtime   │  local_agents.agents.runtime
                     │ (config, registry,│  builds agents, implements AgentExecutor
                     │  tool kit, models)│
                     └─────────┬─────────┘
                               │
              ┌────────────────┼────────────────┐
              │                │                │
        ┌─────▼─────┐    ┌─────▼─────┐    ┌────▼────┐
        │  Agents   │    │   Tools   │    │ Memory  │
        │ BaseAgent │    │ BaseTool  │    │BaseMemory│
        │ registry  │    │ ToolSet   │    │Conversation│
        └─────┬─────┘    │ Workspace │    │JSONFile │
              │          └───────────┘    └─────────┘
        ┌─────▼─────┐
        │   Model   │  local_agents.models.base
        │ BaseModel │
        └─────┬─────┘
              │
        ┌─────▼─────┐
        │  Ollama   │  local_agents.models.ollama (provider + selection)
        └───────────┘
```

Package layout:

```text
src/local_agents/
├── agents/          BaseAgent, tool loop, registry, runtime, builtin/ reference agents
├── models/          BaseModel, OllamaProvider/OllamaModel, ModelSelector
├── tools/           BaseTool, ToolSet, filesystem, shell, web, data, ToolKit factory
├── memory/          BaseMemory, ConversationMemory, JSONFileMemory
├── security/        Workspace, CommandPolicy, PermissionSet
├── config/          typed Settings + YAML/env loading
├── cli/             menu, session, model manager, agent manager, error rendering
├── scaffold/        `create-agent` generator
├── testing.py       ScriptedModel for Ollama-free tests
└── errors.py        exception hierarchy with user-facing hints
```

Design rules that keep it extensible:

- Agents, tools, models and memory are dependency-injected; there are no globals.
- New providers implement `BaseModel`; new tool backends (search, MCP, vector stores)
  implement `BaseTool`; new agents subclass `BaseAgent` and register themselves.
- The CLI never contains agent-specific branches: the menu is generated from the registry.

## Development

```bash
pip install -e ".[dev]"

pytest                      # unit tests (mocked Ollama, no server needed)
pytest -m integration       # optional: tests against a running Ollama
ruff check src tests        # lint
ruff format src tests       # format
mypy                        # strict type checking of src/
```

Layout conventions: `src/` layout, `pytest-asyncio` in auto mode, 100-column lines,
strict mypy. Tests mock Ollama through `httpx.MockTransport` (see `tests/conftest.py`),
and agent behaviour is tested with `ScriptedModel`.

## License

MIT
