# local-agents

Run AI agents on your own machine with [Ollama](https://ollama.com). No API keys, no
config files to start. It finds the models you have installed and picks a sensible one
for each job.

```
$ local-agents

Ollama: ● Connected   http://localhost:11434
Models: 7 detected

Which agent would you like to deploy?

  Research & Knowledge
   1  Research Agent      Research questions and synthesize findings
   2  Web Agent           Search the web, compare sources, and summarize
   3  Planning Agent      Break big tasks into plans and delegate to agents

  Software Development
   4  Coding Agent        Inspect, edit, and test code in the workspace
   5  Review Agent        Review code changes for bugs, risks, and style
   ...
```

## Install

Needs Python 3.10+ and Ollama running with at least one model (`ollama pull qwen3:8b`).

```bash
git clone https://github.com/dsneed123/local-agents.git
cd local-agents
python3 -m venv .venv && source .venv/bin/activate
pip install -e .
local-agents
```

`python main.py` does the same thing without installing.

## Usage

```bash
local-agents                          # interactive menu
local-agents list                     # every agent, grouped
local-agents models                   # installed models and which one each role gets
local-agents run coding "add type hints to utils.py"
local-agents run sql "top 5 customers by revenue in shop.db" --quiet
local-agents create-agent my-agent    # scaffold your own
local-agents --debug                  # tracebacks and logging
```

Inside a session: `/model qwen3:8b` switches models, `/tools` lists tools, `/reset`
clears memory, `/back` returns to the menu.

## Agents

27 agents, all built from the same base class. A few highlights:

| | |
|---|---|
| **coding, review, test, debug** | read, search, edit and test code in a workspace; review and security are read-only |
| **git, docs, regex, api** | git history and commit messages, docs from source, a real regex tester, HTTP requests |
| **data, sql, logs, convert** | CSV/JSON statistics, read-only SQLite queries, log frequency tables, format conversion |
| **file, notes, system** | find and organise files, keep markdown notes, inspect disk/processes/env |
| **research, web, summarize, planning** | search and cite, summarise documents and pages, plan and delegate |
| **writer, translate, meeting** | drafting and editing, translation that keeps formatting, action items from notes |
| **chat, tutor, math, brainstorm** | quick answers, explanations, exact calculation, idea generation |

`local-agents list` shows all of them with descriptions.

## How models are chosen

On startup the installed models are queried once and scored per role: coder-tuned
models for coding, larger instruct models for general work, thinking models for
reasoning, the smallest for fast tasks. Models without native tool calling still work
through a text protocol. Pin a model per agent in the Agent Manager or in config.

## Safety

File tools are confined to a workspace (the current directory by default) and cannot
delete unless you turn that on. Destructive shell commands are blocked outright, and
the rest ask for confirmation. SQLite databases open read-only. This guards against
accidents; it is not a sandbox for untrusted code.

## Configuration

Optional. Drop a `local-agents.yaml` in the project or in `~/.config/local-agents/`.
See [local-agents.example.yaml](local-agents.example.yaml) for every option. Environment
variables (`OLLAMA_HOST`, `LOCAL_AGENTS_WORKSPACE`, `LOCAL_AGENTS_MODEL`) override the
file, and flags like `--host`, `--workspace`, `--model` override those.

## Use it from Python

```python
import asyncio
from local_agents import AgentRuntime

async def main():
    async with AgentRuntime() as runtime:
        agent = await runtime.create_agent("research")
        result = await agent.run("Compare SQLite and DuckDB for analytics")
        print(result.output)

asyncio.run(main())
```

A custom agent is a class with a prompt:

```python
from local_agents import BaseAgent, register_agent

@register_agent(name="reviewer", description="Reviews pull requests")
class Reviewer(BaseAgent):
    model_role = "coding"
    default_tool_groups = ("filesystem", "shell")
    system_prompt = "You review code for correctness first, style second. ..."
```

Put it in `agents/<name>/agent.py` (or let `create-agent` generate it) and it shows up
in the menu. Tools implement `BaseTool`, providers implement `BaseModel`; see
[docs/custom-agents.md](docs/custom-agents.md) and
[docs/architecture.md](docs/architecture.md).

## Development

```bash
pip install -e ".[dev]"
pytest                 # unit tests, Ollama is mocked
pytest -m integration  # against a running Ollama
ruff check src tests && mypy
```

MIT licensed.
