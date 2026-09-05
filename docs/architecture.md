# Architecture

local-agents is a thin, typed framework. This document describes how a request
flows through it and where to extend it.

## Request flow

```text
local-agents (CLI)                     Python API
      │                                     │
      ▼                                     ▼
AgentRuntime.create_agent("coding")  ◄──────┘
      │  1. load Settings (defaults → YAML → env → CLI flags)
      │  2. registry.discover(): builtins, entry points, ./agents/**
      │  3. resolve model: explicit > agents.<name>.model > default_model > auto
      │  4. ToolKit.build(agent.default_tool_groups + custom_tools + delegate)
      │  5. ConversationMemory(), AgentConfig
      ▼
BaseAgent.stream(task)
      │  build messages: system prompt (+ workspace, + text tool protocol if needed)
      │                  + memory + user task (+ context)
      ├─► model.stream(messages, tools) ──► OllamaProvider ──► POST /api/chat
      │        tokens / thinking / tool_calls
      ├─► ToolSet.call(name, args)
      │        permission check → approval hook → validate → execute → ToolResult
      │        (SecurityError → blocked result, never raised into the model loop)
      └─► repeat until no tool calls or max_iterations
             → AgentResult(output, tool_calls, usage, messages, stop_reason)
```

## Modules

| Module | Responsibility | Extend by |
| --- | --- | --- |
| `models.base` | `BaseModel`, `Message`, `ToolSpec`, `GenerationOptions`, `ModelInfo` | Implementing `BaseModel` for a new provider |
| `models.ollama` | `OllamaProvider` (HTTP, discovery cache, errors) and `OllamaModel` (bound, lazy `auto()`) | — |
| `models.selection` | `ModelSelector` role-based scoring | Adjusting `ModelRequirements` / hints |
| `tools.base` | `BaseTool`, `ToolParameter`, `ToolResult`, `ToolSet` with permissions and approval hook | Subclassing `BaseTool` |
| `tools.*` | filesystem, shell, web, http, data, sqlite, text, convert, calculator, forms tool groups; `ToolKit` factory | Adding a group to `ToolKit.group` |
| `security` | `Workspace` path boundary, `CommandPolicy`, `PermissionSet` | Extra patterns / allow-lists |
| `memory` | `BaseMemory`, `ConversationMemory`, `JSONFileMemory` | Implementing `BaseMemory` (e.g. vector store) |
| `agents.base` | `BaseAgent` loop, events, text tool protocol | Subclassing |
| `agents.registry` | `AgentRegistry`, `register_agent`, discovery | Directories / entry points |
| `agents.runtime` | `AgentRuntime` (wiring) + `AgentExecutor` + `DelegateTool` | — |
| `config` | dataclass `Settings`, YAML + env layering | New sections are dataclasses |
| `cli` | menu, session, managers, error rendering | New screens read the registry |
| `scaffold` | `create-agent` templates | Editing templates |

## Model provider contract

`BaseModel` is intentionally small:

- `generate(messages, tools, options) -> GenerationResult`
- `stream(messages, tools, options) -> AsyncIterator[StreamChunk]`
- `health_check() -> bool`
- `list_models(refresh=False) -> list[ModelInfo]`
- `supports_tools() -> bool` (default: from `ModelInfo.capabilities`)

`OllamaModel.auto(role)` defers model resolution to the first call so it can be
constructed synchronously. `OllamaProvider` caches `/api/tags` for the session;
`invalidate_cache()` / `list_models(refresh=True)` re-query.

## Tool calling

If the model advertises the `tools` capability, tool specs are sent as JSON
schema and calls come back structured. Otherwise the system prompt describes a
fenced `tool_call` protocol. Both paths converge on `parse_text_tool_calls`,
which also rescues models that write bare JSON tool calls into their text; the
streaming token gate keeps that JSON out of the terminal.

## Security boundaries

- `Workspace.resolve()` is the only way file tools turn user paths into real paths.
- `CommandPolicy.check()` runs before every shell command; `ShellTool` additionally
  requires a confirmation callback when the policy demands it.
- `ToolSet` hides tools whose permission is not granted and refuses calls to them.
- `AgentRuntime` caps delegation depth.

## Future directions the design leaves open

Remote providers (`BaseModel`), other local runtimes, vector memories
(`BaseMemory`), MCP tools (`BaseTool` wrappers), agent marketplaces (entry points),
orchestration (the `AgentExecutor` protocol), a GUI or API server (both would sit
where the CLI sits, on top of `AgentRuntime`), and Docker packaging.
