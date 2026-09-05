"""``local-agents`` command-line entry point.

The CLI is a thin layer over :class:`~local_agents.agents.runtime.AgentRuntime`;
nothing here is required to use the framework from Python.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
from collections.abc import Sequence
from pathlib import Path

from local_agents import __version__
from local_agents.agents.runtime import AgentRuntime
from local_agents.cli.errors import report_error
from local_agents.cli.menu import MainMenu
from local_agents.cli.session import AgentSession
from local_agents.cli.ui import Console, format_size
from local_agents.config.settings import Settings, load_settings
from local_agents.errors import LocalAgentsError, ProviderConnectionError
from local_agents.models.selection import ROLES, ModelSelector
from local_agents.scaffold.generator import create_agent_scaffold, suggest_agent_name
from local_agents.tools.base import BaseTool
from local_agents.tools.factory import TOOL_GROUPS

PROG = "local-agents"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog=PROG,
        description=(
            "Local AI agents on top of Ollama. Run without arguments for the interactive menu."
        ),
    )
    parser.add_argument("--version", action="version", version=f"{PROG} {__version__}")
    parser.add_argument("--debug", action="store_true", help="verbose logging and full tracebacks")
    parser.add_argument("--config", metavar="PATH", help="path to a local-agents.yaml config file")
    parser.add_argument(
        "--host",
        metavar="URL",
        help="Ollama endpoint (default: $OLLAMA_HOST or http://localhost:11434)",
    )
    parser.add_argument(
        "--workspace",
        metavar="DIR",
        help="directory agents may read and write (default: current directory)",
    )
    parser.add_argument("--model", metavar="NAME", help="force this Ollama model for every agent")
    parser.add_argument("--no-color", action="store_true", help="disable ANSI colours")

    sub = parser.add_subparsers(dest="command")

    run = sub.add_parser("run", help="run one agent on a task and exit")
    run.add_argument("agent", help="agent name (see `local-agents list`)")
    run.add_argument("task", nargs="+", help="the task to perform")
    run.add_argument("--quiet", action="store_true", help="print only the final answer")
    run.add_argument("--json", action="store_true", help="print the result as JSON")
    run.add_argument(
        "--auto-approve",
        action="store_true",
        help="approve shell commands without asking (non-interactive use; be careful)",
    )

    sub.add_parser("list", help="list available agents")
    models = sub.add_parser("models", help="show installed Ollama models and recommendations")
    models.add_argument("--refresh", action="store_true", help="ignore cached discovery")
    models.add_argument("--json", action="store_true")

    create = sub.add_parser("create-agent", help="scaffold a new agent under ./agents/")
    create.add_argument("name", help="agent name, e.g. my-agent")
    create.add_argument("--description", default=None, help="one-line description")
    create.add_argument("--dir", default="agents", help="parent directory (default: ./agents)")
    create.add_argument(
        "--tools",
        default="filesystem",
        help=f"comma-separated tool groups ({','.join(TOOL_GROUPS)})",
    )
    create.add_argument("--force", action="store_true", help="overwrite existing files")

    sub.add_parser("config", help="print the effective configuration")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    console = Console(color=False if args.no_color else None)
    debug = bool(args.debug)
    logging.basicConfig(
        level=logging.DEBUG if debug else logging.WARNING,
        format="%(levelname)s %(name)s: %(message)s",
    )
    try:
        settings = _load_settings(args)
        return asyncio.run(_dispatch(args, settings, console))
    except KeyboardInterrupt:
        console.print()
        return 130
    except LocalAgentsError as exc:
        report_error(console, exc, debug=debug)
        return 1
    except Exception as exc:
        report_error(console, exc, debug=debug)
        return 1


def _load_settings(args: argparse.Namespace) -> Settings:
    overrides: dict[str, object] = {}
    if args.host:
        overrides["ollama"] = {"host": args.host}
    if args.workspace:
        overrides["workspace"] = {"root": args.workspace}
    if args.model:
        overrides["default_model"] = args.model
    if args.debug:
        overrides["debug"] = True
    return load_settings(args.config, overrides=overrides)


async def _dispatch(args: argparse.Namespace, settings: Settings, console: Console) -> int:
    if args.command == "create-agent":
        return _cmd_create_agent(args, console)
    if args.command == "config":
        console.print(json.dumps(settings.to_dict(), indent=2, default=str))
        if settings.config_path:
            console.print(console.style.dim(f"# loaded from {settings.config_path}"))
        return 0

    interactive = sys.stdin.isatty()

    async def confirm_command(command: str) -> bool:
        if args.command == "run" and args.auto_approve:
            return True
        if not interactive:
            console.warn(f"Refusing to run command without a terminal: {command}")
            console.print(console.style.dim("Use --auto-approve to allow commands in scripts."))
            return False
        console.print()
        console.print(console.style.yellow("The agent wants to run a command:"))
        console.print(f"  $ {command}")
        return console.confirm("Allow?")

    async def approve_write(tool: BaseTool, arguments: dict[str, object]) -> bool:
        if args.command == "run" and args.auto_approve:
            return True
        if not interactive:
            return False
        target = arguments.get("path") or arguments.get("destination") or ""
        return console.confirm(f"Allow {tool.name} on {target}?")

    async with AgentRuntime(settings, confirm=confirm_command, approval=approve_write) as runtime:
        runtime.discover()
        if args.command == "run":
            return await _cmd_run(args, runtime, console)
        if args.command == "list":
            return _cmd_list(runtime, console)
        if args.command == "models":
            return await _cmd_models(args, runtime, console)
        return await MainMenu(runtime, console).run()


async def _cmd_run(args: argparse.Namespace, runtime: AgentRuntime, console: Console) -> int:
    task = " ".join(args.task)
    agent = await runtime.create_agent(args.agent)
    if args.json or args.quiet:
        result = await agent.run(task)
        if args.json:
            payload = {
                "agent": result.agent,
                "model": result.model,
                "output": result.output,
                "iterations": result.iterations,
                "stop_reason": result.stop_reason,
                "tool_calls": [
                    {"name": r.name, "arguments": r.arguments, "success": r.result.success}
                    for r in result.tool_calls
                ],
                "usage": result.usage.__dict__,
            }
            console.print(json.dumps(payload, indent=2, default=str))
        else:
            console.print(result.output)
        return 0 if result.stop_reason == "final" else 2
    session = AgentSession(runtime, agent, console, show_thinking=runtime.settings.debug)
    final = await session.run_task(task)
    return 0 if final else 2


def _cmd_list(runtime: AgentRuntime, console: Console) -> int:
    console.print("Available agents:")
    for category, regs in runtime.registry.grouped():
        console.print()
        console.print(console.style.dim(f"  {category}"))
        for reg in regs:
            enabled = runtime.settings.agent(reg.name).enabled
            marker = "●" if enabled else "○"
            console.print(f"  {marker} {reg.name:<12} {reg.description}")
    if runtime.discovery and runtime.discovery.errors:
        console.print()
        for path, error in runtime.discovery.errors:
            console.warn(f"{path}: {error}")
    return 0


async def _cmd_models(args: argparse.Namespace, runtime: AgentRuntime, console: Console) -> int:
    status = await runtime.status(refresh=args.refresh)
    if not status.connected:
        report_error(console, _connection_error(runtime), debug=runtime.settings.debug)
        return 1
    if args.json:
        console.print(
            json.dumps(
                {
                    "host": status.host,
                    "version": status.version,
                    "models": [m.__dict__ for m in status.models],
                    "recommended": {
                        role: (m.name if (m := status.recommendations.get(role)) else None)
                        for role in ROLES
                    }
                    if status.recommendations
                    else {},
                },
                indent=2,
                default=str,
            )
        )
        return 0
    console.print(f"Ollama {status.version or ''} at {status.host}")
    console.print()
    console.print("Detected Ollama models:")
    console.print()
    for model in status.models:
        caps = ",".join(model.capabilities)
        console.print(
            f"  ● {model.name:<28} {model.parameter_size:>6}  {format_size(model.size_bytes):>9}  "
            f"{console.style.dim(caps)}"
        )
    if status.models:
        console.print()
        console.print("Recommended:")
        selector = ModelSelector(status.models)
        for role in ROLES:
            chosen = selector.recommend([role]).get(role)
            if chosen:
                console.print(f"  {role.title():<10}→ {chosen.name}")
    else:
        console.warn("No models installed. Try `ollama pull qwen3:8b`.")
    return 0


def _cmd_create_agent(args: argparse.Namespace, console: Console) -> int:
    name = suggest_agent_name(args.name)
    groups = tuple(g.strip() for g in args.tools.split(",") if g.strip())
    created = create_agent_scaffold(
        name,
        directory=Path(args.dir),
        description=args.description,
        tool_groups=groups,
        force=args.force,
    )
    console.success(f"Created agent '{name}'")
    for path in created:
        console.print(f"  {path}")
    console.print()
    console.print("Next steps:")
    console.print(
        f"  1. Edit {created[0].parent / 'agent.py'} to customise the system prompt and tools"
    )
    console.print(f"  2. Run `{PROG}` — the agent appears in the menu automatically")
    console.print(f"  3. Run `pytest {created[0].parent / 'tests'}` to test it without Ollama")
    return 0


def _connection_error(runtime: AgentRuntime) -> LocalAgentsError:
    return ProviderConnectionError(runtime.provider.host)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
