"""Turn framework exceptions into friendly terminal messages."""

from __future__ import annotations

import traceback

from local_agents.cli.ui import Console
from local_agents.errors import (
    LocalAgentsError,
    ModelNotFoundError,
    NoModelsAvailableError,
    ProviderConnectionError,
)


def report_error(console: Console, exc: BaseException, *, debug: bool = False) -> None:
    """Print ``exc`` for a human. With ``debug`` the traceback follows."""
    if isinstance(exc, ProviderConnectionError):
        console.error("Could not connect to Ollama")
        console.print()
        console.print("Endpoint:")
        console.print(f"  {exc.endpoint}")
        console.print()
        console.print("Possible fix:")
        console.print("  Start Ollama (`ollama serve`) and try again.")
        console.print("  If it runs elsewhere, set OLLAMA_HOST=http://host:11434")
    elif isinstance(exc, NoModelsAvailableError):
        console.error("Ollama is running but has no models installed")
        console.print()
        console.print("Possible fix:")
        console.print("  ollama pull qwen3:8b        # general purpose")
        console.print("  ollama pull qwen2.5-coder:7b # coding")
    elif isinstance(exc, ModelNotFoundError):
        console.error(str(exc))
        if exc.available:
            console.print()
            console.print("Installed models:")
            for name in exc.available:
                console.print(f"  ● {name}")
        console.print()
        console.print(f"Possible fix:\n  ollama pull {exc.model}")
    elif isinstance(exc, LocalAgentsError):
        console.error(str(exc))
        if exc.hint:
            console.print()
            console.print("Hint:")
            console.print(f"  {exc.hint}")
    elif isinstance(exc, KeyboardInterrupt):
        console.print()
        console.warn("Interrupted")
    else:
        console.error(f"Unexpected error: {type(exc).__name__}: {exc}")
        if not debug:
            console.print(console.style.dim("Run with --debug for a full traceback."))
    if debug:
        console.print()
        if exc.__traceback__ is not None:
            console.print(console.style.dim("".join(traceback.format_exception(exc))))
        else:
            console.print(console.style.dim(f"{type(exc).__name__}: {exc}"))
