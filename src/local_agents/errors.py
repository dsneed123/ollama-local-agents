"""Exception hierarchy for local-agents.

Every error raised by the framework derives from :class:`LocalAgentsError` so
callers (and the CLI) can catch framework failures without swallowing
unrelated bugs. Each error carries an optional ``hint`` describing a likely fix,
which the CLI renders as a friendly message instead of a traceback.
"""

from __future__ import annotations

from pathlib import Path


class LocalAgentsError(Exception):
    """Base class for all framework errors."""

    def __init__(self, message: str, *, hint: str | None = None) -> None:
        super().__init__(message)
        self.message = message
        self.hint = hint

    def __str__(self) -> str:
        return self.message


class ConfigError(LocalAgentsError):
    """Raised when configuration cannot be loaded or is invalid."""


# --------------------------------------------------------------------------- #
# Model / provider errors
# --------------------------------------------------------------------------- #


class ModelError(LocalAgentsError):
    """Base class for model-provider failures."""


class ProviderConnectionError(ModelError):
    """The model provider could not be reached."""

    def __init__(self, endpoint: str, *, cause: BaseException | None = None) -> None:
        self.endpoint = endpoint
        self.cause = cause
        super().__init__(
            f"Could not connect to Ollama at {endpoint}",
            hint="Make sure Ollama is running (try `ollama serve`) and that the "
            "endpoint is correct. Set OLLAMA_HOST to override it.",
        )


class ModelNotFoundError(ModelError):
    """The requested model is not installed on the provider."""

    def __init__(self, model: str, available: list[str] | None = None) -> None:
        self.model = model
        self.available = available or []
        hint = f"Pull it with `ollama pull {model}`."
        if self.available:
            hint += " Installed models: " + ", ".join(self.available)
        super().__init__(f"Model '{model}' is not installed", hint=hint)


class NoModelsAvailableError(ModelError):
    """The provider is reachable but has no models installed."""

    def __init__(self, endpoint: str) -> None:
        self.endpoint = endpoint
        super().__init__(
            f"Ollama at {endpoint} has no models installed",
            hint="Pull a model first, for example `ollama pull qwen3:8b`.",
        )


class ModelTimeoutError(ModelError):
    """A generation request exceeded its timeout."""

    def __init__(self, model: str, timeout: float) -> None:
        self.model = model
        self.timeout = timeout
        super().__init__(
            f"Model '{model}' did not respond within {timeout:g}s",
            hint="Increase the timeout, or use a smaller/faster model.",
        )


class ModelResponseError(ModelError):
    """The provider returned an error or an unparseable response."""


# --------------------------------------------------------------------------- #
# Tool / security errors
# --------------------------------------------------------------------------- #


class ToolError(LocalAgentsError):
    """Base class for tool failures."""


class ToolNotFoundError(ToolError):
    """An agent asked for a tool that is not registered with it."""

    def __init__(self, name: str, available: list[str] | None = None) -> None:
        self.name = name
        self.available = available or []
        super().__init__(
            f"Tool '{name}' is not available",
            hint=("Available tools: " + ", ".join(self.available)) if self.available else None,
        )


class ToolValidationError(ToolError):
    """Tool arguments were missing or of the wrong type."""


class SecurityError(LocalAgentsError):
    """Base class for policy violations. Never silently swallowed."""


class WorkspaceViolationError(SecurityError):
    """A path escaped the configured workspace boundary."""

    def __init__(self, path: str, root: str) -> None:
        self.path = path
        self.root = root
        super().__init__(
            f"Path '{path}' is outside the workspace '{root}'",
            hint=(
                f"Use a path relative to the workspace root, e.g. '{Path(path).name}'. "
                "(The workspace can be widened with --workspace or `workspace.root`.)"
            ),
        )


class PermissionDeniedError(SecurityError):
    """A tool tried an operation its permissions do not allow."""


class CommandBlockedError(SecurityError):
    """A shell command matched the destructive-command policy."""

    def __init__(self, command: str, reason: str) -> None:
        self.command = command
        self.reason = reason
        super().__init__(f"Command blocked: {reason}", hint=f"Command was: {command}")


# --------------------------------------------------------------------------- #
# Agent errors
# --------------------------------------------------------------------------- #


class AgentError(LocalAgentsError):
    """Base class for agent failures."""


class AgentNotFoundError(AgentError):
    """No agent with the requested name is registered."""

    def __init__(self, name: str, available: list[str] | None = None) -> None:
        self.name = name
        self.available = available or []
        super().__init__(
            f"No agent named '{name}' is registered",
            hint=("Registered agents: " + ", ".join(self.available)) if self.available else None,
        )


class AgentRegistrationError(AgentError):
    """An agent could not be registered (duplicate name, bad class, ...)."""


class MaxIterationsError(AgentError):
    """The agent loop hit its iteration ceiling without finishing."""

    def __init__(self, agent: str, iterations: int) -> None:
        self.agent = agent
        self.iterations = iterations
        super().__init__(
            f"Agent '{agent}' stopped after {iterations} iterations without a final answer",
            hint="Raise `max_iterations` in the agent config, or simplify the task.",
        )
