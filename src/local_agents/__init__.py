"""local-agents: a modular local AI agent platform built on Ollama.

Quick start::

    from local_agents import AgentRuntime

    runtime = AgentRuntime()                     # detects Ollama + models
    agent = await runtime.create_agent("research")
    result = await agent.run("Summarise the pros and cons of SQLite for embedded apps")

Or wire an agent by hand::

    from local_agents import BaseAgent, OllamaModel, register_agent

    @register_agent(name="greeter", description="Says hello")
    class Greeter(BaseAgent):
        system_prompt = "Greet the user warmly in one sentence."

    agent = Greeter(model=OllamaModel.auto())
    print(await agent.run("Hi!"))
"""

from local_agents.agents.base import (
    AgentCapabilities,
    AgentConfig,
    AgentEvent,
    AgentExecutor,
    AgentResult,
    BaseAgent,
)
from local_agents.agents.builtin import (
    CodingAgent,
    DataAgent,
    FileAgent,
    PlanningAgent,
    ResearchAgent,
    WebAgent,
)
from local_agents.agents.registry import AgentRegistry, get_registry, register_agent
from local_agents.agents.runtime import AgentRuntime, RuntimeStatus
from local_agents.config.settings import Settings, load_settings
from local_agents.errors import (
    AgentError,
    AgentNotFoundError,
    CommandBlockedError,
    ConfigError,
    LocalAgentsError,
    ModelError,
    ModelNotFoundError,
    NoModelsAvailableError,
    PermissionDeniedError,
    ProviderConnectionError,
    SecurityError,
    ToolError,
    WorkspaceViolationError,
)
from local_agents.memory.base import BaseMemory
from local_agents.memory.conversation import ConversationMemory
from local_agents.memory.persistent import JSONFileMemory
from local_agents.models.base import (
    BaseModel,
    GenerationOptions,
    GenerationResult,
    Message,
    ModelInfo,
    StreamChunk,
    ToolCall,
    ToolSpec,
)
from local_agents.models.ollama import OllamaModel, OllamaProvider
from local_agents.models.selection import ModelRequirements, ModelSelector, select_model
from local_agents.security.commands import CommandPolicy
from local_agents.security.permissions import Permission, PermissionSet
from local_agents.security.workspace import Workspace
from local_agents.tools.base import BaseTool, ToolParameter, ToolResult, ToolSet
from local_agents.tools.factory import ToolKit

__version__ = "0.1.0"

__all__ = [
    "AgentCapabilities",
    "AgentConfig",
    "AgentError",
    "AgentEvent",
    "AgentExecutor",
    "AgentNotFoundError",
    "AgentRegistry",
    "AgentResult",
    "AgentRuntime",
    "BaseAgent",
    "BaseMemory",
    "BaseModel",
    "BaseTool",
    "CodingAgent",
    "CommandBlockedError",
    "CommandPolicy",
    "ConfigError",
    "ConversationMemory",
    "DataAgent",
    "FileAgent",
    "GenerationOptions",
    "GenerationResult",
    "JSONFileMemory",
    "LocalAgentsError",
    "Message",
    "ModelError",
    "ModelInfo",
    "ModelNotFoundError",
    "ModelRequirements",
    "ModelSelector",
    "NoModelsAvailableError",
    "OllamaModel",
    "OllamaProvider",
    "Permission",
    "PermissionDeniedError",
    "PermissionSet",
    "PlanningAgent",
    "ProviderConnectionError",
    "ResearchAgent",
    "RuntimeStatus",
    "SecurityError",
    "Settings",
    "StreamChunk",
    "ToolCall",
    "ToolError",
    "ToolKit",
    "ToolParameter",
    "ToolResult",
    "ToolSet",
    "ToolSpec",
    "WebAgent",
    "Workspace",
    "WorkspaceViolationError",
    "__version__",
    "get_registry",
    "load_settings",
    "register_agent",
    "select_model",
]
