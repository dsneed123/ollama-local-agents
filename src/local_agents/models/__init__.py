"""Model abstraction and providers."""

from local_agents.models.base import (
    BaseModel,
    GenerationOptions,
    GenerationResult,
    Message,
    ModelInfo,
    StreamChunk,
    ToolCall,
    ToolSpec,
    Usage,
)
from local_agents.models.ollama import OllamaModel, OllamaProvider, detect_ollama_host
from local_agents.models.selection import (
    ROLES,
    ModelRecommendations,
    ModelRequirements,
    ModelSelector,
    ScoredModel,
    select_model,
)

__all__ = [
    "ROLES",
    "BaseModel",
    "GenerationOptions",
    "GenerationResult",
    "Message",
    "ModelInfo",
    "ModelRecommendations",
    "ModelRequirements",
    "ModelSelector",
    "OllamaModel",
    "OllamaProvider",
    "ScoredModel",
    "StreamChunk",
    "ToolCall",
    "ToolSpec",
    "Usage",
    "detect_ollama_host",
    "select_model",
]
