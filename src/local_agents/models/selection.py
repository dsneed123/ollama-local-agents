"""Automatic model selection.

Given the models actually installed, pick the most appropriate one for a
*role*. Selection is heuristic and intentionally conservative: it never
assumes a model exists and always falls back to *something* installed.

Roles:

``coding``
    Prefer coder-tuned models (``*coder*``, ``codestral``, ``starcoder`` ...).
``general`` / ``reasoning``
    Prefer larger general-purpose instruct models.
``fast``
    Prefer the smallest capable model.

Callers may require capabilities (e.g. ``tools``); if no installed model
satisfies them the requirement is relaxed rather than failing outright, so a
user with a single non-tool model still gets a working agent.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field

from local_agents.models.base import ModelInfo

ModelRole = str

ROLES: tuple[ModelRole, ...] = ("general", "coding", "reasoning", "fast")

CODER_HINTS = (
    "coder",
    "code",
    "codestral",
    "starcoder",
    "codegemma",
    "codellama",
    "deepseek-coder",
)
REASONING_HINTS = ("r1", "reason", "think", "qwq", "o1")
TINY_HINTS = ("tiny", "mini", "small", "0.5b", "1b", "1.5b", "2b", "3b")
EXCLUDED_HINTS = ("embed", "bge", "minilm", "nomic", "rerank", "whisper", "clip")


@dataclass
class ModelRequirements:
    """What an agent needs from a model."""

    role: ModelRole = "general"
    require_tools: bool = False
    min_context: int | None = None
    prefer: Sequence[str] = ()
    #: Soft upper bound (billions of parameters). Larger models still qualify
    #: but stop gaining points, so a 70B model does not automatically win.
    preferred_max_params: float = 35.0


@dataclass
class ScoredModel:
    model: ModelInfo
    score: float
    reasons: list[str] = field(default_factory=list)


@dataclass
class ModelRecommendations:
    """Role -> chosen model, plus the ranked candidates for display."""

    by_role: dict[ModelRole, ModelInfo]
    ranked: dict[ModelRole, list[ScoredModel]]

    def get(self, role: ModelRole) -> ModelInfo | None:
        return self.by_role.get(role)


class ModelSelector:
    """Score installed models against requirements and pick the best."""

    def __init__(self, models: Iterable[ModelInfo]) -> None:
        self._models = [m for m in models if not _is_excluded(m)]
        self._all = list(models)

    @property
    def models(self) -> list[ModelInfo]:
        return list(self._models)

    def rank(self, requirements: ModelRequirements | None = None) -> list[ScoredModel]:
        req = requirements or ModelRequirements()
        candidates = self._models or self._all
        if req.require_tools:
            tool_capable = [m for m in candidates if m.supports_tools]
            if tool_capable:
                candidates = tool_capable
        if req.min_context:
            roomy = [m for m in candidates if (m.context_length or 0) >= req.min_context]
            if roomy:
                candidates = roomy
        scored = [self._score(m, req) for m in candidates]
        scored.sort(key=lambda s: (-s.score, s.model.name))
        return scored

    def select(self, requirements: ModelRequirements | None = None) -> ModelInfo | None:
        """Return the best model for ``requirements`` or ``None`` when nothing is installed."""
        req = requirements or ModelRequirements()
        for wanted in req.prefer:
            match = self.find(wanted)
            if match is not None:
                return match
        ranked = self.rank(req)
        return ranked[0].model if ranked else None

    def find(self, name: str) -> ModelInfo | None:
        """Find a model by exact name, or by base name if no tag was given."""
        for model in self._all:
            if model.name == name:
                return model
        if ":" not in name:
            for model in self._all:
                if model.base_name == name:
                    return model
            for model in self._all:
                if model.name == f"{name}:latest":
                    return model
        return None

    def recommend(self, roles: Iterable[ModelRole] = ROLES) -> ModelRecommendations:
        by_role: dict[ModelRole, ModelInfo] = {}
        ranked: dict[ModelRole, list[ScoredModel]] = {}
        for role in roles:
            scored = self.rank(ModelRequirements(role=role, require_tools=role != "reasoning"))
            ranked[role] = scored
            if scored:
                by_role[role] = scored[0].model
        return ModelRecommendations(by_role=by_role, ranked=ranked)

    # ------------------------------------------------------------------ #

    def _score(self, model: ModelInfo, req: ModelRequirements) -> ScoredModel:
        reasons: list[str] = []
        name = model.name.lower()
        params = model.parameters_billions
        score = 0.0

        # Size: diminishing returns, capped at preferred_max_params.
        if params > 0:
            effective = min(params, req.preferred_max_params)
            size_score = math.log2(effective + 1) * 10
            if req.role == "fast":
                size_score = -math.log2(params + 1) * 10
                reasons.append(f"small ({model.parameter_size})")
            else:
                reasons.append(f"{model.parameter_size}")
                if params > req.preferred_max_params:
                    # Very large models are slower on local hardware; only pick them
                    # when nothing comparable exists.
                    size_score -= math.log2(params / req.preferred_max_params) * 5
                    reasons.append("very large")
            score += size_score

        is_coder = any(h in name for h in CODER_HINTS)
        is_reasoner = any(h in name for h in REASONING_HINTS) or model.supports_thinking

        if req.role == "coding":
            if is_coder:
                score += 40
                reasons.append("coder-tuned")
        else:
            if is_coder:
                score -= 15
                reasons.append("coder-tuned (less general)")

        if req.role == "reasoning" and is_reasoner:
            score += 20
            reasons.append("reasoning-capable")
        elif req.role == "general" and model.supports_thinking:
            score += 3

        if model.supports_tools:
            score += 8
            reasons.append("tools")
        elif req.require_tools:
            score -= 20
            reasons.append("no native tool calling")

        if model.is_vision_model:
            score -= 5

        if model.context_length:
            score += min(math.log2(model.context_length / 4096 + 1), 4)

        # Deterministic tie-break: newer modification wins by a hair.
        score += 0.001 * len(model.modified_at)
        return ScoredModel(model=model, score=round(score, 3), reasons=reasons)


def _is_excluded(model: ModelInfo) -> bool:
    if model.is_embedding_model:
        return True
    name = model.name.lower()
    return any(h in name for h in EXCLUDED_HINTS) and "completion" not in model.capabilities


def select_model(
    models: Iterable[ModelInfo],
    role: ModelRole = "general",
    *,
    require_tools: bool = False,
    prefer: Sequence[str] = (),
) -> ModelInfo | None:
    """Convenience wrapper around :class:`ModelSelector`."""
    return ModelSelector(models).select(
        ModelRequirements(role=role, require_tools=require_tools, prefer=prefer)
    )
