from __future__ import annotations

from local_agents.models.base import ModelInfo
from local_agents.models.selection import ModelRequirements, ModelSelector, select_model


def _m(
    name: str, size: str, caps: tuple[str, ...] = ("completion", "tools"), ctx: int = 32768
) -> ModelInfo:
    return ModelInfo(name=name, parameter_size=size, capabilities=caps, context_length=ctx)


MODELS = [
    _m("qwen3:8b", "8.2B", ("completion", "tools", "thinking")),
    _m("qwen2.5-coder:14b", "14.8B"),
    _m("llama3.1:8b", "8.0B", ctx=131072),
    _m("deepseek-r1:70b", "70.6B", ("completion", "thinking")),
    _m("nomic-embed-text", "137M", ("embedding",)),
    _m("qwen2.5:0.5b", "0.5B"),
]


def test_coding_prefers_coder_models() -> None:
    assert select_model(MODELS, "coding").name == "qwen2.5-coder:14b"


def test_general_penalises_coder_models() -> None:
    chosen = select_model(MODELS, "general", require_tools=True)
    assert chosen.name in ("qwen3:8b", "llama3.1:8b")
    assert "coder" not in chosen.name


def test_fast_prefers_small_models() -> None:
    assert select_model(MODELS, "fast").name == "qwen2.5:0.5b"


def test_reasoning_without_tool_requirement_can_pick_thinking_model() -> None:
    assert select_model(MODELS, "reasoning").name == "deepseek-r1:70b"


def test_reasoning_with_tools_falls_back_to_tool_capable() -> None:
    chosen = select_model(MODELS, "reasoning", require_tools=True)
    assert chosen.supports_tools
    assert chosen.name == "qwen3:8b"


def test_embedding_models_are_excluded() -> None:
    selector = ModelSelector(MODELS)
    assert all("embed" not in s.model.name for s in selector.rank())


def test_prefer_list_wins_when_installed() -> None:
    chosen = select_model(MODELS, "coding", prefer=["missing:1b", "llama3.1"])
    assert chosen.name == "llama3.1:8b"


def test_falls_back_when_no_specialised_model() -> None:
    only = [_m("llama3.1:8b", "8.0B")]
    assert select_model(only, "coding").name == "llama3.1:8b"
    no_tools = [_m("plain:7b", "7B", ("completion",))]
    assert select_model(no_tools, "general", require_tools=True).name == "plain:7b"


def test_empty_returns_none() -> None:
    assert select_model([], "general") is None
    assert ModelSelector([]).recommend().by_role == {}


def test_find_by_base_name_and_latest() -> None:
    selector = ModelSelector([_m("mistral:latest", "7B"), _m("qwen3:8b", "8B")])
    assert selector.find("mistral").name == "mistral:latest"
    assert selector.find("qwen3").name == "qwen3:8b"
    assert selector.find("qwen3:8b").name == "qwen3:8b"
    assert selector.find("qwen3:70b") is None


def test_recommend_covers_roles() -> None:
    recs = ModelSelector(MODELS).recommend()
    assert set(recs.by_role) == {"general", "coding", "reasoning", "fast"}
    assert recs.get("coding").name == "qwen2.5-coder:14b"
    assert recs.ranked["coding"][0].reasons


def test_min_context_filter() -> None:
    chosen = ModelSelector(MODELS).select(ModelRequirements(role="general", min_context=100_000))
    assert chosen.name in ("llama3.1:8b", "deepseek-r1:70b")


def test_huge_models_do_not_automatically_win() -> None:
    models = [_m("giant:405b", "405B"), _m("mid:32b", "32.8B")]
    chosen = select_model(models, "general")
    assert chosen.name == "mid:32b"  # size score is capped; newer/shorter name tie-breaks


def test_parameters_billions_parsing() -> None:
    assert ModelInfo(name="x", parameter_size="137M").parameters_billions == 0.137
    assert ModelInfo(name="x", parameter_size="").parameters_billions == 0.0
    assert ModelInfo(name="x", parameter_size="garbage").parameters_billions == 0.0
