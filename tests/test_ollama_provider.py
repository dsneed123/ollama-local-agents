from __future__ import annotations

import httpx
import pytest
from tests.conftest import FakeOllama

from local_agents.errors import (
    ModelNotFoundError,
    ModelResponseError,
    ModelTimeoutError,
    NoModelsAvailableError,
    ProviderConnectionError,
)
from local_agents.models.base import GenerationOptions, Message, ToolSpec
from local_agents.models.ollama import OllamaModel, OllamaProvider, detect_ollama_host


def test_detect_host_prefers_explicit_then_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OLLAMA_HOST", "0.0.0.0:2222")
    assert detect_ollama_host() == "http://localhost:2222"
    assert detect_ollama_host("http://x:1") == "http://x:1"
    monkeypatch.delenv("OLLAMA_HOST")
    assert detect_ollama_host() == "http://localhost:11434"


async def test_health_check_and_version(provider: OllamaProvider) -> None:
    assert await provider.health_check() is True
    assert await provider.version() == "0.32.9"


async def test_health_check_never_raises() -> None:
    fake = FakeOllama()
    fake.fail_with = httpx.ConnectError("refused")
    provider = OllamaProvider("http://down:11434", transport=fake.transport())
    assert await provider.health_check() is False


async def test_list_models_parses_and_caches(
    provider: OllamaProvider, fake_ollama: FakeOllama
) -> None:
    models = await provider.list_models()
    names = [m.name for m in models]
    assert names == [
        "qwen3:8b",
        "qwen2.5-coder:14b",
        "llama3.1:8b",
        "deepseek-r1:70b",
        "nomic-embed-text:latest",
    ]
    qwen = models[0]
    assert qwen.parameter_size == "8.2B"
    assert qwen.parameters_billions == pytest.approx(8.2)
    assert qwen.context_length == 40960
    assert qwen.supports_tools and qwen.supports_thinking
    assert models[-1].is_embedding_model
    tag_requests = [r for r in fake_ollama.requests if r.url.path == "/api/tags"]
    assert len(tag_requests) == 1
    await provider.list_models()
    assert len([r for r in fake_ollama.requests if r.url.path == "/api/tags"]) == 1
    await provider.list_models(refresh=True)
    assert len([r for r in fake_ollama.requests if r.url.path == "/api/tags"]) == 2


async def test_missing_capabilities_are_filled_from_show() -> None:
    tags = {
        "models": [
            {"name": "old:7b", "size": 1, "details": {"parameter_size": "7B"}},
        ]
    }
    fake = FakeOllama(tags)
    provider = OllamaProvider("http://o:11434", transport=fake.transport())
    models = await provider.list_models()
    assert models[0].capabilities == ()
    assert models[0].context_length == 8192  # from model_info in /api/show
    assert any(r.url.path == "/api/show" for r in fake.requests)


async def test_connection_error_is_friendly() -> None:
    fake = FakeOllama()
    fake.fail_with = httpx.ConnectError("refused")
    provider = OllamaProvider("http://down:11434", transport=fake.transport())
    with pytest.raises(ProviderConnectionError) as info:
        await provider.list_models()
    assert info.value.endpoint == "http://down:11434"
    assert "Ollama" in str(info.value)
    assert info.value.hint


async def test_no_models_error() -> None:
    fake = FakeOllama({"models": []})
    provider = OllamaProvider("http://o:11434", transport=fake.transport())
    with pytest.raises(NoModelsAvailableError):
        await provider.require_models()


async def test_chat_parses_tool_calls_and_usage(
    provider: OllamaProvider, fake_ollama: FakeOllama
) -> None:
    fake_ollama.queue(
        "Let me check.",
        tool_calls=[{"function": {"name": "read_file", "arguments": {"path": "a.txt"}}}],
    )
    result = await provider.chat("qwen3:8b", [Message.user("hi")])
    assert result.content == "Let me check."
    assert result.tool_calls[0].name == "read_file"
    assert result.tool_calls[0].arguments == {"path": "a.txt"}
    assert result.usage.prompt_tokens == 10 and result.usage.completion_tokens == 5


async def test_chat_payload_contains_options_and_tools(
    provider: OllamaProvider, fake_ollama: FakeOllama
) -> None:
    import json

    fake_ollama.queue("ok")
    spec = ToolSpec("t", "desc", {"type": "object", "properties": {}})
    await provider.chat(
        "qwen3:8b",
        [Message.system("s"), Message.user("u")],
        tools=[spec],
        options=GenerationOptions(temperature=0.2, context_length=4096, max_tokens=50, think=False),
    )
    body = json.loads(fake_ollama.requests[-1].content)
    assert body["stream"] is False
    assert body["options"] == {"temperature": 0.2, "num_ctx": 4096, "num_predict": 50}
    assert body["think"] is False
    assert body["tools"][0]["function"]["name"] == "t"
    assert [m["role"] for m in body["messages"]] == ["system", "user"]


async def test_stream_chat_yields_chunks(provider: OllamaProvider, fake_ollama: FakeOllama) -> None:
    fake_ollama.queue("Hello world", thinking="hmm")
    chunks = [c async for c in provider.stream_chat("qwen3:8b", [Message.user("hi")])]
    assert "".join(c.content for c in chunks) == "Hello world"
    assert "".join(c.thinking for c in chunks) == "hmm"
    assert chunks[-1].done and chunks[-1].usage is not None


async def test_model_not_found(provider: OllamaProvider) -> None:
    await provider.list_models()
    with pytest.raises(ModelNotFoundError) as info:
        await provider.chat("missing:1b", [Message.user("x")])
    assert "qwen3:8b" in info.value.available
    assert "ollama pull missing:1b" in (info.value.hint or "")


async def test_stream_model_not_found(provider: OllamaProvider) -> None:
    with pytest.raises(ModelNotFoundError):
        async for _ in provider.stream_chat("missing:1b", [Message.user("x")]):
            pass


async def test_server_error(fake_ollama: FakeOllama) -> None:
    fake_ollama.status_override = (500, {"error": "boom"})
    provider = OllamaProvider("http://o:11434", transport=fake_ollama.transport())
    with pytest.raises(ModelResponseError, match="HTTP 500"):
        await provider.list_models()


async def test_timeout_is_wrapped() -> None:
    fake = FakeOllama()
    fake.fail_with = httpx.ReadTimeout("slow")
    provider = OllamaProvider("http://o:11434", transport=fake.transport())
    with pytest.raises(ModelTimeoutError):
        await provider.chat("qwen3:8b", [Message.user("x")], options=GenerationOptions(timeout=1))


async def test_ollama_model_auto_resolves_lazily(
    provider: OllamaProvider, fake_ollama: FakeOllama
) -> None:
    model = OllamaModel.auto("coding", provider=provider)
    assert not model.is_resolved
    assert model.name == "auto:coding"
    info = await model.resolve()
    assert info.name == "qwen2.5-coder:14b"
    assert model.name == "qwen2.5-coder:14b"
    assert await model.supports_tools() is True


async def test_ollama_model_explicit_and_rebind(
    provider: OllamaProvider, fake_ollama: FakeOllama
) -> None:
    model = OllamaModel("llama3.1", provider=provider)  # base name resolves to tagged model
    assert (await model.resolve()).name == "llama3.1:8b"
    fake_ollama.queue("pong")
    result = await model.generate([Message.user("ping")])
    assert result.content == "pong" and result.model == "llama3.1:8b"
    model.rebind("qwen3:8b")
    assert (await model.resolve()).name == "qwen3:8b"


async def test_ollama_model_unknown_name(provider: OllamaProvider) -> None:
    model = OllamaModel("nope:1b", provider=provider)
    with pytest.raises(ModelNotFoundError):
        await model.resolve()


async def test_provider_from_settings_and_close(fake_ollama: FakeOllama) -> None:
    from local_agents.config.settings import OllamaSettings

    provider = OllamaProvider.from_settings(
        OllamaSettings(host="0.0.0.0", timeout=9), transport=fake_ollama.transport()
    )
    assert provider.host == "http://localhost:11434"
    assert provider.timeout == 9
    async with provider:
        assert await provider.health_check()
