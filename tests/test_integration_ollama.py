"""Integration tests against a real Ollama. Skipped unless it is reachable.

Run with: ``pytest -m integration``
"""

from __future__ import annotations

import pytest

from local_agents.agents.runtime import AgentRuntime
from local_agents.config.settings import Settings
from local_agents.models.ollama import OllamaModel, OllamaProvider

pytestmark = pytest.mark.integration


@pytest.fixture
async def live_provider() -> OllamaProvider:
    provider = OllamaProvider()
    if not await provider.health_check():
        pytest.skip(f"Ollama not reachable at {provider.host}")
    return provider


async def test_discovers_real_models(live_provider: OllamaProvider) -> None:
    models = await live_provider.list_models()
    assert models, "expected at least one installed model"
    assert all(m.name for m in models)


async def test_auto_model_generates(live_provider: OllamaProvider) -> None:
    model = OllamaModel.auto("fast", provider=live_provider, require_tools=False)
    from local_agents.models.base import Message

    result = await model.generate([Message.user("Reply with the single word: pong")])
    assert result.content.strip()


async def test_runtime_status_and_agent(live_provider: OllamaProvider, tmp_path) -> None:
    settings = Settings()
    settings.workspace.root = str(tmp_path)
    settings.security.confirm_shell = False
    (tmp_path / "note.txt").write_text("The secret word is pineapple.\n")
    async with AgentRuntime(settings, provider=live_provider) as runtime:
        status = await runtime.status()
        assert status.connected
        agent = await runtime.create_agent("file", model="auto")
        result = await agent.run("Read note.txt and tell me the secret word.")
        assert "pineapple" in result.output.lower()
