"""Shared fixtures. Nothing here talks to a real Ollama."""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import httpx
import pytest

from local_agents.agents.registry import AgentRegistry
from local_agents.config.settings import Settings
from local_agents.models.ollama import OllamaProvider
from local_agents.security.workspace import Workspace

TAGS_PAYLOAD: dict[str, Any] = {
    "models": [
        {
            "name": "qwen3:8b",
            "size": 5_225_388_164,
            "modified_at": "2026-08-29T20:13:45Z",
            "details": {
                "family": "qwen3",
                "parameter_size": "8.2B",
                "quantization_level": "Q4_K_M",
                "context_length": 40960,
            },
            "capabilities": ["completion", "tools", "thinking"],
        },
        {
            "name": "qwen2.5-coder:14b",
            "size": 8_988_124_069,
            "modified_at": "2026-06-20T13:38:12Z",
            "details": {
                "family": "qwen2",
                "parameter_size": "14.8B",
                "quantization_level": "Q4_K_M",
                "context_length": 32768,
            },
            "capabilities": ["completion", "tools"],
        },
        {
            "name": "llama3.1:8b",
            "size": 4_920_753_328,
            "modified_at": "2026-06-20T13:37:17Z",
            "details": {
                "family": "llama",
                "parameter_size": "8.0B",
                "quantization_level": "Q4_K_M",
                "context_length": 131072,
            },
            "capabilities": ["completion", "tools"],
        },
        {
            "name": "deepseek-r1:70b",
            "size": 39_600_000_000,
            "modified_at": "2026-06-20T13:37:17Z",
            "details": {
                "family": "llama",
                "parameter_size": "70.6B",
                "quantization_level": "Q4_K_M",
                "context_length": 131072,
            },
            "capabilities": ["completion", "thinking"],
        },
        {
            "name": "nomic-embed-text:latest",
            "size": 274_000_000,
            "modified_at": "2026-06-20T13:37:17Z",
            "details": {"family": "nomic-bert", "parameter_size": "137M"},
            "capabilities": ["embedding"],
        },
    ]
}


class FakeOllama:
    """A scriptable fake of the Ollama HTTP API for ``httpx.MockTransport``.

    ``chat_responses`` is a queue of assistant messages returned by
    ``/api/chat`` in order. Each entry is a dict with ``content`` and optional
    ``tool_calls``/``thinking``. Requests are recorded for assertions.
    """

    def __init__(self, tags: dict[str, Any] | None = None) -> None:
        self.tags = TAGS_PAYLOAD if tags is None else tags
        self.chat_responses: list[dict[str, Any]] = []
        self.requests: list[httpx.Request] = []
        self.version = "0.32.9"
        self.fail_with: Exception | None = None
        self.status_override: tuple[int, dict[str, Any]] | None = None

    def queue(self, content: str = "", **extra: Any) -> None:
        self.chat_responses.append({"content": content, **extra})

    def handler(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        if self.fail_with is not None:
            raise self.fail_with
        if self.status_override is not None:
            status, body = self.status_override
            return httpx.Response(status, json=body)
        path = request.url.path
        if path == "/api/version":
            return httpx.Response(200, json={"version": self.version})
        if path == "/api/tags":
            return httpx.Response(200, json=self.tags)
        if path == "/api/show":
            name = json.loads(request.content)["model"]
            for model in self.tags["models"]:
                if model["name"] == name:
                    return httpx.Response(
                        200,
                        json={
                            "details": model.get("details", {}),
                            "capabilities": model.get("capabilities", []),
                            "model_info": {"general.context_length": 8192},
                            "parameters": "temperature 0.7",
                            "license": "MIT",
                        },
                    )
            return httpx.Response(404, json={"error": f"model '{name}' not found"})
        if path == "/api/chat":
            body = json.loads(request.content)
            model = body["model"]
            if not any(m["name"] == model for m in self.tags["models"]):
                return httpx.Response(404, json={"error": f"model '{model}' not found"})
            reply = self.chat_responses.pop(0) if self.chat_responses else {"content": ""}
            message: dict[str, Any] = {"role": "assistant", "content": reply.get("content", "")}
            if reply.get("tool_calls"):
                message["tool_calls"] = reply["tool_calls"]
            if reply.get("thinking"):
                message["thinking"] = reply["thinking"]
            final = {
                "model": model,
                "message": message,
                "done": True,
                "done_reason": "stop",
                "prompt_eval_count": 10,
                "eval_count": 5,
            }
            if body.get("stream"):
                chunks = []
                text = message["content"]
                for start in range(0, len(text), 3):
                    chunks.append(
                        {
                            "model": model,
                            "message": {"role": "assistant", "content": text[start : start + 3]},
                            "done": False,
                        }
                    )
                if message.get("tool_calls"):
                    chunks.append(
                        {
                            "model": model,
                            "message": {
                                "role": "assistant",
                                "content": "",
                                "tool_calls": message["tool_calls"],
                            },
                            "done": False,
                        }
                    )
                if message.get("thinking"):
                    chunks.insert(
                        0,
                        {
                            "model": model,
                            "message": {
                                "role": "assistant",
                                "content": "",
                                "thinking": message["thinking"],
                            },
                            "done": False,
                        },
                    )
                final["message"] = {"role": "assistant", "content": ""}
                chunks.append(final)
                payload = "\n".join(json.dumps(c) for c in chunks) + "\n"
                return httpx.Response(
                    200, content=payload, headers={"content-type": "application/x-ndjson"}
                )
            return httpx.Response(200, json=final)
        return httpx.Response(404, json={"error": "unknown endpoint"})

    def transport(self) -> httpx.MockTransport:
        return httpx.MockTransport(self.handler)


@pytest.fixture
def fake_ollama() -> FakeOllama:
    return FakeOllama()


@pytest.fixture
def provider(fake_ollama: FakeOllama) -> OllamaProvider:
    return OllamaProvider("http://ollama.test:11434", transport=fake_ollama.transport())


@pytest.fixture
def workspace(tmp_path: Path) -> Workspace:
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "app.py").write_text("def hello():\n    return 'hi'\n")
    (tmp_path / "README.md").write_text("# Demo\n\nHello world.\n")
    (tmp_path / "data.csv").write_text(
        "region,revenue,units\nnorth,100,5\nsouth,250,10\nnorth,50,2\neast,,1\n"
    )
    return Workspace(root=tmp_path, allow_write=True, allow_delete=False)


@pytest.fixture
def registry() -> AgentRegistry:
    reg = AgentRegistry()
    reg.discover(plugins=False, include_default_dir=False)
    return reg


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    cfg = Settings()
    cfg.workspace.root = str(tmp_path)
    cfg.security.confirm_shell = False
    return cfg


@pytest.fixture
def make_settings(tmp_path: Path) -> Callable[..., Settings]:
    def _make(**overrides: Any) -> Settings:
        cfg = Settings()
        cfg.workspace.root = str(tmp_path)
        cfg.security.confirm_shell = False
        for key, value in overrides.items():
            setattr(cfg, key, value)
        return cfg

    return _make
