"""Ollama provider.

Two classes live here:

* :class:`OllamaProvider` — one HTTP client per Ollama endpoint. It discovers
  installed models (cached for the session, refreshable), performs health
  checks and executes chat requests.
* :class:`OllamaModel` — a :class:`BaseModel` bound to one model name on a
  provider. ``OllamaModel.auto()`` returns a model whose concrete name is
  resolved lazily from the installed models on first use, so no configuration
  is needed for the common case.

All network I/O is async (``httpx.AsyncClient``). Tests inject an
``httpx.MockTransport`` so no Ollama instance is required.
"""

from __future__ import annotations

import asyncio
import json
import os
from collections.abc import AsyncIterator, Sequence
from typing import Any

import httpx

from local_agents.config.settings import (
    DEFAULT_OLLAMA_HOST,
    ENV_OLLAMA_HOST,
    OllamaSettings,
    normalize_host,
)
from local_agents.errors import (
    ModelNotFoundError,
    ModelResponseError,
    ModelTimeoutError,
    NoModelsAvailableError,
    ProviderConnectionError,
)
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
from local_agents.models.selection import ModelRequirements, ModelRole, ModelSelector


def detect_ollama_host(explicit: str | None = None) -> str:
    """Determine the Ollama endpoint: explicit > ``OLLAMA_HOST`` > default."""
    if explicit:
        return normalize_host(explicit)
    env = os.environ.get(ENV_OLLAMA_HOST)
    if env:
        return normalize_host(env)
    return DEFAULT_OLLAMA_HOST


class OllamaProvider:
    """Async client for one Ollama endpoint with session-level model caching."""

    def __init__(
        self,
        host: str | None = None,
        *,
        timeout: float = 120.0,
        connect_timeout: float = 5.0,
        keep_alive: str | None = None,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        self._host = detect_ollama_host(host)
        self._timeout = timeout
        self._connect_timeout = connect_timeout
        self._keep_alive = keep_alive
        self._transport = transport
        self._client: httpx.AsyncClient | None = None
        self._models: list[ModelInfo] | None = None
        self._models_lock = asyncio.Lock()
        self._version: str | None = None

    @classmethod
    def from_settings(
        cls, settings: OllamaSettings, *, transport: httpx.AsyncBaseTransport | None = None
    ) -> OllamaProvider:
        return cls(
            settings.host,
            timeout=settings.timeout,
            connect_timeout=settings.connect_timeout,
            keep_alive=settings.keep_alive,
            transport=transport,
        )

    # ------------------------------------------------------------------ #
    # Basics
    # ------------------------------------------------------------------ #

    @property
    def host(self) -> str:
        return self._host

    @property
    def timeout(self) -> float:
        return self._timeout

    def _get_client(self) -> httpx.AsyncClient:
        if self._client is None:
            timeout = httpx.Timeout(self._timeout, connect=self._connect_timeout)
            self._client = httpx.AsyncClient(
                base_url=self._host, timeout=timeout, transport=self._transport
            )
        return self._client

    async def close(self) -> None:
        if self._client is not None:
            await self._client.aclose()
            self._client = None

    async def __aenter__(self) -> OllamaProvider:
        return self

    async def __aexit__(self, *exc: object) -> None:
        await self.close()

    def invalidate_cache(self) -> None:
        """Forget cached model discovery so the next call re-queries Ollama."""
        self._models = None

    # ------------------------------------------------------------------ #
    # Discovery
    # ------------------------------------------------------------------ #

    async def health_check(self) -> bool:
        """Return ``True`` if the endpoint answers; never raises."""
        try:
            response = await self._get_client().get("/api/version")
        except (httpx.HTTPError, OSError):
            return False
        if response.status_code >= 400:
            return False
        try:
            self._version = str(response.json().get("version", ""))
        except (ValueError, AttributeError):
            self._version = None
        return True

    async def version(self) -> str | None:
        if self._version is None:
            await self.health_check()
        return self._version

    async def list_models(self, *, refresh: bool = False) -> list[ModelInfo]:
        """Return installed models. Cached per session unless ``refresh`` is set."""
        if self._models is not None and not refresh:
            return list(self._models)
        async with self._models_lock:
            if self._models is not None and not refresh:
                return list(self._models)
            data = await self._request_json("GET", "/api/tags")
            raw_models = data.get("models") or []
            models = [_parse_tag_entry(entry) for entry in raw_models]
            models = await self._fill_missing_capabilities(models)
            self._models = models
            return list(models)

    async def _fill_missing_capabilities(self, models: list[ModelInfo]) -> list[ModelInfo]:
        """Older Ollama versions omit ``capabilities`` from ``/api/tags``.

        For those, ask ``/api/show`` once per model (concurrently) so selection
        can still reason about tool support.
        """
        missing = [m for m in models if not m.capabilities]
        if not missing:
            return models
        results = await asyncio.gather(
            *(self.show(m.name) for m in missing), return_exceptions=True
        )
        details = {m.name: r for m, r in zip(missing, results, strict=True) if isinstance(r, dict)}
        filled: list[ModelInfo] = []
        for model in models:
            extra = details.get(model.name)
            filled.append(_merge_show_details(model, extra) if extra else model)
        return filled

    async def show(self, name: str) -> dict[str, Any]:
        """Return the raw ``/api/show`` payload for ``name``."""
        return await self._request_json("POST", "/api/show", json={"model": name})

    async def has_model(self, name: str) -> bool:
        return self.find_model(await self.list_models(), name) is not None

    @staticmethod
    def find_model(models: Sequence[ModelInfo], name: str) -> ModelInfo | None:
        return ModelSelector(models).find(name)

    async def selector(self, *, refresh: bool = False) -> ModelSelector:
        return ModelSelector(await self.list_models(refresh=refresh))

    async def require_models(self) -> list[ModelInfo]:
        """Like :meth:`list_models` but raises when the list is empty."""
        models = await self.list_models()
        if not models:
            raise NoModelsAvailableError(self._host)
        return models

    # ------------------------------------------------------------------ #
    # Model factories
    # ------------------------------------------------------------------ #

    def model(self, name: str, *, options: GenerationOptions | None = None) -> OllamaModel:
        """Bind a concrete model name (validated lazily on first call)."""
        return OllamaModel(name, provider=self, options=options)

    def auto(
        self,
        role: ModelRole = "general",
        *,
        require_tools: bool = True,
        prefer: Sequence[str] = (),
        options: GenerationOptions | None = None,
    ) -> OllamaModel:
        """Bind a model chosen automatically for ``role`` on first use."""
        return OllamaModel(
            None,
            provider=self,
            requirements=ModelRequirements(role=role, require_tools=require_tools, prefer=prefer),
            options=options,
        )

    async def resolve(self, name: str | None, requirements: ModelRequirements | None) -> ModelInfo:
        """Resolve ``name`` (or ``"auto"``/``None``) to an installed model."""
        models = await self.require_models()
        selector = ModelSelector(models)
        if name and name != "auto":
            found = selector.find(name)
            if found is None:
                raise ModelNotFoundError(name, [m.name for m in models])
            return found
        chosen = selector.select(requirements)
        if chosen is None:  # pragma: no cover - require_models guards this
            raise NoModelsAvailableError(self._host)
        return chosen

    # ------------------------------------------------------------------ #
    # Chat
    # ------------------------------------------------------------------ #

    async def chat(
        self,
        model: str,
        messages: Sequence[Message],
        *,
        tools: Sequence[ToolSpec] | None = None,
        options: GenerationOptions | None = None,
    ) -> GenerationResult:
        payload = self._chat_payload(model, messages, tools, options, stream=False)
        timeout = options.timeout if options and options.timeout else self._timeout
        data = await self._request_json(
            "POST", "/api/chat", json=payload, timeout=timeout, model=model
        )
        return _parse_chat_response(data, model)

    async def stream_chat(
        self,
        model: str,
        messages: Sequence[Message],
        *,
        tools: Sequence[ToolSpec] | None = None,
        options: GenerationOptions | None = None,
    ) -> AsyncIterator[StreamChunk]:
        payload = self._chat_payload(model, messages, tools, options, stream=True)
        timeout = options.timeout if options and options.timeout else self._timeout
        client = self._get_client()
        try:
            async with client.stream(
                "POST",
                "/api/chat",
                json=payload,
                timeout=httpx.Timeout(timeout, connect=self._connect_timeout),
            ) as response:
                if response.status_code >= 400:
                    body = await response.aread()
                    raise self._http_error(response.status_code, body, model)
                async for line in response.aiter_lines():
                    if not line.strip():
                        continue
                    try:
                        data = json.loads(line)
                    except json.JSONDecodeError as exc:
                        raise ModelResponseError(
                            f"Malformed stream chunk from Ollama: {line[:200]}"
                        ) from exc
                    if "error" in data:
                        raise ModelResponseError(f"Ollama error: {data['error']}")
                    yield _parse_stream_chunk(data)
        except httpx.ConnectError as exc:
            raise ProviderConnectionError(self._host, cause=exc) from exc
        except httpx.TimeoutException as exc:
            raise ModelTimeoutError(model, timeout) from exc
        except httpx.HTTPError as exc:
            raise ModelResponseError(f"HTTP error talking to Ollama: {exc}") from exc

    def _chat_payload(
        self,
        model: str,
        messages: Sequence[Message],
        tools: Sequence[ToolSpec] | None,
        options: GenerationOptions | None,
        *,
        stream: bool,
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "model": model,
            "messages": [m.to_dict() for m in messages],
            "stream": stream,
        }
        if tools:
            payload["tools"] = [t.to_dict() for t in tools]
        if self._keep_alive:
            payload["keep_alive"] = self._keep_alive
        if options:
            ollama_opts: dict[str, Any] = {}
            if options.temperature is not None:
                ollama_opts["temperature"] = options.temperature
            if options.context_length is not None:
                ollama_opts["num_ctx"] = options.context_length
            if options.max_tokens is not None:
                ollama_opts["num_predict"] = options.max_tokens
            if options.stop:
                ollama_opts["stop"] = list(options.stop)
            if options.seed is not None:
                ollama_opts["seed"] = options.seed
            if ollama_opts:
                payload["options"] = ollama_opts
            if options.think is not None:
                payload["think"] = options.think
        return payload

    # ------------------------------------------------------------------ #
    # HTTP plumbing
    # ------------------------------------------------------------------ #

    async def _request_json(
        self,
        method: str,
        path: str,
        *,
        json: dict[str, Any] | None = None,
        timeout: float | None = None,
        model: str | None = None,
    ) -> dict[str, Any]:
        client = self._get_client()
        kwargs: dict[str, Any] = {}
        if json is not None:
            kwargs["json"] = json
        if timeout is not None:
            kwargs["timeout"] = httpx.Timeout(timeout, connect=self._connect_timeout)
        try:
            response = await client.request(method, path, **kwargs)
        except httpx.ConnectError as exc:
            raise ProviderConnectionError(self._host, cause=exc) from exc
        except httpx.TimeoutException as exc:
            if model:
                raise ModelTimeoutError(model, timeout or self._timeout) from exc
            raise ProviderConnectionError(self._host, cause=exc) from exc
        except httpx.HTTPError as exc:
            raise ModelResponseError(f"HTTP error talking to Ollama: {exc}") from exc
        if response.status_code >= 400:
            raise self._http_error(response.status_code, response.content, model)
        try:
            data = response.json()
        except ValueError as exc:
            raise ModelResponseError("Ollama returned a non-JSON response") from exc
        if not isinstance(data, dict):
            raise ModelResponseError("Ollama returned an unexpected response shape")
        if "error" in data:
            raise ModelResponseError(f"Ollama error: {data['error']}")
        return data

    def _http_error(self, status: int, body: bytes, model: str | None) -> Exception:
        message = ""
        try:
            message = str(json.loads(body).get("error", ""))
        except (ValueError, AttributeError):
            message = body.decode("utf-8", "replace")[:300]
        lowered = message.lower()
        if status == 404 and model and ("not found" in lowered or "no such" in lowered):
            names = [m.name for m in self._models] if self._models else []
            return ModelNotFoundError(model, names)
        return ModelResponseError(
            f"Ollama responded with HTTP {status}: {message or 'unknown error'}",
            hint="Check `ollama logs` or run the request manually to debug.",
        )


# --------------------------------------------------------------------------- #
# Bound model
# --------------------------------------------------------------------------- #


class OllamaModel(BaseModel):
    """A :class:`BaseModel` bound to one Ollama model.

    Construct with an explicit name (``OllamaModel("qwen3:8b")``) or use
    :meth:`auto` to defer the choice until the first call. Either way, the
    provider is shared and discovery is cached.
    """

    provider = "ollama"

    def __init__(
        self,
        model: str | None = None,
        *,
        provider: OllamaProvider | None = None,
        host: str | None = None,
        requirements: ModelRequirements | None = None,
        options: GenerationOptions | None = None,
    ) -> None:
        self._provider = provider or OllamaProvider(host)
        self._requested = None if model in (None, "auto") else model
        self._requirements = requirements or ModelRequirements()
        self._resolved: ModelInfo | None = None
        self.options = options or GenerationOptions()
        self._resolve_lock = asyncio.Lock()

    @classmethod
    def auto(
        cls,
        role: ModelRole = "general",
        *,
        host: str | None = None,
        provider: OllamaProvider | None = None,
        require_tools: bool = True,
        prefer: Sequence[str] = (),
        options: GenerationOptions | None = None,
    ) -> OllamaModel:
        """Return a model whose concrete name is chosen from installed models.

        Resolution happens on the first ``generate``/``stream`` call (or via
        :meth:`resolve`), so this is safe to call from synchronous code.
        """
        return cls(
            None,
            provider=provider,
            host=host,
            requirements=ModelRequirements(role=role, require_tools=require_tools, prefer=prefer),
            options=options,
        )

    # ------------------------------------------------------------------ #

    @property
    def ollama(self) -> OllamaProvider:
        return self._provider

    @property
    def is_resolved(self) -> bool:
        return self._resolved is not None

    @property
    def requested_name(self) -> str | None:
        return self._requested

    @property
    def role(self) -> ModelRole:
        return self._requirements.role

    @property
    def name(self) -> str:
        if self._resolved is not None:
            return self._resolved.name
        if self._requested:
            return self._requested
        return f"auto:{self._requirements.role}"

    async def resolve(self, *, refresh: bool = False) -> ModelInfo:
        """Resolve and cache the concrete model. Raises if it cannot be found."""
        if self._resolved is not None and not refresh:
            return self._resolved
        async with self._resolve_lock:
            if self._resolved is not None and not refresh:
                return self._resolved
            if refresh:
                self._provider.invalidate_cache()
            self._resolved = await self._provider.resolve(self._requested, self._requirements)
            return self._resolved

    def rebind(self, name: str) -> None:
        """Switch to a different model name (resolved lazily like the original)."""
        self._requested = name
        self._resolved = None

    async def generate(
        self,
        messages: Sequence[Message],
        *,
        tools: Sequence[ToolSpec] | None = None,
        options: GenerationOptions | None = None,
    ) -> GenerationResult:
        info = await self.resolve()
        return await self._provider.chat(
            info.name, messages, tools=tools, options=self.options.merged(options)
        )

    async def stream(
        self,
        messages: Sequence[Message],
        *,
        tools: Sequence[ToolSpec] | None = None,
        options: GenerationOptions | None = None,
    ) -> AsyncIterator[StreamChunk]:
        info = await self.resolve()
        async for chunk in self._provider.stream_chat(
            info.name, messages, tools=tools, options=self.options.merged(options)
        ):
            yield chunk

    async def health_check(self) -> bool:
        return await self._provider.health_check()

    async def list_models(self, *, refresh: bool = False) -> list[ModelInfo]:
        return await self._provider.list_models(refresh=refresh)

    async def info(self) -> ModelInfo | None:
        return await self.resolve()

    async def close(self) -> None:
        await self._provider.close()


# --------------------------------------------------------------------------- #
# Parsing helpers
# --------------------------------------------------------------------------- #


def _parse_tag_entry(entry: dict[str, Any]) -> ModelInfo:
    details = entry.get("details") or {}
    return ModelInfo(
        name=str(entry.get("name") or entry.get("model") or ""),
        family=str(details.get("family") or ""),
        parameter_size=str(details.get("parameter_size") or ""),
        quantization=str(details.get("quantization_level") or ""),
        context_length=_as_int(details.get("context_length")),
        capabilities=tuple(str(c) for c in entry.get("capabilities") or ()),
        size_bytes=int(entry.get("size") or 0),
        modified_at=str(entry.get("modified_at") or ""),
    )


def _merge_show_details(model: ModelInfo, show: dict[str, Any]) -> ModelInfo:
    capabilities = tuple(str(c) for c in show.get("capabilities") or ())
    context_length = model.context_length
    model_info = show.get("model_info") or {}
    if context_length is None:
        for key, value in model_info.items():
            if key.endswith(".context_length"):
                context_length = _as_int(value)
                break
    details = show.get("details") or {}
    return ModelInfo(
        name=model.name,
        family=model.family or str(details.get("family") or ""),
        parameter_size=model.parameter_size or str(details.get("parameter_size") or ""),
        quantization=model.quantization or str(details.get("quantization_level") or ""),
        context_length=context_length,
        capabilities=capabilities or model.capabilities,
        size_bytes=model.size_bytes,
        modified_at=model.modified_at,
    )


def _as_int(value: Any) -> int | None:
    try:
        return int(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def _parse_tool_calls(raw: Any) -> list[ToolCall]:
    calls: list[ToolCall] = []
    for item in raw or []:
        fn = item.get("function", item) if isinstance(item, dict) else {}
        args = fn.get("arguments") or {}
        if isinstance(args, str):
            try:
                args = json.loads(args)
            except json.JSONDecodeError:
                args = {"input": args}
        calls.append(ToolCall(name=str(fn.get("name", "")), arguments=args, id=item.get("id")))
    return calls


def _parse_chat_response(data: dict[str, Any], model: str) -> GenerationResult:
    message = data.get("message") or {}
    return GenerationResult(
        content=str(message.get("content") or ""),
        model=str(data.get("model") or model),
        tool_calls=_parse_tool_calls(message.get("tool_calls")),
        thinking=message.get("thinking") or None,
        done_reason=data.get("done_reason"),
        usage=Usage(
            prompt_tokens=int(data.get("prompt_eval_count") or 0),
            completion_tokens=int(data.get("eval_count") or 0),
        ),
        raw=data,
    )


def _parse_stream_chunk(data: dict[str, Any]) -> StreamChunk:
    message = data.get("message") or {}
    done = bool(data.get("done"))
    usage = None
    if done:
        usage = Usage(
            prompt_tokens=int(data.get("prompt_eval_count") or 0),
            completion_tokens=int(data.get("eval_count") or 0),
        )
    return StreamChunk(
        content=str(message.get("content") or ""),
        thinking=str(message.get("thinking") or ""),
        tool_calls=_parse_tool_calls(message.get("tool_calls")),
        done=done,
        done_reason=data.get("done_reason"),
        usage=usage,
    )
