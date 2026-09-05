from __future__ import annotations

import httpx
import pytest

from local_agents.tools.web import (
    DuckDuckGoBackend,
    SearchResult,
    WebFetchTool,
    WebSearchTool,
    html_to_text,
    make_http_client,
    parse_duckduckgo_html,
    web_tools,
)

PAGE = """<html><head><title>Demo Page</title><style>body{}</style></head>
<body><nav>skip me</nav><h1>Heading</h1><p>First &amp; second.</p>
<script>alert(1)</script><div>Third</div></body></html>"""

DDG = """
<div class="result">
<a rel="nofollow" class="result__a" href="//duckduckgo.com/l/?uddg=https%3A%2F%2Fexample.com%2Fa&amp;rut=1">Example <b>A</b></a>
<a class="result__snippet" href="x">Snippet A</a>
</div>
<div class="result">
<a class="result__a" href="https://example.org/b">Example B</a>
</div>
"""


def test_html_to_text_strips_noise() -> None:
    title, text = html_to_text(PAGE)
    assert title == "Demo Page"
    assert "skip me" not in text and "alert" not in text
    assert "Heading" in text and "First & second." in text and "Third" in text


def test_parse_duckduckgo() -> None:
    results = parse_duckduckgo_html(DDG)
    assert [r.url for r in results] == ["https://example.com/a", "https://example.org/b"]
    assert results[0].title == "Example A" and results[0].snippet == "Snippet A"
    assert "1. Example A" in results[0].format(1)


def _client(handler) -> httpx.AsyncClient:
    return make_http_client(transport=httpx.MockTransport(handler))


async def test_fetch_tool_html_and_truncation() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, text=PAGE, headers={"content-type": "text/html; charset=utf-8"})

    tool = WebFetchTool(_client(handler))
    result = await tool.run(url="https://example.com/page", max_chars=20)
    assert result.success
    assert "Title: Demo Page" in result.output
    assert "truncated" in result.output


async def test_fetch_tool_json_passthrough() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"ok": True})

    result = await WebFetchTool(_client(handler)).run(url="https://api.example.com/x")
    assert '"ok"' in result.output and "application/json" in result.output


async def test_fetch_tool_errors() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/404":
            return httpx.Response(404, text="nope")
        raise httpx.ConnectTimeout("slow")

    tool = WebFetchTool(_client(handler))
    not_found = await tool.run(url="https://example.com/404")
    assert not not_found.success and "HTTP 404" in (not_found.error or "")
    timeout = await tool.run(url="https://example.com/slow")
    assert not timeout.success and "Timed out" in (timeout.error or "")
    bad = await tool.run(url="ftp://example.com/x")
    assert not bad.success and "http(s)" in (bad.error or "")


async def test_search_tool_with_ddg_backend() -> None:
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(200, text=DDG)

    tool = WebSearchTool(client=_client(handler))
    result = await tool.run(query="example", max_results=1)
    assert result.success
    assert "https://example.com/a" in result.output and "example.org" not in result.output
    assert b"q=example" in seen[0].content


async def test_search_tool_with_custom_backend() -> None:
    class Backend:
        async def search(self, query: str, max_results: int) -> list[SearchResult]:
            return [SearchResult("T", "https://t", "s")][:max_results]

    result = await WebSearchTool(backend=Backend()).run(query="q")
    assert result.data == [{"title": "T", "url": "https://t", "snippet": "s"}]
    empty = await WebSearchTool(
        backend=type("B", (), {"search": staticmethod(lambda q, n: _empty())})()
    ).run(query="q")
    assert "No results" in empty.output


async def _empty() -> list[SearchResult]:
    return []


async def test_search_backend_http_error() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(503)

    backend = DuckDuckGoBackend(_client(handler))
    with pytest.raises(Exception, match="HTTP 503"):
        await backend.search("x", 3)


def test_web_tools_factory() -> None:
    names = [t.name for t in web_tools()]
    assert names == ["web_search", "fetch_url"]
