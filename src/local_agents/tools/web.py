"""Web tools: fetch pages as readable text and run searches.

Search is pluggable through :class:`SearchBackend`. The default backend uses
DuckDuckGo's HTML endpoint, which needs no API key. Tests inject an
``httpx.AsyncClient`` built on ``httpx.MockTransport``.
"""

from __future__ import annotations

import contextlib
import html
import json
import re
from dataclasses import dataclass
from html.parser import HTMLParser
from typing import Protocol
from urllib.parse import parse_qs, urlparse

import httpx

from local_agents.errors import ToolError
from local_agents.security.permissions import Permission
from local_agents.tools.base import BaseTool, ToolParameter, ToolResult

DEFAULT_USER_AGENT = "local-agents/0.1 (+https://github.com/dsneed123/local-agents)"
DEFAULT_TIMEOUT = 20.0


@dataclass
class SearchResult:
    title: str
    url: str
    snippet: str = ""

    def format(self, index: int) -> str:
        text = f"{index}. {self.title}\n   {self.url}"
        if self.snippet:
            text += f"\n   {self.snippet}"
        return text


class SearchBackend(Protocol):
    """Anything that turns a query into :class:`SearchResult` items."""

    async def search(self, query: str, max_results: int) -> list[SearchResult]: ...


def make_http_client(
    *, timeout: float = DEFAULT_TIMEOUT, transport: httpx.AsyncBaseTransport | None = None
) -> httpx.AsyncClient:
    return httpx.AsyncClient(
        timeout=timeout,
        follow_redirects=True,
        headers={"User-Agent": DEFAULT_USER_AGENT, "Accept-Language": "en"},
        transport=transport,
    )


def validate_url(url: str) -> str:
    parsed = urlparse(url.strip())
    if parsed.scheme not in ("http", "https") or not parsed.netloc:
        raise ToolError(f"Only http(s) URLs are supported (got {url!r})")
    return url.strip()


# --------------------------------------------------------------------------- #
# HTML -> text
# --------------------------------------------------------------------------- #

_SKIP_TAGS = {"script", "style", "noscript", "svg", "head", "nav", "footer", "iframe"}
_BLOCK_TAGS = {
    "p",
    "div",
    "br",
    "li",
    "h1",
    "h2",
    "h3",
    "h4",
    "h5",
    "h6",
    "tr",
    "section",
    "article",
    "header",
    "pre",
    "blockquote",
    "table",
    "ul",
    "ol",
}


class _TextExtractor(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self._chunks: list[str] = []
        self._skip_depth = 0
        self.title = ""
        self._in_title = False

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag in _SKIP_TAGS:
            self._skip_depth += 1
        elif tag == "title":
            self._in_title = True
        elif tag in _BLOCK_TAGS:
            self._chunks.append("\n")
        if tag in ("h1", "h2", "h3"):
            self._chunks.append("\n")

    def handle_endtag(self, tag: str) -> None:
        if tag in _SKIP_TAGS and self._skip_depth:
            self._skip_depth -= 1
        elif tag == "title":
            self._in_title = False
        elif tag in _BLOCK_TAGS:
            self._chunks.append("\n")

    def handle_data(self, data: str) -> None:
        if self._in_title:
            self.title += data
        if self._skip_depth:
            return
        self._chunks.append(data)

    def text(self) -> str:
        raw = "".join(self._chunks)
        raw = re.sub(r"[ \t\r\f\v]+", " ", raw)
        raw = re.sub(r" *\n *", "\n", raw)
        raw = re.sub(r"\n{3,}", "\n\n", raw)
        return raw.strip()


def html_to_text(markup: str) -> tuple[str, str]:
    """Return ``(title, text)`` for an HTML document."""
    parser = _TextExtractor()
    parser.feed(markup)
    parser.close()
    return parser.title.strip(), parser.text()


# --------------------------------------------------------------------------- #
# Backends
# --------------------------------------------------------------------------- #


class DuckDuckGoBackend:
    """Scrapes ``https://html.duckduckgo.com/html/`` (no API key needed)."""

    endpoint = "https://html.duckduckgo.com/html/"

    def __init__(self, client: httpx.AsyncClient) -> None:
        self._client = client

    async def search(self, query: str, max_results: int) -> list[SearchResult]:
        try:
            response = await self._client.post(self.endpoint, data={"q": query, "kl": "us-en"})
        except httpx.HTTPError as exc:
            raise ToolError(f"Search request failed: {exc}") from exc
        if response.status_code >= 400:
            raise ToolError(f"Search returned HTTP {response.status_code}")
        return parse_duckduckgo_html(response.text)[:max_results]


_ANCHOR_RE = re.compile(
    r'<a[^>]+class="result__a"[^>]+href="(?P<href>[^"]+)"[^>]*>(?P<title>.*?)</a>', re.DOTALL
)
_SNIPPET_RE = re.compile(r'class="result__snippet"[^>]*>(?P<snippet>.*?)</a>', re.DOTALL)
_TAG_RE = re.compile(r"<[^>]+>")


def parse_duckduckgo_html(markup: str) -> list[SearchResult]:
    results: list[SearchResult] = []
    anchors = list(_ANCHOR_RE.finditer(markup))
    for index, match in enumerate(anchors):
        href = html.unescape(match.group("href"))
        # DDG wraps links as /l/?uddg=<encoded url>
        if "uddg=" in href:
            query = parse_qs(urlparse(href).query)
            href = query.get("uddg", [href])[0]
        if href.startswith("//"):
            href = "https:" + href
        title = html.unescape(_TAG_RE.sub("", match.group("title"))).strip()
        block_end = anchors[index + 1].start() if index + 1 < len(anchors) else len(markup)
        snippet_match = _SNIPPET_RE.search(markup, match.end(), block_end)
        snippet = ""
        if snippet_match:
            snippet = html.unescape(_TAG_RE.sub("", snippet_match.group("snippet"))).strip()
        if title and href.startswith("http"):
            results.append(SearchResult(title=title, url=href, snippet=snippet))
    return results


# --------------------------------------------------------------------------- #
# Tools
# --------------------------------------------------------------------------- #


class WebFetchTool(BaseTool):
    name = "fetch_url"
    description = (
        "Fetch a web page or API endpoint and return its readable text content. "
        "HTML is converted to plain text."
    )
    parameters = (
        ToolParameter("url", "string", "Absolute http(s) URL"),
        ToolParameter(
            "max_chars",
            "integer",
            "Truncate content to this many characters",
            required=False,
            default=8000,
        ),
    )
    permission = Permission.NETWORK

    def __init__(self, client: httpx.AsyncClient | None = None) -> None:
        self._client = client or make_http_client()

    async def execute(self, url: str, max_chars: int = 8000) -> ToolResult:
        url = validate_url(url)
        try:
            response = await self._client.get(url)
        except httpx.TimeoutException as exc:
            raise ToolError(f"Timed out fetching {url}") from exc
        except httpx.HTTPError as exc:
            raise ToolError(f"Could not fetch {url}: {exc}") from exc
        if response.status_code >= 400:
            raise ToolError(f"{url} returned HTTP {response.status_code}")
        content_type = response.headers.get("content-type", "")
        body = response.text
        title = ""
        if "html" in content_type or body.lstrip()[:200].lower().startswith(("<!doctype", "<html")):
            title, body = html_to_text(body)
        truncated = len(body) > max_chars
        body = body[:max_chars]
        header = f"URL: {response.url!s}\n"
        if title:
            header += f"Title: {title}\n"
        header += f"Content-Type: {content_type.split(';')[0] or 'unknown'}\n\n"
        if truncated:
            body += f"\n... [truncated to {max_chars} characters]"
        return ToolResult.ok(header + body, data={"url": str(response.url), "title": title})


class WebSearchTool(BaseTool):
    name = "web_search"
    description = "Search the web and return titles, URLs and snippets for the top results."
    parameters = (
        ToolParameter("query", "string", "Search query"),
        ToolParameter(
            "max_results", "integer", "How many results to return", required=False, default=8
        ),
    )
    permission = Permission.NETWORK

    def __init__(
        self, backend: SearchBackend | None = None, client: httpx.AsyncClient | None = None
    ) -> None:
        self._backend: SearchBackend = backend or DuckDuckGoBackend(client or make_http_client())

    async def execute(self, query: str, max_results: int = 8) -> ToolResult:
        results = await self._backend.search(query, max(1, min(max_results, 25)))
        if not results:
            return ToolResult.ok(f"No results for {query!r}")
        text = "\n".join(r.format(i) for i, r in enumerate(results, 1))
        return ToolResult.ok(text, data=[r.__dict__ for r in results])


HTTP_METHODS = ("GET", "POST", "PUT", "PATCH", "DELETE", "HEAD", "OPTIONS")


class HttpRequestTool(BaseTool):
    """General HTTP client for testing APIs."""

    name = "http_request"
    description = (
        "Send an HTTP request to an API endpoint and return the status, headers and body. "
        "Supports any method, custom headers, and a JSON or raw body."
    )
    parameters = (
        ToolParameter("url", "string", "Absolute http(s) URL"),
        ToolParameter(
            "method",
            "string",
            "HTTP method",
            required=False,
            default="GET",
            enum=list(HTTP_METHODS),
        ),
        ToolParameter("headers", "object", "Request headers as an object", required=False),
        ToolParameter("json_body", "object", "JSON object to send as the body", required=False),
        ToolParameter(
            "body", "string", "Raw request body (used when json_body is absent)", required=False
        ),
        ToolParameter(
            "max_chars", "integer", "Truncate the response body", required=False, default=6000
        ),
    )
    permission = Permission.NETWORK

    def __init__(self, client: httpx.AsyncClient | None = None) -> None:
        self._client = client or make_http_client()

    async def execute(
        self,
        url: str,
        method: str = "GET",
        headers: dict[str, str] | None = None,
        json_body: dict[str, object] | None = None,
        body: str | None = None,
        max_chars: int = 6000,
    ) -> ToolResult:
        url = validate_url(url)
        kwargs: dict[str, object] = {"headers": headers or {}}
        if json_body is not None:
            kwargs["json"] = json_body
        elif body is not None:
            kwargs["content"] = body.encode("utf-8")
        try:
            response = await self._client.request(method.upper(), url, **kwargs)  # type: ignore[arg-type]
        except httpx.TimeoutException as exc:
            raise ToolError(f"Timed out requesting {url}") from exc
        except httpx.HTTPError as exc:
            raise ToolError(f"Request to {url} failed: {exc}") from exc
        text = response.text
        content_type = response.headers.get("content-type", "")
        if "json" in content_type:
            with contextlib.suppress(ValueError):
                text = json.dumps(response.json(), indent=2)
        truncated = len(text) > max_chars
        shown_headers = "\n".join(
            f"  {k}: {v}"
            for k, v in response.headers.items()
            if k.lower()
            in ("content-type", "content-length", "location", "server", "date", "x-request-id")
        )
        output = (
            f"{method.upper()} {url}\nStatus: {response.status_code} {response.reason_phrase}\n"
            f"Headers:\n{shown_headers}\n\n{text[:max_chars]}"
        )
        if truncated:
            output += f"\n... [truncated to {max_chars} characters]"
        return ToolResult(
            output=output,
            success=response.status_code < 400,
            error=None if response.status_code < 400 else f"HTTP {response.status_code}",
            data={
                "status": response.status_code,
                "headers": dict(response.headers),
                "body": response.text,
            },
        )


def http_tools(client: httpx.AsyncClient | None = None) -> list[BaseTool]:
    return [HttpRequestTool(client or make_http_client())]


def web_tools(
    client: httpx.AsyncClient | None = None, *, backend: SearchBackend | None = None
) -> list[BaseTool]:
    shared = client or make_http_client()
    return [WebSearchTool(backend=backend, client=shared), WebFetchTool(shared)]
