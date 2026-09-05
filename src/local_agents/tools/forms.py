"""Extract form fields (questions) from HTML pages or files."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from html.parser import HTMLParser

import httpx

from local_agents.errors import ToolError
from local_agents.security.permissions import Permission
from local_agents.security.workspace import Workspace
from local_agents.tools.base import BaseTool, ToolParameter, ToolResult
from local_agents.tools.web import make_http_client, validate_url

_IGNORED_INPUT_TYPES = {"hidden", "submit", "button", "reset", "image"}


@dataclass
class FormField:
    name: str
    kind: str
    label: str = ""
    required: bool = False
    options: list[str] = field(default_factory=list)
    placeholder: str = ""
    value: str = ""

    def describe(self) -> str:
        text = f"{self.label or self.name}"
        meta = [self.kind]
        if self.required:
            meta.append("required")
        if self.options:
            meta.append(
                "options: "
                + " | ".join(self.options[:12])
                + (" ..." if len(self.options) > 12 else "")
            )
        if self.placeholder:
            meta.append(f"placeholder: {self.placeholder}")
        if self.value and self.kind not in ("checkbox", "radio"):
            meta.append(f"current: {self.value}")
        return f"{text}  [{'; '.join(meta)}]  (name={self.name})"


def parse_form_fields(markup: str) -> list[list[FormField]]:
    """Return the fields of every form on a page (fields outside forms come first)."""
    parser = _SimpleFormParser()
    parser.feed(markup)
    parser.close()
    return [f for f in parser.forms if f]


class _SimpleFormParser(HTMLParser):
    """Straightforward form parser: labels via ``for=``, wrapping labels, or preceding text."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.forms: list[list[FormField]] = [[]]
        self._by_id: dict[str, FormField] = {}
        self._groups: dict[str, FormField] = {}
        self._deferred_labels: dict[str, str] = {}
        self._label_for: str | None = None
        self._label_buf: list[str] | None = None
        self._label_fields: list[FormField] = []
        self._label_option: tuple[FormField, str] | None = None
        self._select: FormField | None = None
        self._option_buf: list[str] | None = None
        self._textarea: FormField | None = None
        self._recent: list[str] = []
        self._pending_option: tuple[FormField, str, str | None] | None = None
        self._skip = 0

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        a = {k: (v or "") for k, v in attrs}
        if tag in ("script", "style"):
            self._skip += 1
        elif tag == "form":
            if self.forms[-1]:
                self.forms.append([])
        elif tag == "label":
            self._label_for = a.get("for") or None
            self._label_buf = []
            self._label_fields = []
            self._label_option = None
        elif tag == "input":
            self._input(a)
        elif tag == "select":
            self._select = FormField(
                a.get("name") or a.get("id") or "", "select", required="required" in a
            )
            self._register(self._select, a.get("id"))
        elif tag == "option" and self._select is not None:
            self._option_buf = []
        elif tag == "textarea":
            self._textarea = FormField(
                a.get("name") or a.get("id") or "",
                "textarea",
                required="required" in a,
                placeholder=a.get("placeholder", ""),
            )
            self._register(self._textarea, a.get("id"))

    def handle_startendtag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        self.handle_starttag(tag, attrs)

    def _input(self, a: dict[str, str]) -> None:
        kind = (a.get("type") or "text").lower()
        if kind in _IGNORED_INPUT_TYPES:
            return
        name = a.get("name") or a.get("id") or ""
        if not name:
            return
        if kind in ("radio", "checkbox"):
            group = self._groups.get(name)
            if group is None:
                group = FormField(name, kind, required="required" in a)
                self._groups[name] = group
                self.forms[-1].append(group)
                group.label = self._context()  # legend / heading text before the group
            if a.get("id"):
                self._by_id[a["id"]] = group
            value = a.get("value", "on")
            if self._label_buf is not None:
                self._label_option = (group, value)
            elif a.get("id") and a["id"] in self._deferred_labels:
                group.options.append(self._deferred_labels.pop(a["id"]))
            else:
                self._pending_option = (group, value, a.get("id"))
            return
        fld = FormField(
            name,
            kind,
            required="required" in a,
            placeholder=a.get("placeholder", ""),
            value=a.get("value", ""),
        )
        self._register(fld, a.get("id"))

    def _register(self, fld: FormField, element_id: str | None) -> None:
        self.forms[-1].append(fld)
        if element_id:
            self._by_id[element_id] = fld
            if element_id in self._deferred_labels:
                fld.label = self._deferred_labels.pop(element_id)
        if self._label_buf is not None and self._label_for is None:
            self._label_fields.append(fld)
        elif not fld.label:
            fld.label = self._context()

    def _context(self) -> str:
        """The most recent text before the field (a heading, legend or paragraph)."""
        text = self._recent[-1] if self._recent else ""
        self._recent = []
        return text[-120:]

    def handle_endtag(self, tag: str) -> None:
        if tag in ("script", "style"):
            self._skip = max(0, self._skip - 1)
        elif tag == "label" and self._label_buf is not None:
            text = re.sub(r"\s+", " ", " ".join(self._label_buf)).strip()
            if self._label_option is not None:
                group, value = self._label_option
                group.options.append(text or value)
            elif self._label_for:
                target = self._by_id.get(self._label_for)
                if target is None:
                    self._deferred_labels[self._label_for] = text
                elif target.kind in ("radio", "checkbox"):
                    pending = self._pending_option
                    if pending and pending[2] == self._label_for:
                        target.options.append(text or pending[1])
                        self._pending_option = None
                    else:
                        target.options.append(text)
                elif not target.label:
                    target.label = text
            for fld in self._label_fields:
                if not fld.label:
                    fld.label = text
            self._label_buf = None
            self._label_for = None
            self._label_fields = []
            self._label_option = None
        elif tag == "option" and self._option_buf is not None and self._select is not None:
            text = " ".join(self._option_buf).strip()
            if text and text.lower().rstrip(".…") not in (
                "",
                "select",
                "choose",
                "please select",
                "-",
            ):
                self._select.options.append(text)
            self._option_buf = None
        elif tag == "select":
            self._select = None
        elif tag == "textarea":
            self._textarea = None

    def handle_data(self, data: str) -> None:
        if self._skip:
            return
        if self._option_buf is not None:
            self._option_buf.append(data)
        elif self._textarea is not None:
            self._textarea.value += data
        elif self._label_buf is not None:
            self._label_buf.append(data)
        else:
            text = data.strip()
            if text:
                self._recent = [*self._recent, text][-4:]


class FormFieldsTool(BaseTool):
    name = "extract_form_fields"
    description = (
        "List the questions/fields of an HTML form from a URL or a local .html file: labels, "
        "field types, options for selects/radios/checkboxes, and which are required. "
        "Use it to see exactly what an application or survey asks before drafting answers."
    )
    parameters = (
        ToolParameter("url", "string", "Page URL (http/https)", required=False),
        ToolParameter("path", "string", "Local HTML file in the workspace", required=False),
    )
    permission = Permission.NETWORK

    def __init__(
        self, workspace: Workspace | None = None, client: httpx.AsyncClient | None = None
    ) -> None:
        self.workspace = workspace
        self._client = client or make_http_client()

    async def execute(self, url: str | None = None, path: str | None = None) -> ToolResult:
        if not url and not path:
            raise ToolError("Give either url or path")
        if path:
            if self.workspace is None:
                raise ToolError("No workspace configured for local files")
            target = self.workspace.resolve(path, must_exist=True)
            markup = target.read_text(encoding="utf-8", errors="replace")
            source = self.workspace.relative(target)
        else:
            assert url is not None
            url = validate_url(url)
            try:
                response = await self._client.get(url)
            except httpx.HTTPError as exc:
                raise ToolError(f"Could not fetch {url}: {exc}") from exc
            if response.status_code >= 400:
                raise ToolError(f"{url} returned HTTP {response.status_code}")
            markup = response.text
            source = str(response.url)
        forms = parse_form_fields(markup)
        if not forms:
            return ToolResult.ok(
                f"No form fields found in {source}. The form may be rendered by JavaScript; "
                "paste the questions as text instead."
            )
        lines = [f"{source}: {sum(len(f) for f in forms)} field(s) in {len(forms)} form(s)"]
        for index, fields in enumerate(forms, 1):
            lines.append(f"\nForm {index}:")
            lines.extend(f"  {n}. {fld.describe()}" for n, fld in enumerate(fields, 1))
        data = [[fld.__dict__ for fld in form] for form in forms]
        return ToolResult.ok("\n".join(lines), data=data)


def form_tools(
    workspace: Workspace | None = None, client: httpx.AsyncClient | None = None
) -> list[BaseTool]:
    return [FormFieldsTool(workspace, client)]
