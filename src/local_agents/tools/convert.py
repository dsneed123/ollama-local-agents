"""Convert structured data files between JSON, YAML, CSV and JSON Lines."""

from __future__ import annotations

import csv
import io
import json
from typing import Any

import yaml

from local_agents.errors import ToolError
from local_agents.security.permissions import Permission
from local_agents.security.workspace import Workspace
from local_agents.tools.base import BaseTool, ToolParameter, ToolResult
from local_agents.tools.data import load_dataset

FORMATS = ("json", "yaml", "csv", "jsonl")
_SUFFIX_TO_FORMAT = {
    ".json": "json",
    ".yaml": "yaml",
    ".yml": "yaml",
    ".csv": "csv",
    ".tsv": "csv",
    ".jsonl": "jsonl",
    ".ndjson": "jsonl",
}


def detect_format(name: str) -> str | None:
    dot = name.rfind(".")
    return _SUFFIX_TO_FORMAT.get(name[dot:].lower()) if dot != -1 else None


def load_structured(text: str, fmt: str, label: str = "input") -> Any:
    try:
        if fmt == "json":
            return json.loads(text)
        if fmt == "yaml":
            return yaml.safe_load(text)
        if fmt == "jsonl":
            return [json.loads(line) for line in text.splitlines() if line.strip()]
        if fmt == "csv":
            return list(csv.DictReader(io.StringIO(text)))
    except (ValueError, yaml.YAMLError) as exc:
        raise ToolError(f"Could not parse {label} as {fmt}: {exc}") from exc
    raise ToolError(f"Unsupported format {fmt!r}", hint=f"Use one of: {', '.join(FORMATS)}")


def dump_structured(data: Any, fmt: str) -> str:
    if fmt == "json":
        return json.dumps(data, indent=2, ensure_ascii=False, default=str) + "\n"
    if fmt == "yaml":
        return yaml.safe_dump(data, sort_keys=False, allow_unicode=True)
    if fmt == "jsonl":
        records = data if isinstance(data, list) else [data]
        return "".join(json.dumps(r, ensure_ascii=False, default=str) + "\n" for r in records)
    if fmt == "csv":
        records = data if isinstance(data, list) else [data]
        if not all(isinstance(r, dict) for r in records):
            raise ToolError("CSV output needs a list of objects (rows with named columns)")
        columns: list[str] = []
        for record in records:
            for key in record:
                if key not in columns:
                    columns.append(key)
        buffer = io.StringIO()
        writer = csv.DictWriter(buffer, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        for record in records:
            writer.writerow({k: _cell(v) for k, v in record.items()})
        return buffer.getvalue()
    raise ToolError(f"Unsupported format {fmt!r}", hint=f"Use one of: {', '.join(FORMATS)}")


def _cell(value: Any) -> Any:
    if isinstance(value, (dict, list)):
        return json.dumps(value, ensure_ascii=False)
    return value


class ConvertFileTool(BaseTool):
    name = "convert_file"
    description = (
        "Convert a data file between JSON, YAML, CSV and JSON Lines. Formats are detected "
        "from file extensions unless given explicitly. Never overwrites an existing file."
    )
    parameters = (
        ToolParameter("source", "string", "Input file"),
        ToolParameter("destination", "string", "Output file to create"),
        ToolParameter("from_format", "string", "Input format", required=False, enum=list(FORMATS)),
        ToolParameter("to_format", "string", "Output format", required=False, enum=list(FORMATS)),
    )
    permission = Permission.WRITE

    def __init__(self, workspace: Workspace) -> None:
        self.workspace = workspace

    async def execute(
        self,
        source: str,
        destination: str,
        from_format: str | None = None,
        to_format: str | None = None,
    ) -> ToolResult:
        src = self.workspace.resolve(source, must_exist=True)
        dst = self.workspace.check_write(destination)
        if dst.exists():
            raise ToolError(f"Destination already exists: {destination}")
        src_fmt = from_format or detect_format(src.name)
        dst_fmt = to_format or detect_format(dst.name)
        if not src_fmt or not dst_fmt:
            raise ToolError(
                "Could not detect the file formats from the extensions",
                hint="Pass from_format / to_format explicitly.",
            )
        if src_fmt in ("csv", "jsonl"):
            data: Any = load_dataset(src, label=self.workspace.relative(src)).rows
        else:
            data = load_structured(
                src.read_text(encoding="utf-8-sig"), src_fmt, self.workspace.relative(src)
            )
        output = dump_structured(data, dst_fmt)
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.write_text(output, encoding="utf-8")
        count = len(data) if isinstance(data, list) else 1
        return ToolResult.ok(
            f"Converted {self.workspace.relative(src)} ({src_fmt}) -> "
            f"{self.workspace.relative(dst)} ({dst_fmt}), {count} record(s)"
        )


class ValidateFileTool(BaseTool):
    name = "validate_file"
    description = (
        "Check that a JSON, YAML, CSV or JSON Lines file parses, and describe its structure."
    )
    parameters = (
        ToolParameter("path", "string", "File to validate"),
        ToolParameter("format", "string", "Format override", required=False, enum=list(FORMATS)),
    )
    permission = Permission.READ

    def __init__(self, workspace: Workspace) -> None:
        self.workspace = workspace

    async def execute(self, path: str, format: str | None = None) -> ToolResult:
        target = self.workspace.resolve(path, must_exist=True)
        fmt = format or detect_format(target.name)
        if not fmt:
            raise ToolError("Unknown file format", hint="Pass format explicitly.")
        try:
            data = load_structured(
                target.read_text(encoding="utf-8-sig"), fmt, self.workspace.relative(target)
            )
        except ToolError as exc:
            return ToolResult.fail(str(exc))
        return ToolResult.ok(f"{self.workspace.relative(target)} is valid {fmt}\n{_describe(data)}")


def _describe(data: Any, depth: int = 0) -> str:
    pad = "  " * depth
    if isinstance(data, dict):
        lines = [f"{pad}object with {len(data)} keys"]
        for key, value in list(data.items())[:15]:
            lines.append(f"{pad}  {key}: {_short(value)}")
        return "\n".join(lines)
    if isinstance(data, list):
        kinds = {type(x).__name__ for x in data[:50]}
        sample = (
            f"\n{_describe(data[0], depth + 1)}"
            if data and isinstance(data[0], (dict, list))
            else ""
        )
        return f"{pad}array of {len(data)} items ({', '.join(sorted(kinds)) or 'empty'}){sample}"
    return f"{pad}{_short(data)}"


def _short(value: Any) -> str:
    if isinstance(value, dict):
        return f"object({len(value)} keys)"
    if isinstance(value, list):
        return f"array({len(value)})"
    text = repr(value)
    return text if len(text) <= 60 else text[:57] + "..."


def convert_tools(workspace: Workspace) -> list[BaseTool]:
    return [ValidateFileTool(workspace), ConvertFileTool(workspace)]
