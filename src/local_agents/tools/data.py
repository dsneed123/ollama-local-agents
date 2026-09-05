"""Data tools for CSV / JSON / JSON Lines files. Standard library only."""

from __future__ import annotations

import csv
import io
import json
import statistics
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from local_agents.errors import ToolError
from local_agents.security.permissions import Permission
from local_agents.security.workspace import Workspace
from local_agents.tools.base import BaseTool, ToolParameter, ToolResult

Row = dict[str, Any]


@dataclass
class Dataset:
    """A loaded tabular dataset."""

    rows: list[Row]
    columns: list[str]
    source: str
    format: str
    types: dict[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.types:
            self.types = {c: infer_column_type(self.column(c)) for c in self.columns}

    def column(self, name: str) -> list[Any]:
        if name not in self.columns:
            raise ToolError(f"Unknown column {name!r}", hint=f"Columns: {', '.join(self.columns)}")
        return [row.get(name) for row in self.rows]

    def numeric_column(self, name: str) -> list[float]:
        values = []
        for value in self.column(name):
            number = to_number(value)
            if number is not None:
                values.append(number)
        return values


def to_number(value: Any) -> float | None:
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        text = value.strip().replace(",", "")
        if not text:
            return None
        try:
            return float(text)
        except ValueError:
            return None
    return None


def infer_column_type(values: list[Any]) -> str:
    non_empty = [v for v in values if v not in (None, "")]
    if not non_empty:
        return "empty"
    if all(isinstance(v, bool) for v in non_empty):
        return "boolean"
    numbers = [to_number(v) for v in non_empty]
    if all(n is not None for n in numbers):
        return (
            "integer" if all(float(n).is_integer() for n in numbers if n is not None) else "number"
        )
    if all(isinstance(v, (dict, list)) for v in non_empty):
        return "object"
    return "string"


def load_dataset(path: Path, *, max_rows: int | None = None, label: str | None = None) -> Dataset:
    suffix = path.suffix.lower()
    text = path.read_text(encoding="utf-8-sig", errors="replace")
    label = label or str(path)
    if suffix in (".jsonl", ".ndjson"):
        return _load_jsonl(text, label, max_rows)
    if suffix == ".json":
        return _load_json(text, label, max_rows)
    if suffix in (".csv", ".tsv", ".txt", ""):
        return _load_csv(text, label, max_rows, delimiter="\t" if suffix == ".tsv" else None)
    raise ToolError(
        f"Unsupported data format: {suffix or 'no extension'}", hint="Use CSV, TSV, JSON or JSONL."
    )


def _load_csv(text: str, label: str, max_rows: int | None, *, delimiter: str | None) -> Dataset:
    sample = text[:4096]
    if delimiter is None:
        try:
            delimiter = csv.Sniffer().sniff(sample, delimiters=",;\t|").delimiter
        except csv.Error:
            delimiter = ","
    reader = csv.DictReader(io.StringIO(text), delimiter=delimiter)
    rows: list[Row] = []
    for row in reader:
        rows.append({k: (v if v != "" else None) for k, v in row.items() if k is not None})
        if max_rows and len(rows) >= max_rows:
            break
    columns = list(reader.fieldnames or [])
    return Dataset(rows=rows, columns=columns, source=label, format="csv")


def _load_json(text: str, label: str, max_rows: int | None) -> Dataset:
    try:
        data = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ToolError(f"Invalid JSON in {label}: {exc}") from exc
    if isinstance(data, dict):
        # Common shapes: {"data": [...]}, {"items": [...]}, or a single record.
        for key in ("data", "items", "results", "records", "rows"):
            if isinstance(data.get(key), list):
                data = data[key]
                break
        else:
            data = [data]
    if not isinstance(data, list):
        raise ToolError(f"{label} does not contain a list of records")
    return _rows_to_dataset(
        [r if isinstance(r, dict) else {"value": r} for r in data], label, "json", max_rows
    )


def _load_jsonl(text: str, label: str, max_rows: int | None) -> Dataset:
    rows: list[Row] = []
    for number, line in enumerate(text.splitlines(), 1):
        if not line.strip():
            continue
        try:
            item = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ToolError(f"Invalid JSON on line {number} of {label}: {exc}") from exc
        rows.append(item if isinstance(item, dict) else {"value": item})
        if max_rows and len(rows) >= max_rows:
            break
    return _rows_to_dataset(rows, label, "jsonl", None)


def _rows_to_dataset(rows: list[Row], label: str, fmt: str, max_rows: int | None) -> Dataset:
    if max_rows:
        rows = rows[:max_rows]
    columns: list[str] = []
    for row in rows:
        for key in row:
            if key not in columns:
                columns.append(key)
    return Dataset(rows=rows, columns=columns, source=label, format=fmt)


# --------------------------------------------------------------------------- #
# Analysis helpers
# --------------------------------------------------------------------------- #


def describe_column(dataset: Dataset, name: str) -> dict[str, Any]:
    values = dataset.column(name)
    missing = sum(1 for v in values if v in (None, ""))
    info: dict[str, Any] = {
        "column": name,
        "type": dataset.types.get(name, "string"),
        "count": len(values) - missing,
        "missing": missing,
    }
    if info["type"] in ("integer", "number"):
        numbers = dataset.numeric_column(name)
        if numbers:
            info.update(
                {
                    "min": min(numbers),
                    "max": max(numbers),
                    "mean": round(statistics.fmean(numbers), 4),
                    "median": statistics.median(numbers),
                    "std": round(statistics.pstdev(numbers), 4) if len(numbers) > 1 else 0.0,
                    "sum": sum(numbers),
                }
            )
    else:
        counter = Counter(str(v) for v in values if v not in (None, ""))
        info["unique"] = len(counter)
        info["top_values"] = counter.most_common(5)
    return info


def _fmt_row(row: Row, columns: list[str]) -> str:
    return " | ".join(str(row.get(c, "")) for c in columns)


def format_table(rows: list[Row], columns: list[str], limit: int = 20) -> str:
    lines = [" | ".join(columns), "-" * min(80, len(" | ".join(columns)))]
    for row in rows[:limit]:
        lines.append(_fmt_row(row, columns))
    if len(rows) > limit:
        lines.append(f"... ({len(rows) - limit} more rows)")
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# Tools
# --------------------------------------------------------------------------- #


class _DataTool(BaseTool):
    def __init__(self, workspace: Workspace) -> None:
        self.workspace = workspace

    def _load(self, path: str, max_rows: int | None = None) -> Dataset:
        target = self.workspace.resolve(path, must_exist=True)
        if target.stat().st_size > self.workspace.max_file_bytes * 10:
            raise ToolError(f"{path} is too large to analyse in memory")
        return load_dataset(target, max_rows=max_rows, label=self.workspace.relative(target))


class InspectDataTool(_DataTool):
    name = "inspect_data"
    description = (
        "Inspect a CSV/TSV/JSON/JSONL file: row count, columns, inferred types, and sample rows."
    )
    parameters = (
        ToolParameter("path", "string", "Data file path"),
        ToolParameter("sample_rows", "integer", "Sample rows to show", required=False, default=5),
    )
    permission = Permission.READ

    async def execute(self, path: str, sample_rows: int = 5) -> ToolResult:
        ds = self._load(path)
        lines = [
            f"{ds.source}: {len(ds.rows)} rows, {len(ds.columns)} columns ({ds.format})",
            "",
            "Columns:",
        ]
        lines.extend(f"  - {c} ({ds.types.get(c, '?')})" for c in ds.columns)
        lines.append("")
        lines.append(f"Sample ({min(sample_rows, len(ds.rows))} rows):")
        lines.append(format_table(ds.rows, ds.columns, limit=sample_rows))
        return ToolResult.ok(
            "\n".join(lines),
            data={"rows": len(ds.rows), "columns": ds.columns, "types": ds.types},
        )


class DataStatsTool(_DataTool):
    name = "data_stats"
    description = (
        "Compute descriptive statistics for columns in a data file: count, missing, "
        "min/max/mean/median/std for numeric columns and top values for categorical ones."
    )
    parameters = (
        ToolParameter("path", "string", "Data file path"),
        ToolParameter(
            "columns", "array", "Columns to describe (default: all)", required=False, items="string"
        ),
    )
    permission = Permission.READ

    async def execute(self, path: str, columns: list[str] | None = None) -> ToolResult:
        ds = self._load(path)
        chosen = columns or ds.columns
        stats = [describe_column(ds, c) for c in chosen]
        lines = [f"{ds.source}: {len(ds.rows)} rows", ""]
        for info in stats:
            lines.append(
                f"{info['column']} ({info['type']}): "
                f"count={info['count']} missing={info['missing']}"
            )
            if "mean" in info:
                lines.append(
                    f"    min={info['min']} max={info['max']} mean={info['mean']} "
                    f"median={info['median']} std={info['std']} sum={info['sum']}"
                )
            elif "unique" in info:
                top = ", ".join(f"{v} ({n})" for v, n in info["top_values"])
                lines.append(f"    unique={info['unique']} top: {top}")
        return ToolResult.ok("\n".join(lines), data=stats)


class DataQueryTool(_DataTool):
    name = "query_data"
    description = (
        "Filter, group, aggregate and sort rows of a data file. "
        "Examples: filter rows where status == 'active'; group_by 'region' with aggregate "
        "'sum' of 'revenue'; sort_by 'revenue' descending; limit 10."
    )
    parameters = (
        ToolParameter("path", "string", "Data file path"),
        ToolParameter("filter_column", "string", "Column to filter on", required=False),
        ToolParameter(
            "filter_op",
            "string",
            "Comparison operator",
            required=False,
            enum=["==", "!=", ">", ">=", "<", "<=", "contains"],
        ),
        ToolParameter("filter_value", "string", "Value to compare against", required=False),
        ToolParameter("group_by", "string", "Column to group rows by", required=False),
        ToolParameter(
            "aggregate",
            "string",
            "Aggregate function applied per group",
            required=False,
            enum=["count", "sum", "mean", "min", "max", "median"],
        ),
        ToolParameter("aggregate_column", "string", "Numeric column to aggregate", required=False),
        ToolParameter("sort_by", "string", "Column to sort by", required=False),
        ToolParameter("descending", "boolean", "Sort descending", required=False, default=False),
        ToolParameter("limit", "integer", "Maximum rows to return", required=False, default=20),
    )
    permission = Permission.READ

    async def execute(
        self,
        path: str,
        filter_column: str | None = None,
        filter_op: str | None = None,
        filter_value: str | None = None,
        group_by: str | None = None,
        aggregate: str | None = None,
        aggregate_column: str | None = None,
        sort_by: str | None = None,
        descending: bool = False,
        limit: int = 20,
    ) -> ToolResult:
        ds = self._load(path)
        rows = ds.rows
        if filter_column:
            rows = _filter_rows(ds, rows, filter_column, filter_op or "==", filter_value)
        columns = ds.columns
        if group_by:
            rows, columns = _group_rows(ds, rows, group_by, aggregate or "count", aggregate_column)
        if sort_by:
            if sort_by not in columns:
                raise ToolError(
                    f"Cannot sort by unknown column {sort_by!r}",
                    hint=f"Columns: {', '.join(columns)}",
                )
            rows = sorted(rows, key=lambda r: _sort_key(r.get(sort_by)), reverse=descending)
        total = len(rows)
        text = f"{total} rows" + (f" (showing {min(limit, total)})" if total > limit else "") + "\n"
        text += format_table(rows, columns, limit=limit)
        return ToolResult.ok(text, data={"rows": rows[:limit], "columns": columns, "total": total})


def _filter_rows(ds: Dataset, rows: list[Row], column: str, op: str, raw_value: Any) -> list[Row]:
    if column not in ds.columns:
        raise ToolError(f"Unknown column {column!r}", hint=f"Columns: {', '.join(ds.columns)}")
    target_number = to_number(raw_value)
    out: list[Row] = []
    for row in rows:
        value = row.get(column)
        if op == "contains":
            if raw_value is not None and str(raw_value).lower() in str(value or "").lower():
                out.append(row)
            continue
        number = to_number(value)
        left: Any
        right: Any
        if target_number is not None and number is not None:
            left, right = number, target_number
        else:
            left = str(value if value is not None else "")
            right = str(raw_value if raw_value is not None else "")
        keep = {
            "==": left == right,
            "!=": left != right,
            ">": left > right,
            ">=": left >= right,
            "<": left < right,
            "<=": left <= right,
        }.get(op)
        if keep is None:
            raise ToolError(f"Unsupported operator {op!r}")
        if keep:
            out.append(row)
    return out


def _group_rows(
    ds: Dataset, rows: list[Row], group_by: str, aggregate: str, column: str | None
) -> tuple[list[Row], list[str]]:
    if group_by not in ds.columns:
        raise ToolError(f"Unknown column {group_by!r}", hint=f"Columns: {', '.join(ds.columns)}")
    if aggregate != "count" and not column:
        raise ToolError(f"aggregate_column is required for aggregate={aggregate!r}")
    if column and column not in ds.columns:
        raise ToolError(f"Unknown column {column!r}", hint=f"Columns: {', '.join(ds.columns)}")
    groups: dict[str, list[Row]] = {}
    for row in rows:
        groups.setdefault(str(row.get(group_by)), []).append(row)
    label = "count" if aggregate == "count" else f"{aggregate}_{column}"
    out: list[Row] = []
    for key, members in groups.items():
        if aggregate == "count":
            value: Any = len(members)
        else:
            numbers = [
                n for n in (to_number(m.get(column or "")) for m in members) if n is not None
            ]
            if not numbers:
                value = None
            elif aggregate == "sum":
                value = sum(numbers)
            elif aggregate == "mean":
                value = round(statistics.fmean(numbers), 4)
            elif aggregate == "min":
                value = min(numbers)
            elif aggregate == "max":
                value = max(numbers)
            else:
                value = statistics.median(numbers)
        out.append({group_by: key, label: value})
    return out, [group_by, label]


def _sort_key(value: Any) -> tuple[int, Any]:
    """Order: missing values, then numbers, then text."""
    if value is None or value == "":
        return (-1, "")
    number = to_number(value)
    if number is not None:
        return (0, number)
    return (1, str(value))


def data_tools(workspace: Workspace) -> list[BaseTool]:
    return [InspectDataTool(workspace), DataStatsTool(workspace), DataQueryTool(workspace)]
