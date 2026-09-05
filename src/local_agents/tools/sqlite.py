"""Read-only SQLite tools (standard library ``sqlite3``)."""

from __future__ import annotations

import sqlite3

from local_agents.errors import ToolError
from local_agents.security.permissions import Permission
from local_agents.security.workspace import Workspace
from local_agents.tools.base import BaseTool, ToolParameter, ToolResult
from local_agents.tools.data import format_table


class _SQLiteTool(BaseTool):
    def __init__(self, workspace: Workspace) -> None:
        self.workspace = workspace

    def _connect(self, path: str) -> sqlite3.Connection:
        target = self.workspace.resolve(path, must_exist=True)
        if target.is_dir():
            raise IsADirectoryError(f"{path} is a directory")
        try:
            # mode=ro guarantees the agent can never modify the database.
            conn = sqlite3.connect(f"{target.as_uri()}?mode=ro", uri=True, timeout=5)
        except sqlite3.Error as exc:
            raise ToolError(f"Could not open {path}: {exc}") from exc
        conn.row_factory = sqlite3.Row
        try:
            conn.execute("SELECT 1 FROM sqlite_master LIMIT 1")
        except sqlite3.DatabaseError as exc:
            conn.close()
            raise ToolError(f"{path} is not a SQLite database: {exc}") from exc
        return conn


class SQLiteSchemaTool(_SQLiteTool):
    name = "sqlite_schema"
    description = "List the tables, columns and row counts of a SQLite database file."
    parameters = (ToolParameter("path", "string", "Path to the .db / .sqlite file"),)
    permission = Permission.READ

    async def execute(self, path: str) -> ToolResult:
        conn = self._connect(path)
        try:
            tables = [
                row["name"]
                for row in conn.execute(
                    "SELECT name FROM sqlite_master WHERE type IN ('table','view') "
                    "AND name NOT LIKE 'sqlite_%' ORDER BY name"
                )
            ]
            lines = [
                f"{self.workspace.relative(self.workspace.resolve(path))}: {len(tables)} tables"
            ]
            schema: dict[str, list[str]] = {}
            for table in tables:
                columns = conn.execute(f'PRAGMA table_info("{table}")').fetchall()
                count = conn.execute(f'SELECT COUNT(*) FROM "{table}"').fetchone()[0]
                cols = [f"{c['name']} {c['type']}{' PK' if c['pk'] else ''}" for c in columns]
                schema[table] = cols
                lines.append(f"\n{table} ({count} rows)")
                lines.extend(f"  - {c}" for c in cols)
        finally:
            conn.close()
        return ToolResult.ok("\n".join(lines), data=schema)


class SQLiteQueryTool(_SQLiteTool):
    name = "sqlite_query"
    description = (
        "Run a read-only SQL query (SELECT / WITH / EXPLAIN) against a SQLite database "
        "file and return the rows. The database is opened read-only."
    )
    parameters = (
        ToolParameter("path", "string", "Path to the .db / .sqlite file"),
        ToolParameter("sql", "string", "SQL query to run"),
        ToolParameter("limit", "integer", "Maximum rows to return", required=False, default=50),
    )
    permission = Permission.READ

    async def execute(self, path: str, sql: str, limit: int = 50) -> ToolResult:
        statement = sql.strip().rstrip(";")
        if not statement.lower().startswith(("select", "with", "explain", "pragma")):
            raise ToolError(
                "Only read-only statements (SELECT, WITH, EXPLAIN, PRAGMA) are allowed",
                hint="The database is opened in read-only mode.",
            )
        conn = self._connect(path)
        try:
            try:
                cursor = conn.execute(statement)
                rows = cursor.fetchmany(max(1, limit) + 1)
            except sqlite3.Error as exc:
                raise ToolError(f"SQL error: {exc}") from exc
            columns = [d[0] for d in cursor.description] if cursor.description else []
        finally:
            conn.close()
        truncated = len(rows) > limit
        records = [dict(zip(columns, row, strict=False)) for row in rows[:limit]]
        header = f"{len(records)} row(s)" + (" (limit reached)" if truncated else "")
        return ToolResult.ok(
            header + "\n" + format_table(records, columns, limit=limit),
            data={"columns": columns, "rows": records, "truncated": truncated},
        )


def sqlite_tools(workspace: Workspace) -> list[BaseTool]:
    return [SQLiteSchemaTool(workspace), SQLiteQueryTool(workspace)]
