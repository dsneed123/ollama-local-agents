"""Tests for the text, sqlite, convert, calculator and http tools."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import httpx
import pytest

from local_agents.errors import ToolError
from local_agents.security.workspace import Workspace
from local_agents.tools.calculator import CalculatorTool, evaluate
from local_agents.tools.convert import ConvertFileTool, ValidateFileTool, detect_format
from local_agents.tools.sqlite import SQLiteQueryTool, SQLiteSchemaTool
from local_agents.tools.text import CountMatchesTool, RegexTestTool, TailFileTool
from local_agents.tools.web import HttpRequestTool, make_http_client

# --------------------------------------------------------------------------- #
# text
# --------------------------------------------------------------------------- #


async def test_regex_tool_matches_groups_and_substitution() -> None:
    result = await RegexTestTool().run(
        pattern=r"(?P<user>\w+)@(\w+)\.com", text="a@x.com, b@y.com", replacement=r"\1 at \2"
    )
    assert result.success and "2 match(es)" in result.output
    assert result.data[0]["groups"] == ["a", "x"]
    assert "a at x, b at y" in result.output
    bad = await RegexTestTool().run(pattern="(", text="x")
    assert not bad.success and "Invalid regular expression" in (bad.error or "")
    flags = await RegexTestTool().run(pattern="^b", text="A\nb", ignore_case=True, multiline=True)
    assert "1 match(es)" in flags.output


async def test_count_matches_and_tail(workspace: Workspace) -> None:
    log = workspace.root / "app.log"
    lines = [f"2026-01-01 INFO ok {i}" for i in range(30)] + [
        "2026-01-01 ERROR TimeoutError upstream",
        "2026-01-01 ERROR ValueError bad input",
        "2026-01-01 ERROR TimeoutError upstream",
        "2026-01-01 WARN slow",
    ]
    log.write_text("\n".join(lines) + "\n")
    counts = await CountMatchesTool(workspace).run(path="app.log", pattern=r"\b(ERROR|WARN|INFO)\b")
    assert counts.success and counts.data["top"][0] == ("INFO", 30)
    assert ("ERROR", 3) in counts.data["top"]
    types = await CountMatchesTool(workspace).run(path="app.log", pattern=r"ERROR (\w+)", top=1)
    assert types.data["top"] == [("TimeoutError", 2)]
    tail = await TailFileTool(workspace).run(path="app.log", lines=2)
    assert tail.success and tail.data == [
        "2026-01-01 ERROR TimeoutError upstream\n",
        "2026-01-01 WARN slow\n",
    ]
    big = await TailFileTool(workspace).run(path="app.log", lines=1000)
    assert len(big.data) == 34
    outside = await TailFileTool(workspace).run(path="../x")
    assert outside.blocked


# --------------------------------------------------------------------------- #
# sqlite
# --------------------------------------------------------------------------- #


@pytest.fixture
def db(workspace: Workspace) -> Path:
    path = workspace.root / "shop.db"
    conn = sqlite3.connect(path)
    conn.execute("CREATE TABLE orders (id INTEGER PRIMARY KEY, region TEXT, total REAL)")
    conn.executemany(
        "INSERT INTO orders (region, total) VALUES (?, ?)",
        [("north", 10.0), ("north", 5.0), ("south", 20.0)],
    )
    conn.commit()
    conn.close()
    return path


async def test_sqlite_schema_and_query(workspace: Workspace, db: Path) -> None:
    schema = await SQLiteSchemaTool(workspace).run(path="shop.db")
    assert schema.success and "orders (3 rows)" in schema.output
    assert schema.data["orders"][0] == "id INTEGER PK"
    query = await SQLiteQueryTool(workspace).run(
        path="shop.db",
        sql="SELECT region, SUM(total) AS t FROM orders GROUP BY region ORDER BY t DESC",
    )
    assert query.success
    assert query.data["rows"] == [{"region": "south", "t": 20.0}, {"region": "north", "t": 15.0}]
    limited = await SQLiteQueryTool(workspace).run(
        path="shop.db", sql="SELECT * FROM orders", limit=2
    )
    assert limited.data["truncated"] and "limit reached" in limited.output


async def test_sqlite_is_read_only(workspace: Workspace, db: Path) -> None:
    tool = SQLiteQueryTool(workspace)
    refused = await tool.run(path="shop.db", sql="DELETE FROM orders")
    assert not refused.success and "read-only" in (refused.error or "")
    # Even a sneaky read-only-looking statement cannot write: the connection is mode=ro.
    sneaky = await tool.run(
        path="shop.db", sql="WITH x AS (SELECT 1) INSERT INTO orders (region) VALUES ('z')"
    )
    assert not sneaky.success
    assert sqlite3.connect(db).execute("SELECT COUNT(*) FROM orders").fetchone()[0] == 3
    bad_sql = await tool.run(path="shop.db", sql="SELECT nope FROM orders")
    assert not bad_sql.success and "SQL error" in (bad_sql.error or "")
    (workspace.root / "not.db").write_text("hello")
    not_db = await SQLiteSchemaTool(workspace).run(path="not.db")
    assert not not_db.success


# --------------------------------------------------------------------------- #
# convert
# --------------------------------------------------------------------------- #


async def test_convert_and_validate(workspace: Workspace) -> None:
    conv = ConvertFileTool(workspace)
    result = await conv.run(source="data.csv", destination="out/data.json")
    assert result.success and "4 record(s)" in result.output
    data = json.loads((workspace.root / "out" / "data.json").read_text())
    assert data[0] == {"region": "north", "revenue": "100", "units": "5"}
    to_yaml = await conv.run(source="out/data.json", destination="data.yaml")
    assert to_yaml.success and "region: north" in (workspace.root / "data.yaml").read_text()
    back = await conv.run(source="data.yaml", destination="round.csv")
    assert back.success and (workspace.root / "round.csv").read_text().startswith(
        "region,revenue,units"
    )
    jsonl = await conv.run(source="out/data.json", destination="rows.jsonl")
    assert jsonl.success and len((workspace.root / "rows.jsonl").read_text().splitlines()) == 4
    clash = await conv.run(source="data.csv", destination="data.yaml")
    assert not clash.success and "already exists" in (clash.error or "")
    unknown = await conv.run(source="data.csv", destination="x.bin")
    assert not unknown.success and "detect" in (unknown.error or "")
    nested = workspace.root / "nested.json"
    nested.write_text(json.dumps({"a": {"b": 1}}))
    nested_csv = await conv.run(source="nested.json", destination="nested.csv")
    assert nested_csv.success  # one row; the nested object is JSON-encoded in its cell
    assert (workspace.root / "nested.csv").read_text().splitlines()[1] == '"{""b"": 1}"'
    (workspace.root / "scalars.json").write_text("[1, 2, 3]")
    bad_csv = await conv.run(source="scalars.json", destination="scalars.csv")
    assert not bad_csv.success and "list of objects" in (bad_csv.error or "")

    valid = await ValidateFileTool(workspace).run(path="nested.json")
    assert valid.success and "object with 1 keys" in valid.output
    (workspace.root / "broken.yaml").write_text("a: [1,\n")
    invalid = await ValidateFileTool(workspace).run(path="broken.yaml")
    assert not invalid.success and "Could not parse" in (invalid.error or "")
    assert detect_format("x.yml") == "yaml" and detect_format("noext") is None


# --------------------------------------------------------------------------- #
# calculator
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    ("expr", "value"),
    [
        ("2 + 3 * 4", 14),
        ("(17.5 * 3) ** 2", 2756.25),
        ("sqrt(16) + abs(-2)", 6.0),
        ("factorial(5) // 7 % 4", 1),
        ("round(pi, 3)", 3.142),
        ("max(1, 9, 3) - min(4, 2)", 7),
        ("comb(5, 2) * gcd(12, 18)", 60),
        ("-2 ** 2", -4),
        ("sum([1, 2, 3])", 6),
    ],
)
def test_evaluate(expr: str, value: float) -> None:
    assert evaluate(expr) == pytest.approx(value)


@pytest.mark.parametrize(
    "expr",
    [
        "__import__('os')",
        "x + 1",
        "1 / 0",
        "2 ** 99999",
        "factorial(100000)",
        "foo(1)",
        "a.b",
        "(1",
        "round(1, ndigits=2)",
    ],
)
def test_evaluate_rejects_unsafe_or_invalid(expr: str) -> None:
    with pytest.raises(ToolError):
        evaluate(expr)


async def test_calculator_tool_formatting() -> None:
    result = await CalculatorTool().run(expression="10 / 4")
    assert result.output == "10 / 4 = 2.5" and result.data["result"] == 2.5
    whole = await CalculatorTool().run(expression="6 / 3")
    assert whole.output == "6 / 3 = 2"
    big = await CalculatorTool().run(expression="2 ** 100")
    assert big.output.endswith("= 1267650600228229401496703205376")


# --------------------------------------------------------------------------- #
# http
# --------------------------------------------------------------------------- #


async def test_http_request_tool() -> None:
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        if request.method == "POST":
            return httpx.Response(
                201,
                json={
                    "echo": json.loads(request.content),
                    "auth": request.headers.get("authorization"),
                },
            )
        if request.url.path == "/missing":
            return httpx.Response(404, text="nope")
        return httpx.Response(200, text="plain", headers={"content-type": "text/plain"})

    tool = HttpRequestTool(make_http_client(transport=httpx.MockTransport(handler)))
    get = await tool.run(url="https://api.test/x")
    assert get.success and "Status: 200" in get.output and "plain" in get.output
    post = await tool.run(
        url="https://api.test/items",
        method="POST",
        json_body={"a": 1},
        headers={"Authorization": "Bearer t"},
    )
    assert post.success and post.data["status"] == 201
    assert '"echo": {\n    "a": 1\n  }' in post.output and '"auth": "Bearer t"' in post.output
    missing = await tool.run(url="https://api.test/missing")
    assert not missing.success and missing.error == "HTTP 404"
    raw = await tool.run(url="https://api.test/raw", method="PUT", body="x=1")
    assert raw.success and seen[-1].content == b"x=1"
    bad = await tool.run(url="file:///etc/passwd")
    assert not bad.success
