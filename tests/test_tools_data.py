from __future__ import annotations

import json
from pathlib import Path

from local_agents.security.workspace import Workspace
from local_agents.tools.data import (
    DataQueryTool,
    DataStatsTool,
    InspectDataTool,
    data_tools,
    infer_column_type,
    load_dataset,
)


async def test_inspect_csv(workspace: Workspace) -> None:
    result = await InspectDataTool(workspace).run(path="data.csv", sample_rows=2)
    assert result.success
    assert "4 rows, 3 columns" in result.output
    assert result.data["types"] == {"region": "string", "revenue": "integer", "units": "integer"}
    assert "north | 100 | 5" in result.output


async def test_stats(workspace: Workspace) -> None:
    result = await DataStatsTool(workspace).run(path="data.csv", columns=["revenue", "region"])
    assert result.success
    revenue = result.data[0]
    assert revenue["count"] == 3 and revenue["missing"] == 1
    assert revenue["min"] == 50 and revenue["max"] == 250 and revenue["sum"] == 400
    region = result.data[1]
    assert region["unique"] == 3 and region["top_values"][0] == ("north", 2)


async def test_query_filter_group_sort(workspace: Workspace) -> None:
    tool = DataQueryTool(workspace)
    filtered = await tool.run(
        path="data.csv", filter_column="revenue", filter_op=">", filter_value="60"
    )
    assert filtered.data["total"] == 2
    grouped = await tool.run(
        path="data.csv",
        group_by="region",
        aggregate="sum",
        aggregate_column="revenue",
        sort_by="sum_revenue",
        descending=True,
    )
    assert grouped.data["columns"] == ["region", "sum_revenue"]
    assert grouped.data["rows"][0] == {"region": "south", "sum_revenue": 250}
    assert grouped.data["rows"][1] == {"region": "north", "sum_revenue": 150}
    counted = await tool.run(path="data.csv", group_by="region", aggregate="count")
    assert {r["region"]: r["count"] for r in counted.data["rows"]} == {
        "north": 2,
        "south": 1,
        "east": 1,
    }
    contains = await tool.run(
        path="data.csv", filter_column="region", filter_op="contains", filter_value="ORTH"
    )
    assert contains.data["total"] == 2


async def test_query_errors(workspace: Workspace) -> None:
    tool = DataQueryTool(workspace)
    bad_col = await tool.run(path="data.csv", filter_column="nope", filter_value="1")
    assert not bad_col.success and "Unknown column" in (bad_col.error or "")
    bad_agg = await tool.run(path="data.csv", group_by="region", aggregate="sum")
    assert not bad_agg.success and "aggregate_column" in (bad_agg.error or "")
    bad_op = await tool.run(
        path="data.csv", filter_column="region", filter_op="~", filter_value="x"
    )
    assert not bad_op.success
    bad_sort = await tool.run(path="data.csv", sort_by="nope")
    assert not bad_sort.success


async def test_json_and_jsonl(workspace: Workspace) -> None:
    (workspace.root / "items.json").write_text(json.dumps({"data": [{"a": 1, "b": "x"}, {"a": 2}]}))
    (workspace.root / "rows.jsonl").write_text('{"a": 1}\n\n{"a": 3, "c": true}\n')
    js = await InspectDataTool(workspace).run(path="items.json")
    assert js.data["columns"] == ["a", "b"] and js.data["rows"] == 2
    jl = await InspectDataTool(workspace).run(path="rows.jsonl")
    assert jl.data["columns"] == ["a", "c"] and jl.data["types"]["c"] == "boolean"
    (workspace.root / "bad.json").write_text("{oops")
    bad = await InspectDataTool(workspace).run(path="bad.json")
    assert not bad.success and "Invalid JSON" in (bad.error or "")
    (workspace.root / "x.parquet").write_bytes(b"PAR1")
    unsupported = await InspectDataTool(workspace).run(path="x.parquet")
    assert not unsupported.success and "Unsupported" in (unsupported.error or "")


def test_load_dataset_tsv_and_type_inference(tmp_path: Path) -> None:
    path = tmp_path / "t.tsv"
    path.write_text("name\tscore\nann\t1.5\nbob\t2\n")
    ds = load_dataset(path)
    assert ds.columns == ["name", "score"]
    assert ds.types == {"name": "string", "score": "number"}
    assert infer_column_type([]) == "empty"
    assert infer_column_type([{"a": 1}]) == "object"
    assert infer_column_type(["1,000", "2"]) == "integer"


def test_factory(workspace: Workspace) -> None:
    assert [t.name for t in data_tools(workspace)] == ["inspect_data", "data_stats", "query_data"]
