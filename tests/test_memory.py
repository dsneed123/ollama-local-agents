from __future__ import annotations

import json
from pathlib import Path

import pytest

from local_agents.memory import ConversationMemory, JSONFileMemory
from local_agents.models.base import Message, ToolCall


def test_conversation_memory_basics() -> None:
    memory = ConversationMemory()
    assert len(memory) == 0
    assert bool(memory) is True
    memory.add(Message.user("a"))
    memory.add(Message.assistant("b"))
    assert [m.content for m in memory.get()] == ["a", "b"]
    assert [m.content for m in memory.get(limit=1)] == ["b"]
    assert memory.get(limit=0) == []
    memory.clear()
    assert memory.get() == []


def test_conversation_memory_window_drops_orphan_tool_results() -> None:
    memory = ConversationMemory(max_messages=3)
    memory.add(Message.user("q"))
    memory.add(Message.assistant("", [ToolCall("t", {})]))
    memory.add(Message.tool("t", "result"))
    memory.add(Message.assistant("final"))
    roles = [m.role for m in memory.get()]
    assert roles[0] != "tool"
    assert roles == ["assistant", "tool", "assistant"] or roles == ["assistant"]


def test_conversation_memory_validates_window() -> None:
    with pytest.raises(ValueError):
        ConversationMemory(max_messages=0)
    unlimited = ConversationMemory(max_messages=None)
    for i in range(500):
        unlimited.add(Message.user(str(i)))
    assert len(unlimited) == 500


def test_json_file_memory_persists(tmp_path: Path) -> None:
    path = tmp_path / "mem" / "history.json"
    memory = JSONFileMemory(path)
    memory.add(Message.user("hello"))
    memory.add(Message.assistant("hi", [ToolCall("t", {"x": 1}, id="c1")]))
    memory.add(Message.tool("t", "out", call_id="c1"))
    assert path.is_file()
    raw = json.loads(path.read_text())
    assert raw[1]["tool_calls"][0]["function"]["name"] == "t"

    reloaded = JSONFileMemory(path)
    messages = reloaded.get()
    assert [m.role for m in messages] == ["user", "assistant", "tool"]
    assert messages[1].tool_calls[0].arguments == {"x": 1}
    assert messages[2].tool_name == "t" and messages[2].tool_call_id == "c1"

    reloaded.clear()
    assert json.loads(path.read_text()) == []


def test_json_file_memory_tolerates_corruption(tmp_path: Path) -> None:
    path = tmp_path / "broken.json"
    path.write_text("{not json")
    memory = JSONFileMemory(path)
    assert memory.get() == []
    memory.add(Message.user("x"))
    assert json.loads(path.read_text())[0]["content"] == "x"


def test_message_roundtrip_with_string_arguments() -> None:
    raw = {
        "role": "assistant",
        "content": "",
        "tool_calls": [{"function": {"name": "t", "arguments": '{"a": 1}'}}],
    }
    message = Message.from_dict(raw)
    assert message.tool_calls[0].arguments == {"a": 1}
    bad = Message.from_dict(
        {"role": "assistant", "tool_calls": [{"function": {"name": "t", "arguments": "oops"}}]}
    )
    assert bad.tool_calls[0].arguments == {"input": "oops"}
