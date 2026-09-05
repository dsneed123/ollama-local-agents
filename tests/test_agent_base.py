from __future__ import annotations

from local_agents.agents.base import AgentConfig, AgentEvent, BaseAgent, parse_text_tool_calls
from local_agents.memory import ConversationMemory
from local_agents.models.base import Message, ToolCall
from local_agents.security.permissions import PermissionSet
from local_agents.security.workspace import Workspace
from local_agents.testing import ScriptedModel, ScriptedTurn
from local_agents.tools.base import BaseTool, ToolParameter, ToolSet
from local_agents.tools.filesystem import filesystem_tools


class EchoTool(BaseTool):
    name = "echo"
    description = "Echo text back"
    parameters = (ToolParameter("text", "string", "Text to echo"),)

    async def execute(self, text: str) -> str:
        return f"echo:{text}"


class Greeter(BaseAgent):
    name = "greeter"
    description = "Says hello"
    system_prompt = "Be brief."


async def test_plain_answer_without_tools() -> None:
    model = ScriptedModel(["Hello there!"])
    agent = Greeter(model=model)
    result = await agent.run("hi")
    assert result.output == "Hello there!"
    assert result.stop_reason == "final" and result.iterations == 1
    assert result.agent == "greeter" and result.model == "scripted-model"
    sent = model.calls[0]["messages"]
    assert sent[0].role == "system" and sent[0].content == "Be brief."
    assert sent[-1] == Message.user("hi")
    assert model.calls[0]["tools"] == []  # no tools -> no specs sent


async def test_tool_loop_with_native_calls() -> None:
    model = ScriptedModel(
        [ScriptedTurn.call("echo", content="Calling…", text="ping"), "Final: pong"]
    )
    agent = Greeter(model=model, tools=[EchoTool()])
    events: list[AgentEvent] = []
    async for event in agent.stream("do it"):
        events.append(event)
    types = [e.type for e in events]
    assert types[0] == "start" and types[-1] == "final"
    assert "tool_call" in types and "tool_result" in types
    result = events[-1].metadata["result"]
    assert result.output == "Final: pong"
    assert result.iterations == 2
    assert result.tool_calls[0].name == "echo" and result.tool_calls[0].result.output == "echo:ping"
    # Second model call saw the tool result message.
    second = model.calls[1]["messages"]
    assert second[-1].role == "tool" and second[-1].content == "echo:ping"
    assert second[-2].tool_calls[0].name == "echo"
    assert model.calls[0]["tools"][0].name == "echo"
    assert "".join(e.content for e in events if e.type == "token") == "Calling…Final: pong"


async def test_text_protocol_for_models_without_native_tools() -> None:
    model = ScriptedModel(
        ['```tool_call\n{"name": "echo", "arguments": {"text": "hi"}}\n```', "done"],
        supports_tools=False,
    )
    agent = Greeter(model=model, tools=[EchoTool()])
    result = await agent.run("go")
    assert result.output == "done"
    assert result.tool_calls[0].result.output == "echo:hi"
    system = model.calls[0]["messages"][0].content
    assert "tool_call" in system and "echo(text: string)" in system
    assert model.calls[0]["tools"] == []  # text protocol: no native specs


async def test_bare_json_tool_call_text_is_handled_and_hidden() -> None:
    model = ScriptedModel(['{"name": "echo", "arguments": {"text": "x"}}\n', "ok"])
    agent = Greeter(model=model, tools=[EchoTool()])
    tokens: list[str] = []
    final = ""
    async for event in agent.stream("go"):
        if event.type == "token":
            tokens.append(event.content)
        if event.type == "final":
            final = event.content
    assert final == "ok"
    assert "".join(tokens) == "ok"  # JSON never reached the UI


async def test_unknown_tool_is_reported_to_model_not_raised() -> None:
    model = ScriptedModel([ScriptedTurn.call("nope"), "recovered"])
    agent = Greeter(model=model, tools=[EchoTool()])
    result = await agent.run("go")
    assert result.output == "recovered"
    assert not result.tool_calls[0].result.success
    assert "not available" in model.calls[1]["messages"][-1].content


async def test_max_iterations_stops_gracefully() -> None:
    model = ScriptedModel([ScriptedTurn.call("echo", content="again", text="x")])
    agent = Greeter(model=model, tools=[EchoTool()], config=AgentConfig(max_iterations=2))
    events = [e async for e in agent.stream("loop")]
    result = events[-1].metadata["result"]
    assert result.stop_reason == "max_iterations"
    assert result.iterations == 2
    assert any(e.type == "error" for e in events)
    assert result.output == "again"


async def test_memory_persists_across_runs_and_reset() -> None:
    model = ScriptedModel(["one", "two"])
    memory = ConversationMemory()
    agent = Greeter(model=model, memory=memory)
    await agent.run("first")
    await agent.run("second")
    roles = [m.role for m in memory.get()]
    assert roles == ["user", "assistant", "user", "assistant"]
    second_call = model.calls[1]["messages"]
    assert [m.content for m in second_call[1:]] == ["first", "one", "second"]
    agent.reset()
    assert memory.get() == []


async def test_remember_tool_calls_option() -> None:
    model = ScriptedModel([ScriptedTurn.call("echo", text="a"), "fin"])
    agent = Greeter(model=model, tools=[EchoTool()], config=AgentConfig(remember_tool_calls=True))
    await agent.run("go")
    roles = [m.role for m in agent.memory.get()]
    assert roles == ["user", "assistant", "tool", "assistant"]


async def test_context_is_appended_to_task() -> None:
    model = ScriptedModel(["ok"])
    agent = Greeter(model=model)
    await agent.run("task", context={"repo": "x"})
    user = model.calls[0]["messages"][-1].content
    assert user.startswith("task") and '"repo": "x"' in user
    await agent.run("task2", context="plain text")
    assert "Context:\nplain text" in model.calls[1]["messages"][-1].content


async def test_thinking_is_collected() -> None:
    model = ScriptedModel([ScriptedTurn(content="ans", thinking="let me think")])
    agent = Greeter(model=model)
    events = [e async for e in agent.stream("q")]
    assert any(e.type == "thinking" and e.content == "let me think" for e in events)
    assert events[-1].metadata["result"].thinking == "let me think"


def test_capabilities_and_workspace_prompt(workspace: Workspace) -> None:
    tools = ToolSet(filesystem_tools(workspace), permissions=PermissionSet(write=True))
    agent = Greeter(model=ScriptedModel(), tools=tools, workspace=workspace)
    caps = agent.get_capabilities()
    assert caps.name == "greeter" and "read_file" in caps.tools and "write" in caps.permissions
    assert caps.can_delegate is False
    prompt = agent.build_system_prompt()
    assert str(workspace.root) in prompt and "must stay inside it" in prompt
    assert repr(agent).startswith("<Greeter")


def test_system_prompt_override_via_config() -> None:
    agent = Greeter(model=ScriptedModel(), config=AgentConfig(system_prompt="Custom."))
    assert agent.build_system_prompt() == "Custom."


def test_parse_text_tool_calls_variants() -> None:
    calls, rest = parse_text_tool_calls(
        'Sure.\n```tool_call\n{"name": "a", "arguments": {"x": 1}}\n```\n'
        '```json\n{"name": "b", "parameters": {"y": 2}}\n```\nThanks'
    )
    assert [c.name for c in calls] == ["a", "b"]
    assert calls[1].arguments == {"y": 2}
    assert rest == "Sure.\n\n\nThanks"
    calls, rest = parse_text_tool_calls('```json\n{"just": "data"}\n```')
    assert calls == [] and "just" in rest
    calls, rest = parse_text_tool_calls('{"name": "c", "arguments": "raw"}')
    assert calls == [ToolCall("c", {"input": "raw"})] and rest == ""
    calls, rest = parse_text_tool_calls("no calls here { at all")
    assert calls == [] and rest == "no calls here { at all"


async def test_agent_survives_supports_tools_failure() -> None:
    class Flaky(ScriptedModel):
        async def supports_tools(self) -> bool:
            raise RuntimeError("provider down")

    model = Flaky(["fine"])
    agent = Greeter(model=model, tools=[EchoTool()])
    result = await agent.run("x")
    assert result.output == "fine"
    assert "tool_call" in model.calls[0]["messages"][0].content  # fell back to text protocol
