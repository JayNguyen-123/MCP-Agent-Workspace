"""Graph wiring tests with scripted models and fake tools (no LLM calls)."""

import asyncio
from typing import Any

import pytest

pytest.importorskip("langgraph")

from langchain_core.language_models import BaseChatModel  # noqa: E402
from langchain_core.messages import AIMessage, ToolMessage  # noqa: E402
from langchain_core.outputs import ChatGeneration, ChatResult  # noqa: E402
from langchain_core.runnables import RunnableLambda  # noqa: E402
from langchain_core.tools import StructuredTool  # noqa: E402
from langgraph.checkpoint.memory import InMemorySaver  # noqa: E402
from langgraph.types import Command  # noqa: E402

from agent_workspace.graph import AgentModels, ReviewVerdict, RouteDecision, build_graph, final_answer  # noqa: E402


class Scripted(BaseChatModel):
    """Returns pre-scripted outputs in order; supports tools and structured output."""

    script: list[Any]

    @property
    def _llm_type(self) -> str:
        return "scripted"

    def _next(self) -> Any:
        assert self.script, "model called more times than scripted"
        return self.script.pop(0)

    def _generate(self, messages, stop=None, run_manager=None, **kwargs) -> ChatResult:
        return ChatResult(generations=[ChatGeneration(message=self._next())])

    def bind_tools(self, tools, **kwargs):
        return self

    def with_structured_output(self, schema, *, include_raw: bool = False, **kwargs):
        return RunnableLambda(lambda _: {"raw": AIMessage(""), "parsed": self._next(), "parsing_error": None})


def tool_call(name: str, args: dict, call_id: str) -> AIMessage:
    return AIMessage(content="", tool_calls=[{"name": name, "args": args, "id": call_id, "type": "tool_call"}])


class Calls:
    def __init__(self) -> None:
        self.run: list[dict] = []
        self.install: list[str] = []


def make_tools(calls: Calls):
    def save_and_run_python_code(filename: str, code_content: str) -> str:
        """Run code."""
        calls.run.append({"filename": filename})
        return "Execution finished (exit code 0).\n--- stdout ---\n8"

    def install_project_dependency(package_name: str) -> str:
        """Install a package."""
        calls.install.append(package_name)
        return f"Installed {package_name!r} successfully."

    return [
        StructuredTool.from_function(save_and_run_python_code),
        StructuredTool.from_function(install_project_dependency),
    ]


def graph_for(supervisor, coder, reviewer, calls, max_hops=25):
    models = AgentModels(Scripted(script=supervisor), Scripted(script=coder), Scripted(script=reviewer))
    return build_graph(models, make_tools(calls), max_hops=max_hops, checkpointer=InMemorySaver())


CFG = {"configurable": {"thread_id": "t1"}, "recursion_limit": 100}
START = {"messages": [("user", "compute 4+4")], "hops": 0}


def test_happy_path_routes_through_reviewer():
    calls = Calls()
    g = graph_for(
        supervisor=[RouteDecision(next="coder", instruction="do it"), RouteDecision(next="finish"),
                    RouteDecision(next="finish")],
        coder=[tool_call("save_and_run_python_code", {"filename": "a.py", "code_content": "print(4+4)"}, "c1"),
               AIMessage("Result: 8")],
        reviewer=[ReviewVerdict(passed=True)],
        calls=calls,
    )  # fmt: skip
    out = asyncio.run(g.ainvoke(START, CFG))
    assert calls.run == [{"filename": "a.py"}]
    assert out["review_passed"] is True  # premature "finish" was forced through review
    assert final_answer(out["messages"]) == "Result: 8"
    assert not out.get("terminated_reason")


def test_install_pauses_and_denial_never_executes():
    calls = Calls()
    g = graph_for(
        supervisor=[RouteDecision(next="coder"), RouteDecision(next="finish"), RouteDecision(next="finish")],
        coder=[
            tool_call("install_project_dependency", {"package_name": "pandas"}, "c1"),
            AIMessage("Used csv module."),
        ],
        reviewer=[ReviewVerdict(passed=True)],
        calls=calls,
    )

    async def scenario():
        await g.ainvoke(START, CFG)
        snap = await g.aget_state(CFG)
        interrupts = [i for t in snap.tasks for i in t.interrupts]
        assert interrupts and interrupts[0].value["packages"] == ["pandas"]
        assert calls.install == []  # paused BEFORE execution
        await g.ainvoke(Command(resume={"decision": "DENIED", "operator": "bob", "reason": "nope"}), CFG)
        return (await g.aget_state(CFG)).values

    values = asyncio.run(scenario())
    assert calls.install == []
    denial = next(m for m in values["messages"] if isinstance(m, ToolMessage))
    assert "denied by bob" in denial.content and denial.status == "error"
    assert final_answer(values["messages"]) == "Used csv module."


def test_install_approved_executes_once():
    calls = Calls()
    g = graph_for(
        supervisor=[RouteDecision(next="coder"), RouteDecision(next="reviewer"), RouteDecision(next="finish")],
        coder=[tool_call("install_project_dependency", {"package_name": "pandas"}, "c1"), AIMessage("Installed.")],
        reviewer=[ReviewVerdict(passed=True)],
        calls=calls,
    )

    async def scenario():
        await g.ainvoke(START, CFG)
        await g.ainvoke(Command(resume={"decision": "APPROVED", "operator": "bob"}), CFG)

    asyncio.run(scenario())
    assert calls.install == ["pandas"]


def test_mixed_batch_is_gated_even_if_install_is_not_first():
    calls = Calls()
    batch = AIMessage(
        content="",
        tool_calls=[
            {"name": "save_and_run_python_code", "args": {"filename": "a", "code_content": "1"}, "id": "c1",
             "type": "tool_call"},
            {"name": "install_project_dependency", "args": {"package_name": "evil"}, "id": "c2", "type": "tool_call"},
        ],
    )  # fmt: skip
    g = graph_for([RouteDecision(next="coder")], [batch], [], calls)

    async def scenario():
        await g.ainvoke(START, CFG)
        return await g.aget_state(CFG)

    snap = asyncio.run(scenario())
    assert snap.next == ("approval_gate",)
    assert calls.run == [] and calls.install == []


def test_circuit_breaker_stops_coder_tool_loop():
    calls = Calls()
    loop = [tool_call("save_and_run_python_code", {"filename": "a", "code_content": "1"}, f"c{i}") for i in range(50)]
    g = graph_for([RouteDecision(next="coder")], loop, [], calls, max_hops=5)
    out = asyncio.run(g.ainvoke(START, CFG))
    assert "Hop limit of 5" in out["terminated_reason"]
    assert out["hops"] == 5 and len(calls.run) == 3  # supervisor + 4 coder turns, 3 executed batches


def test_reviewer_rejection_loops_back_to_coder():
    calls = Calls()
    g = graph_for(
        supervisor=[RouteDecision(next="coder"), RouteDecision(next="reviewer"),
                    RouteDecision(next="coder", instruction="fix"),
                    RouteDecision(next="reviewer"), RouteDecision(next="finish")],
        coder=[AIMessage("v1"), AIMessage("v2")],
        reviewer=[ReviewVerdict(passed=False, issues="off by one"), ReviewVerdict(passed=True)],
        calls=calls,
    )  # fmt: skip
    out = asyncio.run(g.ainvoke(START, CFG))
    texts = [m.content for m in out["messages"]]
    assert any("CHANGES REQUESTED\noff by one" in t for t in texts)
    assert final_answer(out["messages"]) == "v2" and out["review_passed"] is True


def test_new_coder_work_after_passing_review_is_reviewed_again():
    calls = Calls()
    reviewer_script = [ReviewVerdict(passed=True), ReviewVerdict(passed=True)]
    g = graph_for(
        supervisor=[
            RouteDecision(next="coder"),
            RouteDecision(next="reviewer"),
            RouteDecision(next="coder", instruction="tweak"),
            RouteDecision(next="finish"),  # overridden: v2 has not been reviewed
            RouteDecision(next="finish"),
        ],
        coder=[
            tool_call("save_and_run_python_code", {"filename": "a", "code_content": "1"}, "c1"),
            AIMessage("v1"),
            AIMessage("v2"),
        ],
        reviewer=reviewer_script,
        calls=calls,
    )
    out = asyncio.run(g.ainvoke(START, CFG))
    reviews = [m for m in out["messages"] if getattr(m, "name", None) == "reviewer"]
    assert len(reviews) == 2 and final_answer(out["messages"]) == "v2"
