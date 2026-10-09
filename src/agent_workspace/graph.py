"""Supervisor → Coder ⇄ Tools → Reviewer graph with a human approval gate and hop limit.

START → supervisor ─┬─► coder ─┬─► tools ─────────► coder
                    │          ├─► approval_gate ─┬► tools      (APPROVED)
                    │          │   (interrupt)    └► coder      (DENIED → ToolMessages)
                    │          └─► supervisor
                    ├─► reviewer ─► supervisor
                    └─► END
Every agent node consumes one hop; any route that would exceed ``max_hops``
goes to ``circuit_breaker`` → END instead, including the coder ⇄ tools loop.
"""

import functools
import logging
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass
from typing import Annotated, Any, Literal, TypedDict

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, AnyMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import BaseTool
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.errors import GraphBubbleUp
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from langgraph.types import Command, interrupt
from pydantic import BaseModel, Field

from agent_workspace.logging_setup import log_context
from agent_workspace.metrics import CIRCUIT_BREAKER_TRIPS, NODE_LATENCY, RUNTIME_FAILURES, record_usage

logger = logging.getLogger(__name__)

SENSITIVE_TOOLS = frozenset({"install_project_dependency"})


class TeamState(TypedDict, total=False):
    messages: Annotated[list[AnyMessage], add_messages]
    hops: int
    review_passed: bool | None
    terminated_reason: str | None
    route: str  # supervisor's last routing decision


class RouteDecision(BaseModel):
    """Supervisor routing decision."""

    next: Literal["coder", "reviewer", "finish"]
    instruction: str = Field(default="", description="Concrete instruction for the coder when next == 'coder'.")


class ReviewVerdict(BaseModel):
    """Reviewer verdict on the most recent work."""

    passed: bool
    issues: str = Field(default="", description="Specific, actionable problems to fix. Empty when passed.")


@dataclass(frozen=True)
class AgentModels:
    supervisor: BaseChatModel
    coder: BaseChatModel
    reviewer: BaseChatModel


SUPERVISOR_PROMPT = """You supervise a two-agent coding team working in an isolated Python sandbox.
Pick the next step:
- coder: code must be written, changed, linted or run, or the reviewer requested changes.
- reviewer: the coder has produced or executed code that has not yet been reviewed.
- finish: the user's objective is met AND the latest review passed (or no code was needed).
You have {remaining} step(s) left before the run is force-stopped; finish if further work will not help."""

CODER_PROMPT = """You are a senior Python engineer working in an isolated sandbox.
Use the tools to write, lint and run code; always run code to verify it before you finish.
Only request new packages when the standard library cannot reasonably do the job; package
installs require human approval and may be denied. When done, reply with a short summary
of what you built and the verified output."""

REVIEWER_PROMPT = """You are a strict code reviewer. Inspect the conversation, especially the most
recent code and its execution output. Pass only if the code ran successfully, satisfies the
user's request, and has no obvious bugs. Otherwise list specific fixes."""


def message_text(message: AnyMessage) -> str:
    content = message.content
    if isinstance(content, str):
        return content
    return "".join(b.get("text", "") for b in content if isinstance(b, dict) and b.get("type") == "text")


def final_answer(messages: Sequence[AnyMessage]) -> str:
    for msg in reversed(messages):
        if isinstance(msg, AIMessage) and not msg.tool_calls and message_text(msg).strip():
            return message_text(msg)
    return ""


def _instrument(name: str):
    def deco(fn: Callable[..., Awaitable[Any]]):
        @functools.wraps(fn)
        async def wrapper(state: TeamState, config: RunnableConfig):
            with log_context(node=name), NODE_LATENCY.labels(node=name).time():
                try:
                    return await fn(state, config)
                except GraphBubbleUp:  # interrupts / control flow, not failures
                    raise
                except Exception as exc:
                    RUNTIME_FAILURES.labels(component=name, failure_type=type(exc).__name__).inc()
                    logger.exception("Node %s failed", name)
                    raise

        return wrapper

    return deco


def build_graph(
    models: AgentModels,
    tools: Sequence[BaseTool],
    *,
    max_hops: int,
    checkpointer: BaseCheckpointSaver | None,
):
    supervisor_llm = models.supervisor.with_structured_output(RouteDecision, include_raw=True)
    reviewer_llm = models.reviewer.with_structured_output(ReviewVerdict, include_raw=True)
    coder_llm = models.coder.bind_tools(list(tools))

    def over_budget(state: TeamState) -> bool:
        return state.get("hops", 0) >= max_hops

    # ------------------------------------------------------------------ nodes
    @_instrument("supervisor")
    async def supervisor(state: TeamState, config: RunnableConfig) -> dict:
        prompt = SystemMessage(SUPERVISOR_PROMPT.format(remaining=max_hops - state.get("hops", 0)))
        # Always end on a user turn so the provider never treats our history as a prefill.
        out = await supervisor_llm.ainvoke([prompt, *state["messages"], HumanMessage("Decide the next step.")], config)
        record_usage("supervisor", out.get("raw"))
        decision: RouteDecision | None = out.get("parsed")
        if decision is None:
            logger.warning("Unparseable supervisor output; defaulting to reviewer")
            decision = RouteDecision(next="reviewer")
        # Deterministic guard: never finish on unreviewed tool work.
        ran_tools = any(isinstance(m, ToolMessage) for m in state["messages"])
        if decision.next == "finish" and ran_tools and state.get("review_passed") is not True:
            decision = RouteDecision(next="reviewer")
        update: dict = {"hops": state.get("hops", 0) + 1, "route": decision.next}
        if decision.next == "coder":
            text = decision.instruction or "Continue working on the task."
            update["messages"] = [HumanMessage(f"Supervisor: {text}", name="supervisor")]
        return update

    @_instrument("coder")
    async def coder(state: TeamState, config: RunnableConfig) -> dict:
        response = await coder_llm.ainvoke([SystemMessage(CODER_PROMPT), *state["messages"]], config)
        record_usage("coder", response)
        # Any new coder output invalidates an earlier passing review.
        return {"messages": [response], "hops": state.get("hops", 0) + 1, "review_passed": None}

    @_instrument("reviewer")
    async def reviewer(state: TeamState, config: RunnableConfig) -> dict:
        out = await reviewer_llm.ainvoke(
            [SystemMessage(REVIEWER_PROMPT), *state["messages"], HumanMessage("Review the work above.")], config
        )
        record_usage("reviewer", out.get("raw"))
        verdict: ReviewVerdict = out.get("parsed") or ReviewVerdict(passed=False, issues="Reviewer output unreadable.")
        text = "Code review: PASSED" if verdict.passed else f"Code review: CHANGES REQUESTED\n{verdict.issues}"
        return {
            "messages": [HumanMessage(text, name="reviewer")],
            "review_passed": verdict.passed,
            "hops": state.get("hops", 0) + 1,
        }

    @_instrument("approval_gate")
    async def approval_gate(state: TeamState, config: RunnableConfig) -> Command[Literal["tools", "coder"]]:
        last = state["messages"][-1]
        calls = list(getattr(last, "tool_calls", []) or [])
        packages = [str(c["args"].get("package_name", "")) for c in calls if c["name"] in SENSITIVE_TOOLS]
        # Everything above interrupt() re-runs on resume, so it must stay side-effect free.
        decision: dict = interrupt(
            {"kind": "package_install", "packages": packages, "tool_call_ids": [c["id"] for c in calls]}
        )
        if decision.get("decision") == "APPROVED":
            return Command(goto="tools")
        who = decision.get("operator", "an operator")
        why = decision.get("reason") or "no reason given"
        denials = [
            ToolMessage(
                content=(
                    f"Not executed: package installation was denied by {who} ({why}). "
                    "Do not retry the same install; find a standard-library alternative or explain the blocker."
                ),
                tool_call_id=c["id"],
                name=c["name"],
                status="error",
            )
            for c in calls
        ]
        return Command(goto="coder", update={"messages": denials})

    async def circuit_breaker(state: TeamState) -> dict:
        CIRCUIT_BREAKER_TRIPS.inc()
        logger.error("Circuit breaker tripped after %d hops", state.get("hops", 0))
        return {"terminated_reason": f"Hop limit of {max_hops} reached before the task completed."}

    # ---------------------------------------------------------------- routing
    def after_supervisor(state: TeamState) -> str:
        if over_budget(state):
            return "circuit_breaker"
        return {"coder": "coder", "reviewer": "reviewer", "finish": END}[state.get("route", "finish")]

    def after_coder(state: TeamState) -> str:
        last = state["messages"][-1]
        calls = getattr(last, "tool_calls", None) or []
        if not calls:
            return "supervisor"
        if over_budget(state):
            return "circuit_breaker"
        # Gate the whole batch if ANY call is sensitive (the original only checked the first call).
        return "approval_gate" if any(c["name"] in SENSITIVE_TOOLS for c in calls) else "tools"

    def after_reviewer(state: TeamState) -> str:
        return "circuit_breaker" if over_budget(state) else "supervisor"

    builder = StateGraph(TeamState)
    builder.add_node("supervisor", supervisor)
    builder.add_node("coder", coder)
    builder.add_node("reviewer", reviewer)
    builder.add_node("tools", ToolNode(list(tools), handle_tool_errors=True))
    builder.add_node("approval_gate", approval_gate, destinations=("tools", "coder"))
    builder.add_node("circuit_breaker", circuit_breaker)

    builder.add_edge(START, "supervisor")
    builder.add_conditional_edges("supervisor", after_supervisor, ["coder", "reviewer", "circuit_breaker", END])
    builder.add_conditional_edges("coder", after_coder, ["tools", "approval_gate", "supervisor", "circuit_breaker"])
    builder.add_conditional_edges("reviewer", after_reviewer, ["supervisor", "circuit_breaker"])
    builder.add_edge("tools", "coder")
    builder.add_edge("circuit_breaker", END)

    return builder.compile(checkpointer=checkpointer)
