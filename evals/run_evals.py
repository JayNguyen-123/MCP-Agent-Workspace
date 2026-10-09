"""LangSmith regression evals against the real graph and a running sandbox.

    uv run --extra api --group evals python evals/run_evals.py

Fixes over the original tests/test_suite.py:
* Each example gets its own thread id (the original reused one, so examples
  contaminated each other through the checkpointer).
* Token usage is summed over *every* model call via a callback (the original read
  only the final message, missing supervisor/reviewer spend).
* Install approvals are auto-denied so a paused graph cannot silently "pass".
* The exit code reflects the scores, so CI actually fails on regressions.
* Lives outside tests/ so `pytest` never makes paid API calls by accident.
"""

from __future__ import annotations

import asyncio
import os
import sys
import uuid

from langchain_core.callbacks import UsageMetadataCallbackHandler
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command
from langsmith import Client
from langsmith.evaluation import evaluate

from agent_workspace.bootstrap import _chat
from agent_workspace.config import Settings
from agent_workspace.graph import AgentModels, build_graph, final_answer
from agent_workspace.sandbox_tools import SandboxClient, load_sandbox_tools

DATASET = "Coding Sandbox Regression Dataset"
TOKEN_BUDGET = int(os.getenv("EVAL_TOKEN_BUDGET", "20000"))
MIN_ACCURACY = float(os.getenv("EVAL_MIN_ACCURACY", "1.0"))

EXAMPLES = [
    ("Write matrix.py that prints a 2x2 matrix of zeroes as a nested Python list. Run it.", "[[0, 0], [0, 0]]"),
    ("Create simple_calc.py that computes 4+4 and prints the result. Lint it, then run it.", "8"),
    ("Use pandas to print the sum of 1..10.", "55"),  # exercises the denied-install path
]


def ensure_dataset(client: Client) -> None:
    if client.has_dataset(dataset_name=DATASET):
        return
    ds = client.create_dataset(dataset_name=DATASET, description="Agent coding regression set.")
    client.create_examples(
        inputs=[{"query": q} for q, _ in EXAMPLES],
        outputs=[{"expected": e} for _, e in EXAMPLES],
        dataset_id=ds.id,
    )


async def _build_graph(settings: Settings):
    sandbox = SandboxClient(
        settings.sandbox_mcp_url,
        settings.sandbox_auth_token.get_secret_value(),
        timeout_s=settings.sandbox_call_timeout_s,
    )
    tools = await load_sandbox_tools(sandbox)
    models = AgentModels(
        _chat(settings, settings.supervisor_model, 0.0),
        _chat(settings, settings.coder_model, 0.0),
        _chat(settings, settings.reviewer_model, 0.0),
    )
    return build_graph(models, tools, max_hops=settings.max_hops, checkpointer=InMemorySaver())


def make_target(graph):
    async def run(inputs: dict) -> dict:
        usage = UsageMetadataCallbackHandler()
        config = {
            "configurable": {"thread_id": f"eval-{uuid.uuid4().hex}"},
            "callbacks": [usage],
            "recursion_limit": 120,
        }
        await graph.ainvoke({"messages": [("user", inputs["query"])], "hops": 0}, config)
        for _ in range(3):  # auto-deny any install requests
            snap = await graph.aget_state(config)
            if not any(t.interrupts for t in snap.tasks):
                break
            await graph.ainvoke(Command(resume={"decision": "DENIED", "operator": "eval", "reason": "eval"}), config)
        values = (await graph.aget_state(config)).values
        total = sum(u.get("total_tokens", 0) for u in usage.usage_metadata.values())
        return {
            "output": final_answer(values.get("messages", [])),
            "total_tokens": total,
            "hops": values.get("hops", 0),
            "terminated": bool(values.get("terminated_reason")),
        }

    return lambda inputs: asyncio.run(run(inputs))


def accuracy(outputs: dict, reference_outputs: dict) -> dict:
    return {"key": "answer_contains_expected", "score": int(reference_outputs["expected"] in outputs["output"])}


def token_budget(outputs: dict) -> dict:
    return {"key": "within_token_budget", "score": int(outputs["total_tokens"] <= TOKEN_BUDGET)}


def not_terminated(outputs: dict) -> dict:
    return {"key": "not_circuit_broken", "score": int(not outputs["terminated"])}


def main() -> int:
    if not os.getenv("LANGSMITH_API_KEY"):
        print("LANGSMITH_API_KEY is not set; skipping evals.", file=sys.stderr)
        return 2
    settings = Settings()
    client = Client()
    ensure_dataset(client)
    graph = asyncio.run(_build_graph(settings))
    results = evaluate(
        make_target(graph),
        data=DATASET,
        evaluators=[accuracy, token_budget, not_terminated],
        experiment_prefix="agent-regression",
        max_concurrency=1,
    )
    scores: dict[str, list[float]] = {}
    for row in results:
        for res in row["evaluation_results"]["results"]:
            scores.setdefault(res.key, []).append(res.score or 0)
    means = {k: sum(v) / len(v) for k, v in scores.items()}
    print("Eval means:", means)
    accurate = means.get("answer_contains_expected", 0) >= MIN_ACCURACY
    guardrails_ok = all(v == 1 for k, v in means.items() if k != "answer_contains_expected")
    return 0 if accurate and guardrails_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
