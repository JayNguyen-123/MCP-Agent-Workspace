"""Runs the agent graph in the background and keeps the database in sync with it."""

from __future__ import annotations

import asyncio
import logging
from typing import Any, Protocol

from langchain_core.messages import HumanMessage
from langgraph.errors import GraphRecursionError
from langgraph.types import Command

from agent_workspace.graph import final_answer
from agent_workspace.logging_setup import log_context
from agent_workspace.metrics import RUN_HOPS, RUNS_TOTAL, RUNTIME_FAILURES
from agent_workspace.models import ApprovalRequest, RunStatus
from agent_workspace.store import ConflictError, Store

logger = logging.getLogger(__name__)


class CompiledGraph(Protocol):
    async def ainvoke(self, input: Any, config: dict) -> Any: ...
    async def aget_state(self, config: dict) -> Any: ...


class AnswerCache(Protocol):
    async def lookup(self, task: str) -> str | None: ...
    async def store(self, task: str, answer: str) -> None: ...


class AgentService:
    def __init__(
        self,
        graph: CompiledGraph,
        store: Store,
        *,
        max_hops: int,
        max_concurrent_runs: int,
        run_timeout_s: float,
        cache: AnswerCache | None = None,
    ) -> None:
        self.graph, self.store, self.cache = graph, store, cache
        self._max_hops, self._timeout = max_hops, run_timeout_s
        self._slots = asyncio.Semaphore(max_concurrent_runs)
        self._tasks: set[asyncio.Task] = set()

    def _config(self, run_id: str) -> dict:
        # recursion_limit is a hard backstop beneath our own hop counter
        # (each hop can involve several graph super-steps: tools, approval gate...).
        return {"configurable": {"thread_id": run_id}, "recursion_limit": self._max_hops * 4 + 10}

    # ------------------------------------------------------------- scheduling
    def submit_start(self, run_id: str, task: str) -> None:
        self._spawn(self._start(run_id, task), run_id)

    def submit_resume(self, approval: ApprovalRequest, operator: str) -> None:
        payload = {"decision": str(approval.status), "operator": operator, "reason": approval.reason}
        self._spawn(self._resume(approval.run_id, payload), approval.run_id)

    def _spawn(self, coro, run_id: str) -> None:
        task = asyncio.create_task(coro, name=f"agent-run-{run_id}")
        self._tasks.add(task)  # keep a strong reference until done
        task.add_done_callback(self._tasks.discard)

    async def shutdown(self, grace_s: float = 20.0) -> None:
        if not self._tasks:
            return
        _, pending = await asyncio.wait(self._tasks, timeout=grace_s)
        for t in pending:
            t.cancel()
        await asyncio.gather(*pending, return_exceptions=True)

    # -------------------------------------------------------------- execution
    async def _start(self, run_id: str, task: str) -> None:
        with log_context(run_id=run_id):
            if self.cache and (cached := await self.cache.lookup(task)):
                await self.store.transition_run(
                    run_id, to=RunStatus.COMPLETED, from_=[RunStatus.QUEUED], result=cached, cached=True
                )
                RUNS_TOTAL.labels(outcome="cached").inc()
                return
            if not await self.store.transition_run(run_id, to=RunStatus.RUNNING, from_=[RunStatus.QUEUED]):
                return
            inputs = {
                "messages": [HumanMessage(task)],
                "hops": 0,
                "review_passed": None,
                "terminated_reason": None,
            }
            await self._drive(run_id, inputs, task)

    async def _resume(self, run_id: str, payload: dict) -> None:
        with log_context(run_id=run_id):
            moved = await self.store.transition_run(run_id, to=RunStatus.RUNNING, from_=[RunStatus.AWAITING_APPROVAL])
            if not moved:
                logger.warning("Resume skipped: run is not awaiting approval")
                return
            run = await self.store.get_run(run_id)
            await self._drive(run_id, Command(resume=payload), run.task if run else "")

    async def _drive(self, run_id: str, graph_input: Any, task: str) -> None:
        config = self._config(run_id)
        try:

            async def invoke() -> None:
                async with self._slots:
                    await self.graph.ainvoke(graph_input, config)

            # The timeout covers queueing for a slot too, so it bounds a run's total wall time.
            await asyncio.wait_for(invoke(), timeout=self._timeout)
            await self._settle(run_id, config, task)
        except TimeoutError:
            RUNS_TOTAL.labels(outcome="timeout").inc()
            await self.store.transition_run(
                run_id, to=RunStatus.TERMINATED, error=f"Run exceeded the {self._timeout:.0f}s time limit."
            )
        except GraphRecursionError:
            RUNS_TOTAL.labels(outcome="terminated").inc()
            await self.store.transition_run(run_id, to=RunStatus.TERMINATED, error="Graph recursion limit reached.")
        except asyncio.CancelledError:
            await self.store.transition_run(run_id, to=RunStatus.FAILED, error="Cancelled during shutdown.")
            raise
        except Exception as exc:
            logger.exception("Run failed")
            RUNTIME_FAILURES.labels(component="runner", failure_type=type(exc).__name__).inc()
            RUNS_TOTAL.labels(outcome="failed").inc()
            # Store the type only; exception text can contain prompts or provider details.
            await self.store.transition_run(
                run_id, to=RunStatus.FAILED, error=f"Internal error ({type(exc).__name__})."
            )

    async def _settle(self, run_id: str, config: dict, task: str) -> None:
        snapshot = await self.graph.aget_state(config)
        interrupts = [i for t in (snapshot.tasks or ()) for i in (getattr(t, "interrupts", ()) or ())]
        values = snapshot.values or {}

        if interrupts:
            req = interrupts[0].value or {}
            run = await self.store.get_run(run_id)
            try:
                await self.store.create_approval(
                    run_id=run_id,
                    requested_by=run.owner if run else "unknown",
                    packages=list(req.get("packages", [])),
                    tool_call_ids=list(req.get("tool_call_ids", [])),
                )
            except ConflictError:
                logger.warning("Duplicate approval request suppressed")
            await self.store.transition_run(run_id, to=RunStatus.AWAITING_APPROVAL, from_=[RunStatus.RUNNING])
            RUNS_TOTAL.labels(outcome="awaiting_approval").inc()
            return

        RUN_HOPS.observe(values.get("hops", 0))
        answer = final_answer(values.get("messages", []))
        if reason := values.get("terminated_reason"):
            RUNS_TOTAL.labels(outcome="terminated").inc()
            await self.store.transition_run(run_id, to=RunStatus.TERMINATED, error=reason, result=answer or None)
            return
        await self.store.transition_run(run_id, to=RunStatus.COMPLETED, from_=[RunStatus.RUNNING], result=answer)
        RUNS_TOTAL.labels(outcome="completed").inc()
        if self.cache and answer and values.get("review_passed"):
            await self.cache.store(task, answer)
