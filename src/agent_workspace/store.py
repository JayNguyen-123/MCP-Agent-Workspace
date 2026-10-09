"""Async persistence layer. All state transitions are conditional UPDATEs so they are
atomic across concurrent requests *and* across API replicas."""

from __future__ import annotations

from collections.abc import AsyncIterator, Sequence
from contextlib import asynccontextmanager
from datetime import timedelta

from sqlalchemy import text, update
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncEngine, async_sessionmaker, create_async_engine
from sqlmodel import SQLModel, col, select
from sqlmodel.ext.asyncio.session import AsyncSession

from agent_workspace.models import AgentRun, ApprovalRequest, ApprovalStatus, Decision, RunStatus, utcnow


class ConflictError(Exception):
    """The requested state transition lost a race or is no longer valid."""


def make_engine(url: str, *, pool_size: int = 10) -> AsyncEngine:
    kwargs: dict = {"pool_pre_ping": True}
    if not url.startswith("sqlite"):
        kwargs.update(pool_size=pool_size, max_overflow=pool_size)
    return create_async_engine(url, **kwargs)


class Store:
    def __init__(self, engine: AsyncEngine) -> None:
        self.engine = engine
        self._sessions = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)

    async def create_schema(self) -> None:
        async with self.engine.begin() as conn:
            await conn.run_sync(SQLModel.metadata.create_all)

    @asynccontextmanager
    async def session(self) -> AsyncIterator[AsyncSession]:
        async with self._sessions() as s:
            yield s

    async def ping(self) -> None:
        async with self.session() as s:
            await s.execute(text("SELECT 1"))

    # ------------------------------------------------------------------ runs
    async def create_run(self, run_id: str, owner: str, task: str) -> AgentRun:
        run = AgentRun(id=run_id, owner=owner, task=task)
        async with self.session() as s:
            s.add(run)
            await s.commit()
        return run

    async def get_run(self, run_id: str) -> AgentRun | None:
        async with self.session() as s:
            return await s.get(AgentRun, run_id)

    async def list_runs(self, *, owner: str | None, limit: int, offset: int) -> Sequence[AgentRun]:
        stmt = select(AgentRun).order_by(col(AgentRun.created_at).desc()).limit(limit).offset(offset)
        if owner is not None:
            stmt = stmt.where(AgentRun.owner == owner)
        async with self.session() as s:
            return (await s.exec(stmt)).all()

    async def transition_run(
        self, run_id: str, *, to: RunStatus, from_: Sequence[RunStatus] | None = None, **fields
    ) -> bool:
        """Move a run to ``to`` only if it is currently in one of ``from_``. Returns success."""
        stmt = update(AgentRun).where(col(AgentRun.id) == run_id)
        if from_:
            stmt = stmt.where(col(AgentRun.status).in_(list(from_)))
        stmt = stmt.values(status=to, updated_at=utcnow(), **fields)
        async with self.session() as s:
            res = await s.execute(stmt)
            await s.commit()
            return res.rowcount == 1

    async def fail_orphaned_runs(self, *, older_than: timedelta) -> int:
        """Fail runs left RUNNING/QUEUED by a crashed replica.

        Only runs idle for longer than the run timeout are touched: a live run would have
        timed out by then, so this is safe with several replicas starting independently.
        """
        cutoff = utcnow() - older_than
        stmt = (
            update(AgentRun)
            .where(col(AgentRun.status).in_([RunStatus.RUNNING, RunStatus.QUEUED]))
            .where(col(AgentRun.updated_at) < cutoff)
            .values(status=RunStatus.FAILED, error="Interrupted by a service restart.", updated_at=utcnow())
        )
        async with self.session() as s:
            res = await s.execute(stmt)
            await s.commit()
            return res.rowcount or 0

    # ------------------------------------------------------------- approvals
    async def create_approval(
        self, *, run_id: str, requested_by: str, packages: list[str], tool_call_ids: list[str]
    ) -> ApprovalRequest:
        row = ApprovalRequest(
            run_id=run_id,
            requested_by=requested_by,
            packages=",".join(packages),
            tool_call_ids=",".join(tool_call_ids),
        )
        async with self.session() as s:
            s.add(row)
            try:
                await s.commit()
            except IntegrityError as exc:
                await s.rollback()
                raise ConflictError("A pending approval already exists for this run.") from exc
        return row

    async def get_approval(self, approval_id: int) -> ApprovalRequest | None:
        async with self.session() as s:
            return await s.get(ApprovalRequest, approval_id)

    async def list_approvals(
        self, *, status: ApprovalStatus | None, limit: int, offset: int
    ) -> Sequence[ApprovalRequest]:
        stmt = select(ApprovalRequest).order_by(col(ApprovalRequest.created_at).desc()).limit(limit).offset(offset)
        if status is not None:
            stmt = stmt.where(ApprovalRequest.status == status)
        async with self.session() as s:
            return (await s.exec(stmt)).all()

    async def decide_approval(
        self, approval_id: int, *, decision: Decision, operator: str, reason: str | None
    ) -> ApprovalRequest:
        """Atomically claim a PENDING approval. Exactly one concurrent caller wins;
        all others get ConflictError (the original relied on a process-local dict)."""
        stmt = (
            update(ApprovalRequest)
            .where(col(ApprovalRequest.id) == approval_id, col(ApprovalRequest.status) == ApprovalStatus.PENDING)
            .values(status=ApprovalStatus(decision.value), decided_by=operator, reason=reason, decided_at=utcnow())
        )
        async with self.session() as s:
            res = await s.execute(stmt)
            await s.commit()
            if res.rowcount != 1:
                raise ConflictError("Approval not found or already decided.")
            row = await s.get(ApprovalRequest, approval_id, populate_existing=True)
            assert row is not None
            return row
