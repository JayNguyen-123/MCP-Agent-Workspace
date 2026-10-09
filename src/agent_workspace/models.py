"""Database tables and API contracts."""

from __future__ import annotations

from datetime import UTC, datetime
from enum import StrEnum
from typing import Generic, TypeVar

from pydantic import BaseModel, Field
from sqlalchemy import Column, DateTime, Index, String, Text, text
from sqlmodel import Field as SQLField
from sqlmodel import SQLModel


def utcnow() -> datetime:
    return datetime.now(UTC)


def _ts(**kw) -> Column:
    return Column(DateTime(timezone=True), **kw)


# Statuses are stored as plain strings (portable, migration-friendly); StrEnum members
# compare equal to those strings, and rows read back hold ``str``.
class RunStatus(StrEnum):
    QUEUED = "queued"
    RUNNING = "running"
    AWAITING_APPROVAL = "awaiting_approval"
    COMPLETED = "completed"
    TERMINATED = "terminated"  # hop limit / timeout guardrail
    FAILED = "failed"


class ApprovalStatus(StrEnum):
    PENDING = "PENDING"
    APPROVED = "APPROVED"
    DENIED = "DENIED"


class Decision(StrEnum):
    APPROVED = "APPROVED"
    DENIED = "DENIED"


# --------------------------------------------------------------------------- #
# Tables
# --------------------------------------------------------------------------- #
class AgentRun(SQLModel, table=True):
    __tablename__ = "agent_runs"

    id: str = SQLField(primary_key=True, max_length=64)  # also the LangGraph thread_id
    owner: str = SQLField(index=True, max_length=255)
    task: str = SQLField(sa_column=Column(Text, nullable=False))
    status: RunStatus = SQLField(default=RunStatus.QUEUED, sa_column=Column(String(32), nullable=False, index=True))
    result: str | None = SQLField(default=None, sa_column=Column(Text))
    error: str | None = SQLField(default=None, sa_column=Column(Text))
    cached: bool = SQLField(default=False)
    created_at: datetime = SQLField(default_factory=utcnow, sa_column=_ts(nullable=False))
    updated_at: datetime = SQLField(default_factory=utcnow, sa_column=_ts(nullable=False))


class ApprovalRequest(SQLModel, table=True):
    """Both the live approval queue and the permanent, append-only audit trail."""

    __tablename__ = "approval_requests"
    __table_args__ = (
        # At most one PENDING approval per run, enforced by the database.
        Index(
            "uq_one_pending_approval_per_run",
            "run_id",
            unique=True,
            postgresql_where=text("status = 'PENDING'"),
            sqlite_where=text("status = 'PENDING'"),
        ),
    )

    id: int | None = SQLField(default=None, primary_key=True)
    run_id: str = SQLField(foreign_key="agent_runs.id", index=True, max_length=64)
    requested_by: str = SQLField(max_length=255)  # owner of the run
    packages: str = SQLField(sa_column=Column(Text, nullable=False))  # comma-separated specs
    tool_call_ids: str = SQLField(sa_column=Column(Text, nullable=False))
    status: ApprovalStatus = SQLField(
        default=ApprovalStatus.PENDING, sa_column=Column(String(16), nullable=False, index=True)
    )
    decided_by: str | None = SQLField(default=None, max_length=255)  # from the JWT, never the body
    reason: str | None = SQLField(default=None, sa_column=Column(Text))
    created_at: datetime = SQLField(default_factory=utcnow, sa_column=_ts(nullable=False))
    decided_at: datetime | None = SQLField(default=None, sa_column=_ts(nullable=True))


# --------------------------------------------------------------------------- #
# API contracts
# --------------------------------------------------------------------------- #
class LoginRequest(BaseModel):
    username: str = Field(min_length=1, max_length=255)
    password: str = Field(min_length=1, max_length=1024)


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"  # noqa: S105
    expires_at: datetime
    role: str


class OperatorInfo(BaseModel):
    username: str
    role: str


class RunCreate(BaseModel):
    task: str = Field(min_length=1, max_length=8000)


class RunRead(BaseModel):
    id: str
    owner: str
    task: str
    status: RunStatus
    result: str | None
    error: str | None
    cached: bool
    created_at: datetime
    updated_at: datetime


class ApprovalDecisionRequest(BaseModel):
    decision: Decision
    reason: str | None = Field(default=None, max_length=2000)


class ApprovalRead(BaseModel):
    id: int
    run_id: str
    requested_by: str
    packages: list[str]
    status: ApprovalStatus
    decided_by: str | None
    reason: str | None
    created_at: datetime
    decided_at: datetime | None

    @classmethod
    def from_row(cls, row: ApprovalRequest) -> ApprovalRead:
        data = row.model_dump()
        data["packages"] = [p for p in row.packages.split(",") if p]
        return cls(**data)


T = TypeVar("T")


class Page(BaseModel, Generic[T]):  # noqa: UP046 - keep 3.11 compatible
    items: list[T]
    limit: int
    offset: int
