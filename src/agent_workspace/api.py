"""FastAPI application: auth, run lifecycle, human approvals, health."""

from __future__ import annotations

import logging
import uuid
from collections.abc import Awaitable, Callable
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import Annotated, Any

from fastapi import Depends, FastAPI, HTTPException, Query, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from agent_workspace.config import Settings
from agent_workspace.metrics import APPROVAL_DECISIONS
from agent_workspace.models import (
    ApprovalDecisionRequest,
    ApprovalRead,
    ApprovalStatus,
    LoginRequest,
    OperatorInfo,
    Page,
    RunCreate,
    RunRead,
    TokenResponse,
)
from agent_workspace.runner import AgentService
from agent_workspace.security import (
    AuthError,
    LoginThrottle,
    Operator,
    OperatorDirectory,
    RevocationStore,
    RevocationUnavailable,
    Role,
    TokenClaims,
    TokenService,
)
from agent_workspace.store import ConflictError, Store

logger = logging.getLogger(__name__)


@dataclass
class Container:
    settings: Settings
    store: Store
    service: AgentService
    directory: OperatorDirectory
    tokens: TokenService
    revocations: RevocationStore
    throttle: LoginThrottle
    ping_redis: Callable[[], Awaitable[Any]]
    allow_self_approval: bool = False


def create_app(container_factory: Callable[[], Any], *, settings: Settings) -> FastAPI:
    """``container_factory`` is an async context manager factory yielding a Container.
    Production wiring lives in ``bootstrap.py``; tests inject fakes."""

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        async with container_factory() as container:
            app.state.container = container
            yield

    app = FastAPI(
        title="Agent Workspace API",
        version="0.2.0",
        lifespan=lifespan,
        docs_url=None if settings.environment == "production" else "/docs",
        redoc_url=None,
    )
    if settings.cors_allow_origins:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=settings.cors_allow_origins,
            allow_methods=["GET", "POST"],
            allow_headers=["Authorization", "Content-Type"],
        )

    bearer = HTTPBearer(auto_error=False)

    def get_container(request: Request) -> Container:
        return request.app.state.container

    C = Annotated[Container, Depends(get_container)]

    def unauthorized(detail: str = "Not authenticated") -> HTTPException:
        return HTTPException(status.HTTP_401_UNAUTHORIZED, detail, headers={"WWW-Authenticate": "Bearer"})

    async def current_claims(
        c: C, creds: Annotated[HTTPAuthorizationCredentials | None, Depends(bearer)]
    ) -> TokenClaims:
        if creds is None:
            raise unauthorized()
        try:
            claims = c.tokens.decode(creds.credentials)
            revoked = await c.revocations.is_revoked(claims.jti)
        except AuthError:
            raise unauthorized("Invalid or expired token") from None
        except RevocationUnavailable:
            raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, "Auth backend unavailable") from None
        # Revoked tokens and operators removed from the directory are rejected immediately.
        if revoked or c.directory.get(claims.subject) is None:
            raise unauthorized("Invalid or expired token")
        return claims

    def require(role: Role):
        async def dep(claims: Annotated[TokenClaims, Depends(current_claims)], c: C) -> Operator:
            operator = c.directory.get(claims.subject)
            if operator is None or not operator.has_role(role):  # role re-read: demotions apply at once
                raise HTTPException(status.HTTP_403_FORBIDDEN, "Insufficient role")
            return operator

        return dep

    AnyOperator = Annotated[Operator, Depends(require("operator"))]
    Approver = Annotated[Operator, Depends(require("approver"))]

    # ------------------------------------------------------------------ health
    @app.get("/healthz", tags=["health"])
    async def healthz() -> dict:
        return {"status": "ok"}

    @app.get("/readyz", tags=["health"])
    async def readyz(c: C) -> dict:
        try:
            await c.store.ping()
            await c.ping_redis()
        except Exception:
            logger.warning("Readiness check failed", exc_info=True)
            raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, "Dependencies unavailable") from None
        return {"status": "ready"}

    # -------------------------------------------------------------------- auth
    @app.post("/auth/token", response_model=TokenResponse, tags=["auth"])
    async def login(body: LoginRequest, c: C) -> TokenResponse:
        try:
            if await c.throttle.is_locked(body.username):
                raise HTTPException(status.HTTP_429_TOO_MANY_REQUESTS, "Too many failed attempts; try later")
            try:
                operator = c.directory.authenticate(body.username, body.password)
            except AuthError:
                await c.throttle.record_failure(body.username)
                raise unauthorized("Invalid credentials") from None
            await c.throttle.reset(body.username)
        except HTTPException:
            raise
        except Exception:
            logger.exception("Login backend failure")
            raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, "Auth backend unavailable") from None
        token, claims = c.tokens.issue(operator)
        return TokenResponse(access_token=token, expires_at=claims.expires_at, role=claims.role)

    @app.post("/auth/logout", status_code=status.HTTP_204_NO_CONTENT, tags=["auth"])
    async def logout(claims: Annotated[TokenClaims, Depends(current_claims)], c: C) -> None:
        try:
            await c.revocations.revoke(claims)
        except RevocationUnavailable:
            raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, "Auth backend unavailable") from None

    @app.get("/auth/me", response_model=OperatorInfo, tags=["auth"])
    async def me(operator: AnyOperator) -> OperatorInfo:
        return OperatorInfo(username=operator.username, role=operator.role)

    # -------------------------------------------------------------------- runs
    def visible(run, operator: Operator) -> bool:
        return run is not None and (run.owner == operator.username or operator.has_role("approver"))

    @app.post("/agent/runs", response_model=RunRead, status_code=status.HTTP_202_ACCEPTED, tags=["runs"])
    async def create_run(body: RunCreate, operator: AnyOperator, c: C) -> RunRead:
        if len(body.task) > c.settings.max_task_chars:
            raise HTTPException(status.HTTP_422_UNPROCESSABLE_ENTITY, "Task too long")
        run_id = uuid.uuid4().hex  # server-generated: clients cannot hijack another thread
        run = await c.store.create_run(run_id, operator.username, body.task)
        c.service.submit_start(run_id, body.task)
        return RunRead(**run.model_dump())

    @app.get("/agent/runs", response_model=Page[RunRead], tags=["runs"])
    async def list_runs(
        operator: AnyOperator,
        c: C,
        limit: Annotated[int, Query(ge=1, le=200)] = 50,
        offset: Annotated[int, Query(ge=0)] = 0,
    ) -> Page[RunRead]:
        owner = None if operator.has_role("approver") else operator.username
        rows = await c.store.list_runs(owner=owner, limit=limit, offset=offset)
        return Page(items=[RunRead(**r.model_dump()) for r in rows], limit=limit, offset=offset)

    @app.get("/agent/runs/{run_id}", response_model=RunRead, tags=["runs"])
    async def get_run(run_id: str, operator: AnyOperator, c: C) -> RunRead:
        run = await c.store.get_run(run_id)
        if not visible(run, operator):
            raise HTTPException(status.HTTP_404_NOT_FOUND, "Run not found")
        return RunRead(**run.model_dump())

    # --------------------------------------------------------------- approvals
    @app.get("/agent/approvals", response_model=Page[ApprovalRead], tags=["approvals"])
    async def list_approvals(
        _: Approver,
        c: C,
        status_filter: Annotated[ApprovalStatus | None, Query(alias="status")] = ApprovalStatus.PENDING,
        limit: Annotated[int, Query(ge=1, le=200)] = 50,
        offset: Annotated[int, Query(ge=0)] = 0,
    ) -> Page[ApprovalRead]:
        rows = await c.store.list_approvals(status=status_filter, limit=limit, offset=offset)
        return Page(items=[ApprovalRead.from_row(r) for r in rows], limit=limit, offset=offset)

    @app.get("/agent/audit-trail", response_model=Page[ApprovalRead], tags=["approvals"])
    async def audit_trail(
        _: Approver,
        c: C,
        limit: Annotated[int, Query(ge=1, le=500)] = 100,
        offset: Annotated[int, Query(ge=0)] = 0,
    ) -> Page[ApprovalRead]:
        rows = await c.store.list_approvals(status=None, limit=limit, offset=offset)
        return Page(items=[ApprovalRead.from_row(r) for r in rows], limit=limit, offset=offset)

    @app.post("/agent/approvals/{approval_id}/decision", response_model=ApprovalRead, tags=["approvals"])
    async def decide(approval_id: int, body: ApprovalDecisionRequest, operator: Approver, c: C) -> ApprovalRead:
        existing = await c.store.get_approval(approval_id)
        if existing is None:
            raise HTTPException(status.HTTP_404_NOT_FOUND, "Approval not found")
        if not c.allow_self_approval and existing.requested_by == operator.username:
            raise HTTPException(status.HTTP_403_FORBIDDEN, "Four-eyes rule: you cannot approve your own run")
        try:
            row = await c.store.decide_approval(
                approval_id, decision=body.decision, operator=operator.username, reason=body.reason
            )
        except ConflictError:
            raise HTTPException(status.HTTP_409_CONFLICT, "Approval already decided") from None
        APPROVAL_DECISIONS.labels(decision=str(row.status)).inc()
        c.service.submit_resume(row, operator.username)
        return ApprovalRead.from_row(row)

    return app
