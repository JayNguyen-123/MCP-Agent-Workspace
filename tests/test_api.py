"""HTTP-level tests: auth, revocation, RBAC, four-eyes rule and approval races.

Runs fully in-process against SQLite and in-memory fakes; no server, no Redis, no LLM.
"""

import asyncio
import json
from contextlib import asynccontextmanager
from datetime import timedelta

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("aiosqlite")

import httpx  # noqa: E402

from agent_workspace.api import Container, create_app  # noqa: E402
from agent_workspace.config import Settings  # noqa: E402
from agent_workspace.models import ApprovalStatus, RunStatus  # noqa: E402
from agent_workspace.security import (  # noqa: E402
    LoginThrottle,
    OperatorDirectory,
    RevocationStore,
    TokenService,
    hash_password,
)
from agent_workspace.store import Store, make_engine  # noqa: E402

from fakes import FakeKV  # noqa: E402

PW = "correct horse battery staple"
_HASH = hash_password(PW)
OPERATORS = {
    "olivia": {"password_hash": _HASH, "role": "operator"},
    "bob": {"password_hash": _HASH, "role": "approver"},
    "carol": {"password_hash": _HASH, "role": "approver"},
}


class FakeService:
    def __init__(self) -> None:
        self.started: list[str] = []
        self.resumed: list[tuple[int, ApprovalStatus, str]] = []

    def submit_start(self, run_id: str, task: str) -> None:
        self.started.append(run_id)

    def submit_resume(self, approval, operator: str) -> None:
        self.resumed.append((approval.id, approval.status, operator))


class Harness:
    def __init__(self, tmp_path) -> None:
        self.settings = Settings(environment="test", jwt_secret_key="k" * 48, operators_json=json.dumps(OPERATORS))
        self.kv = FakeKV()
        self.service = FakeService()
        self.store = Store(make_engine(f"sqlite+aiosqlite:///{tmp_path / 'test.db'}"))

        @asynccontextmanager
        async def factory():
            await self.store.create_schema()
            yield Container(
                settings=self.settings,
                store=self.store,
                service=self.service,  # type: ignore[arg-type]
                directory=OperatorDirectory(json.dumps(OPERATORS)),
                tokens=TokenService("k" * 48, "iss", "aud", timedelta(minutes=15)),
                revocations=RevocationStore(self.kv),
                throttle=LoginThrottle(self.kv, max_failures=3, lockout_seconds=60),
                ping_redis=self.kv.ping,
            )
            await self.store.engine.dispose()

        self.app = create_app(factory, settings=self.settings)

    @asynccontextmanager
    async def client(self):
        async with self.app.router.lifespan_context(self.app):
            transport = httpx.ASGITransport(app=self.app)
            async with httpx.AsyncClient(transport=transport, base_url="http://test") as c:
                yield c


async def login(c, user: str) -> dict[str, str]:
    r = await c.post("/auth/token", json={"username": user, "password": PW})
    assert r.status_code == 200, r.text
    return {"Authorization": f"Bearer {r.json()['access_token']}"}


@pytest.fixture
def h(tmp_path):
    return Harness(tmp_path)


def test_endpoints_require_auth(h):
    async def go():
        async with h.client() as c:
            for method, path in [("get", "/agent/runs"), ("post", "/agent/runs"), ("get", "/agent/approvals"),
                                 ("get", "/agent/audit-trail"), ("post", "/agent/approvals/1/decision")]:  # fmt: skip
                r = await c.request(method.upper(), path, json={})
                assert r.status_code == 401, (path, r.status_code)
            assert (await c.get("/healthz")).status_code == 200
            assert (await c.get("/readyz")).status_code == 200

    asyncio.run(go())


def test_login_throttle(h):
    async def go():
        async with h.client() as c:
            for _ in range(3):
                assert (await c.post("/auth/token", json={"username": "bob", "password": "x"})).status_code == 401
            r = await c.post("/auth/token", json={"username": "bob", "password": PW})
            assert r.status_code == 429

    asyncio.run(go())


def test_logout_revokes_token_immediately(h):
    async def go():
        async with h.client() as c:
            hdr = await login(c, "olivia")
            assert (await c.get("/auth/me", headers=hdr)).json() == {"username": "olivia", "role": "operator"}
            # A logout racing with parallel reads: every read is either 200 (before) or 401 (after), never 5xx.
            results = await asyncio.gather(
                c.post("/auth/logout", headers=hdr), *[c.get("/auth/me", headers=hdr) for _ in range(5)]
            )
            assert results[0].status_code == 204
            assert {r.status_code for r in results[1:]} <= {200, 401}
            assert (await c.get("/auth/me", headers=hdr)).status_code == 401

    asyncio.run(go())


def test_redis_outage_fails_closed(h):
    async def go():
        async with h.client() as c:
            hdr = await login(c, "olivia")
            h.kv.fail = True
            assert (await c.get("/auth/me", headers=hdr)).status_code == 503
            assert (await c.get("/readyz")).status_code == 503

    asyncio.run(go())


def test_run_lifecycle_and_visibility(h):
    async def go():
        async with h.client() as c:
            olivia, bob = await login(c, "olivia"), await login(c, "bob")
            r = await c.post("/agent/runs", json={"task": "print 8"}, headers=olivia)
            assert r.status_code == 202
            run_id = r.json()["id"]
            assert len(run_id) == 32 and h.service.started == [run_id]  # server-generated id
            assert (await c.get(f"/agent/runs/{run_id}", headers=olivia)).json()["status"] == "queued"
            assert (await c.get(f"/agent/runs/{run_id}", headers=bob)).status_code == 200  # approver sees all
            # another plain operator cannot see it
            await h.store.create_run("other", "someone-else", "x")
            assert (await c.get("/agent/runs/other", headers=olivia)).status_code == 404
            assert [r["id"] for r in (await c.get("/agent/runs", headers=olivia)).json()["items"]] == [run_id]
            # operators cannot reach approval endpoints
            assert (await c.get("/agent/approvals", headers=olivia)).status_code == 403

    asyncio.run(go())


async def _pending_approval(store: Store, owner: str = "olivia") -> int:
    await store.create_run("run1", owner, "task")
    await store.transition_run("run1", to=RunStatus.AWAITING_APPROVAL)
    row = await store.create_approval(run_id="run1", requested_by=owner, packages=["pandas"], tool_call_ids=["c1"])
    return row.id


def test_concurrent_decisions_exactly_one_wins(h):
    async def go():
        async with h.client() as c:
            approval_id = await _pending_approval(h.store)
            bob, carol = await login(c, "bob"), await login(c, "carol")
            ra, rb = await asyncio.gather(
                c.post(f"/agent/approvals/{approval_id}/decision", json={"decision": "APPROVED"}, headers=bob),
                c.post(f"/agent/approvals/{approval_id}/decision", json={"decision": "DENIED"}, headers=carol),
            )
            assert sorted([ra.status_code, rb.status_code]) == [200, 409]
            assert len(h.service.resumed) == 1  # the graph is resumed exactly once
            winner = ra if ra.status_code == 200 else rb
            trail = (await c.get("/agent/audit-trail", headers=bob)).json()["items"]
            assert len(trail) == 1 and trail[0]["decided_by"] == winner.json()["decided_by"]
            assert (await c.get("/agent/approvals", headers=bob)).json()["items"] == []  # nothing pending

    asyncio.run(go())


def test_decider_identity_comes_from_token_not_body(h):
    async def go():
        async with h.client() as c:
            approval_id = await _pending_approval(h.store)
            bob = await login(c, "bob")
            body = {"decision": "DENIED", "reason": "unvetted", "operator_name": "carol"}
            r = await c.post(f"/agent/approvals/{approval_id}/decision", json=body, headers=bob)
            assert r.status_code == 200 and r.json()["decided_by"] == "bob"
            assert h.service.resumed == [(approval_id, ApprovalStatus.DENIED, "bob")]

    asyncio.run(go())


def test_four_eyes_rule(h):
    async def go():
        async with h.client() as c:
            approval_id = await _pending_approval(h.store, owner="bob")
            bob = await login(c, "bob")
            r = await c.post(f"/agent/approvals/{approval_id}/decision", json={"decision": "APPROVED"}, headers=bob)
            assert r.status_code == 403

    asyncio.run(go())


def test_one_pending_approval_per_run(h):
    async def go():
        async with h.client():
            from agent_workspace.store import ConflictError

            await _pending_approval(h.store)
            with pytest.raises(ConflictError):
                await h.store.create_approval(run_id="run1", requested_by="x", packages=["a"], tool_call_ids=["c"])

    asyncio.run(go())
