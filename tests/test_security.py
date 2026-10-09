import asyncio
import os
import time
from datetime import UTC, datetime, timedelta

import jwt
import pytest

from agent_workspace.security import (
    AuthError,
    LoginThrottle,
    Operator,
    OperatorDirectory,
    RevocationStore,
    RevocationUnavailable,
    TokenService,
    hash_password,
    verify_password,
)

from fakes import FakeKV

SECRET = "x" * 48
ALICE = Operator("alice", "approver")


def make_tokens(ttl_minutes: int = 15) -> TokenService:
    return TokenService(SECRET, "iss", "aud", timedelta(minutes=ttl_minutes))


def test_password_hash_roundtrip_and_salt():
    h1, h2 = hash_password("hunter2"), hash_password("hunter2")
    assert h1 != h2  # unique salt
    assert verify_password("hunter2", h1)
    assert not verify_password("hunter3", h1)
    assert not verify_password("hunter2", "garbage")


def test_directory_authenticate():
    directory = OperatorDirectory(f'{{"alice": {{"password_hash": "{hash_password("pw")}", "role": "approver"}}}}')
    assert directory.authenticate("alice", "pw") == ALICE
    with pytest.raises(AuthError):
        directory.authenticate("alice", "wrong")
    with pytest.raises(AuthError):
        directory.authenticate("mallory", "pw")
    with pytest.raises(ValueError):
        OperatorDirectory('{"bob": {"password_hash": "x", "role": "root"}}')


def test_token_roundtrip_and_claims():
    tokens = make_tokens()
    token, claims = tokens.issue(ALICE)
    decoded = tokens.decode(token)
    assert decoded.subject == "alice" and decoded.role == "approver" and decoded.jti == claims.jti


def test_token_expiry_is_timezone_independent():
    """Regression: datetime.utcnow().timestamp() treated UTC as local time, so in
    America/Chicago tokens lived ~5h longer than configured."""
    if not hasattr(time, "tzset"):
        pytest.skip("tzset unavailable")
    old = os.environ.get("TZ")
    try:
        os.environ["TZ"] = "America/Chicago"
        time.tzset()
        _, claims = make_tokens(ttl_minutes=15).issue(ALICE)
        remaining = (claims.expires_at - datetime.now(UTC)).total_seconds()
        assert 14 * 60 < remaining <= 15 * 60
    finally:
        if old is None:
            os.environ.pop("TZ", None)
        else:
            os.environ["TZ"] = old
        time.tzset()


@pytest.mark.parametrize(
    "mutate",
    [
        lambda p: {**p, "aud": "other"},
        lambda p: {**p, "iss": "other"},
        lambda p: {**p, "exp": int(time.time()) - 3600},
        lambda p: {**p, "role": "root"},
        lambda p: {k: v for k, v in p.items() if k != "jti"},
    ],
)
def test_token_rejections(mutate):
    tokens = make_tokens()
    token, _ = tokens.issue(ALICE)
    payload = jwt.decode(token, SECRET, algorithms=["HS256"], audience="aud")
    forged = jwt.encode(mutate(payload), SECRET, algorithm="HS256")
    with pytest.raises(AuthError):
        tokens.decode(forged)


def test_alg_none_and_wrong_key_rejected():
    tokens = make_tokens()
    token, _ = tokens.issue(ALICE)
    payload = jwt.decode(token, SECRET, algorithms=["HS256"], audience="aud")
    with pytest.raises(AuthError):
        tokens.decode(jwt.encode(payload, None, algorithm="none"))
    with pytest.raises(AuthError):
        tokens.decode(jwt.encode(payload, "y" * 48, algorithm="HS256"))


def test_revocation_by_jti_with_ttl():
    async def scenario():
        kv = FakeKV()
        store = RevocationStore(kv)
        _, claims = make_tokens().issue(ALICE)
        assert not await store.is_revoked(claims.jti)
        await store.revoke(claims)
        assert await store.is_revoked(claims.jti)
        ((_, expires),) = kv.data.values()
        assert 15 * 60 < expires - time.monotonic() <= 15 * 60 + 16  # TTL tracks token lifetime

    asyncio.run(scenario())


def test_revocation_fails_closed_when_redis_is_down():
    async def scenario():
        kv = FakeKV()
        kv.fail = True
        with pytest.raises(RevocationUnavailable):
            await RevocationStore(kv).is_revoked("any")

    asyncio.run(scenario())


def test_login_throttle_locks_and_resets():
    async def scenario():
        throttle = LoginThrottle(FakeKV(), max_failures=3, lockout_seconds=60)
        for _ in range(3):
            assert not await throttle.is_locked("bob")
            await throttle.record_failure("bob")
        assert await throttle.is_locked("bob")
        await throttle.reset("bob")
        assert not await throttle.is_locked("bob")

    asyncio.run(scenario())
