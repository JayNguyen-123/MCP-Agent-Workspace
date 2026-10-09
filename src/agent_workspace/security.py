"""Operator authentication: password hashing, JWTs and token revocation.

Pure logic with no web-framework imports so it can be unit-tested in isolation;
FastAPI dependencies live in ``agent_workspace.api.deps``.
"""

from __future__ import annotations

import base64
import getpass
import hashlib
import hmac
import json
import secrets
import sys
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Literal, Protocol

import jwt

Role = Literal["operator", "approver"]
_ROLE_RANK = {"operator": 1, "approver": 2}

# scrypt parameters (~64 MiB, ~50 ms). Stored in the hash so they can be raised later.
_SCRYPT_N, _SCRYPT_R, _SCRYPT_P = 2**16, 8, 1


class AuthError(Exception):
    """Credentials or token are invalid. Never leaks *why* to the client."""


class RevocationUnavailable(Exception):
    """The revocation store cannot be reached. Callers must fail closed."""


# --------------------------------------------------------------------------- #
# Passwords
# --------------------------------------------------------------------------- #
def _b64(raw: bytes) -> str:
    return base64.urlsafe_b64encode(raw).decode().rstrip("=")


def _unb64(text: str) -> bytes:
    return base64.urlsafe_b64decode(text + "=" * (-len(text) % 4))


def hash_password(password: str) -> str:
    salt = secrets.token_bytes(16)
    digest = hashlib.scrypt(
        password.encode(), salt=salt, n=_SCRYPT_N, r=_SCRYPT_R, p=_SCRYPT_P, maxmem=128 * 1024 * 1024
    )
    return f"scrypt${_SCRYPT_N}${_SCRYPT_R}${_SCRYPT_P}${_b64(salt)}${_b64(digest)}"


def verify_password(password: str, encoded: str) -> bool:
    try:
        scheme, n, r, p, salt, expected = encoded.split("$")
        if scheme != "scrypt":
            return False
        digest = hashlib.scrypt(
            password.encode(), salt=_unb64(salt), n=int(n), r=int(r), p=int(p), maxmem=128 * 1024 * 1024
        )
    except (ValueError, TypeError):
        return False
    return hmac.compare_digest(digest, _unb64(expected))


# Used to equalise timing when the username does not exist.
_DUMMY_HASH = hash_password(secrets.token_urlsafe(16))


# --------------------------------------------------------------------------- #
# Operator directory
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Operator:
    username: str
    role: Role

    def has_role(self, required: Role) -> bool:
        return _ROLE_RANK[self.role] >= _ROLE_RANK[required]


class OperatorDirectory:
    """Static operator accounts from configuration.

    Swap for an OIDC / SSO integration in larger deployments; the rest of the
    service only depends on ``authenticate`` returning an ``Operator``.
    """

    def __init__(self, raw_json: str) -> None:
        data = json.loads(raw_json or "{}")
        self._users: dict[str, tuple[str, Role]] = {}
        for name, entry in data.items():
            role = entry.get("role", "operator")
            if role not in _ROLE_RANK:
                raise ValueError(f"Unknown role {role!r} for operator {name!r}")
            self._users[name] = (entry["password_hash"], role)

    def authenticate(self, username: str, password: str) -> Operator:
        stored = self._users.get(username)
        ok = verify_password(password, stored[0] if stored else _DUMMY_HASH)
        if not (stored and ok):
            raise AuthError("Invalid credentials")
        return Operator(username=username, role=stored[1])

    def get(self, username: str) -> Operator | None:
        stored = self._users.get(username)
        return Operator(username, stored[1]) if stored else None


# --------------------------------------------------------------------------- #
# Tokens
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class TokenClaims:
    subject: str
    role: Role
    jti: str
    expires_at: datetime


class TokenService:
    algorithm = "HS256"

    def __init__(self, secret: str, issuer: str, audience: str, ttl: timedelta) -> None:
        if not secret:
            raise ValueError("JWT secret must not be empty")
        self._secret, self._issuer, self._audience, self._ttl = secret, issuer, audience, ttl

    def issue(self, operator: Operator, *, now: datetime | None = None) -> tuple[str, TokenClaims]:
        now = now or datetime.now(UTC)  # timezone-aware: .timestamp() is correct in any server TZ
        exp = now + self._ttl
        jti = uuid.uuid4().hex
        payload = {
            "sub": operator.username,
            "role": operator.role,
            "jti": jti,
            "iss": self._issuer,
            "aud": self._audience,
            "iat": int(now.timestamp()),
            "nbf": int(now.timestamp()),
            "exp": int(exp.timestamp()),
        }
        token = jwt.encode(payload, self._secret, algorithm=self.algorithm)
        return token, TokenClaims(operator.username, operator.role, jti, exp)

    def decode(self, token: str) -> TokenClaims:
        try:
            payload = jwt.decode(
                token,
                self._secret,
                algorithms=[self.algorithm],  # pinned: never trust the token header's alg
                audience=self._audience,
                issuer=self._issuer,
                leeway=10,
                options={"require": ["exp", "iat", "sub", "jti", "iss", "aud"]},
            )
        except jwt.PyJWTError as exc:
            raise AuthError("Invalid token") from exc
        role = payload.get("role")
        if role not in _ROLE_RANK:
            raise AuthError("Invalid token")
        return TokenClaims(payload["sub"], role, payload["jti"], datetime.fromtimestamp(payload["exp"], UTC))


# --------------------------------------------------------------------------- #
# Revocation (logout) and login throttling, backed by Redis
# --------------------------------------------------------------------------- #
class AsyncKV(Protocol):
    async def set(self, name: str, value: str, ex: int | None = None) -> object: ...
    async def exists(self, *names: str) -> int: ...
    async def incr(self, name: str) -> int: ...
    async def expire(self, name: str, time: int) -> object: ...
    async def get(self, name: str) -> object: ...
    async def delete(self, *names: str) -> object: ...


class RevocationStore:
    """Blocklist keyed by the token's ``jti`` (not the raw token) with a TTL equal to
    the token's remaining lifetime, so entries expire on their own."""

    def __init__(self, kv: AsyncKV) -> None:
        self._kv = kv

    async def revoke(self, claims: TokenClaims, *, now: datetime | None = None) -> None:
        now = now or datetime.now(UTC)
        ttl = int((claims.expires_at - now).total_seconds()) + 15  # cover the decode leeway
        if ttl <= 0:
            return
        try:
            await self._kv.set(f"revoked:{claims.jti}", "1", ex=ttl)
        except Exception as exc:
            raise RevocationUnavailable from exc

    async def is_revoked(self, jti: str) -> bool:
        try:
            return bool(await self._kv.exists(f"revoked:{jti}"))
        except Exception as exc:
            # Fail CLOSED: the original returned False here, silently re-enabling
            # every revoked token whenever Redis blipped.
            raise RevocationUnavailable from exc


class LoginThrottle:
    def __init__(self, kv: AsyncKV, max_failures: int, lockout_seconds: int) -> None:
        self._kv, self._max, self._lockout = kv, max_failures, lockout_seconds

    def _key(self, username: str) -> str:
        return f"login-failures:{hashlib.sha256(username.encode()).hexdigest()}"

    async def is_locked(self, username: str) -> bool:
        value = await self._kv.get(self._key(username))
        return value is not None and int(value) >= self._max

    async def record_failure(self, username: str) -> None:
        key = self._key(username)
        if await self._kv.incr(key) == 1:
            await self._kv.expire(key, self._lockout)

    async def reset(self, username: str) -> None:
        await self._kv.delete(self._key(username))


def _cli() -> None:  # pragma: no cover - interactive helper
    if sys.argv[1:] != ["hash-password"]:
        print("usage: python -m agent_workspace.security hash-password", file=sys.stderr)
        raise SystemExit(2)
    pw = getpass.getpass("Password: ")
    if pw != getpass.getpass("Repeat: "):
        raise SystemExit("Passwords do not match")
    print(hash_password(pw))


if __name__ == "__main__":
    _cli()
