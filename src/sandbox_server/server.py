"""MCP server exposing the coding-sandbox tools over Streamable HTTP.

Runs in its own container (see docker-compose.yml) so model-generated code never
shares a process, filesystem or network with the API service and its secrets.
"""

from __future__ import annotations

import hmac
import logging
import os
from pathlib import Path

import uvicorn
from mcp.server.mcpserver import MCPServer
from mcp.server.transport_security import TransportSecuritySettings
from starlette.requests import Request
from starlette.responses import JSONResponse
from starlette.types import ASGIApp, Receive, Scope, Send

from sandbox_server import executor
from sandbox_server.executor import SandboxError, SandboxLimits

logger = logging.getLogger("sandbox_server")

WORKSPACE_ROOT = Path(os.getenv("SANDBOX_WORKSPACE", "./sandbox_workspace")).resolve()
LIMITS = SandboxLimits.from_env()

mcp = MCPServer(
    name="LocalCodingSandbox",
    instructions=(
        "Tools for writing, linting and running Python inside an isolated sandbox. "
        "Each session has a private working directory."
    ),
)


def _workdir(session_id: str) -> Path:
    return executor.session_dir(WORKSPACE_ROOT, session_id)


@mcp.tool()
async def save_and_run_python_code(session_id: str, filename: str, code_content: str) -> str:
    """Save Python source to `filename` in the session workspace and execute it.

    Returns exit status, stdout and stderr (both truncated if very large).
    """
    try:
        workdir = _workdir(session_id)
        path = executor.write_source(workdir, filename, code_content, LIMITS)
        result = await executor.run_python_file(workdir, path, LIMITS)
        return result.render()
    except SandboxError as exc:
        return f"Rejected: {exc}"


@mcp.tool()
async def lint_python_file(session_id: str, filename: str) -> str:
    """Run `ruff check` on a file previously saved in the session workspace."""
    try:
        result = await executor.lint_python_file(_workdir(session_id), filename, LIMITS)
    except SandboxError as exc:
        return f"Rejected: {exc}"
    if result.exit_code == 0:
        return "Lint passed: no issues found."
    return f"Lint reported issues:\n{result.stdout}{result.stderr}"


@mcp.tool()
async def install_project_dependency(session_id: str, package_name: str) -> str:
    """Install a third-party package (PEP 508 spec, e.g. `pandas>=2`) for sandbox code.

    Requires human approval; the orchestrator pauses before this tool runs.
    """
    _workdir(session_id)  # validates the session id
    try:
        result = await executor.install_package(package_name, LIMITS)
    except SandboxError as exc:
        return f"Rejected: {exc}"
    if result.exit_code == 0:
        return f"Installed {package_name!r} successfully."
    return f"Installation failed.\n{result.stderr or result.stdout}"


@mcp.custom_route("/healthz", methods=["GET"], include_in_schema=False)
async def healthz(_: Request) -> JSONResponse:
    return JSONResponse({"status": "ok"})


class BearerAuthMiddleware:
    """Require a shared bearer token on every route except /healthz."""

    def __init__(self, app: ASGIApp, token: str) -> None:
        self.app = app
        self.expected = f"Bearer {token}".encode()

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] == "http" and scope.get("path") != "/healthz":
            supplied = dict(scope.get("headers") or []).get(b"authorization", b"")
            if not hmac.compare_digest(supplied, self.expected):
                await JSONResponse({"error": "unauthorized"}, status_code=401)(scope, receive, send)
                return
        await self.app(scope, receive, send)


def build_app() -> ASGIApp:
    token = os.getenv("SANDBOX_AUTH_TOKEN", "")
    if len(token) < 32:
        raise RuntimeError("SANDBOX_AUTH_TOKEN must be set to a random value of at least 32 characters.")
    allowed_hosts = [h.strip() for h in os.getenv("SANDBOX_ALLOWED_HOSTS", "sandbox:*,localhost:*").split(",")]
    WORKSPACE_ROOT.mkdir(parents=True, exist_ok=True)
    app = mcp.streamable_http_app(
        stateless_http=True,
        json_response=True,
        transport_security=TransportSecuritySettings(allowed_hosts=allowed_hosts, allowed_origins=[]),
    )
    return BearerAuthMiddleware(app, token)


def main() -> None:
    logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"))
    uvicorn.run(
        build_app(),
        host=os.getenv("SANDBOX_HOST", "0.0.0.0"),  # noqa: S104 - container-internal network only
        port=int(os.getenv("SANDBOX_PORT", "9000")),
        log_level=os.getenv("LOG_LEVEL", "info").lower(),
    )


if __name__ == "__main__":
    main()
