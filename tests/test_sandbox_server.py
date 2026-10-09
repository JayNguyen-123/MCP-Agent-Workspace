import asyncio
import json

import pytest

pytest.importorskip("mcp")

from mcp.client.client import Client  # noqa: E402

from sandbox_server import server  # noqa: E402


@pytest.fixture(autouse=True)
def isolated_workspace(tmp_path, monkeypatch):
    monkeypatch.setattr(server, "WORKSPACE_ROOT", tmp_path)


def call(name: str, **args) -> str:
    async def go():
        async with Client(server.mcp) as client:
            result = await client.call_tool(name, args)
            return "".join(getattr(b, "text", "") for b in result.content)

    return asyncio.run(go())


def test_tool_catalogue():
    async def go():
        async with Client(server.mcp) as client:
            return {t.name: set(t.input_schema["properties"]) for t in (await client.list_tools()).tools}

    tools = asyncio.run(go())
    assert tools == {
        "save_and_run_python_code": {"session_id", "filename", "code_content"},
        "lint_python_file": {"session_id", "filename"},
        "install_project_dependency": {"session_id", "package_name"},
    }


def test_run_roundtrip():
    out = call("save_and_run_python_code", session_id="s1", filename="calc", code_content="print(4+4)")
    assert "exit code 0" in out and "8" in out


def test_sessions_cannot_see_each_other():
    call("save_and_run_python_code", session_id="a", filename="secret.py", code_content="pass")
    out = call(
        "save_and_run_python_code",
        session_id="b",
        filename="peek.py",
        code_content="import os; print(os.path.exists('secret.py'), os.listdir('..'))",
    )
    assert "False" in out


def test_traversal_rejected_as_tool_result():
    out = call("save_and_run_python_code", session_id="s1", filename="../../etc/x.py", code_content="1")
    assert out.startswith("Rejected")


def test_install_rejects_flags():
    out = call("install_project_dependency", session_id="s1", package_name="--index-url=http://evil x")
    assert out.startswith("Rejected")


# ------------------------------------------------------------- HTTP auth layer
async def _asgi(app, path: str, headers: dict[str, str]) -> int:
    sent = []
    scope = {
        "type": "http",
        "method": "POST",
        "path": path,
        "headers": [(k.lower().encode(), v.encode()) for k, v in headers.items()],
        "query_string": b"",
    }

    async def receive():
        return {"type": "http.request", "body": json.dumps({}).encode(), "more_body": False}

    async def send(message):
        sent.append(message)

    await app(scope, receive, send)
    return next(m["status"] for m in sent if m["type"] == "http.response.start")


def test_bearer_middleware():
    reached = []

    async def inner(scope, receive, send):
        reached.append(scope["path"])
        await send({"type": "http.response.start", "status": 200, "headers": []})
        await send({"type": "http.response.body", "body": b""})

    app = server.BearerAuthMiddleware(inner, "t" * 40)
    assert asyncio.run(_asgi(app, "/mcp", {})) == 401
    assert asyncio.run(_asgi(app, "/mcp", {"Authorization": "Bearer wrong"})) == 401
    assert asyncio.run(_asgi(app, "/mcp", {"Authorization": "Bearer " + "t" * 40})) == 200
    assert asyncio.run(_asgi(app, "/healthz", {})) == 200
    assert reached == ["/mcp", "/healthz"]


def test_build_app_requires_strong_token(monkeypatch):
    monkeypatch.setenv("SANDBOX_AUTH_TOKEN", "short")
    with pytest.raises(RuntimeError):
        server.build_app()
