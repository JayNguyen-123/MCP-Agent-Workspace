"""Bridge the sandbox MCP server's tools into LangChain tools.

* ``session_id`` is hidden from the model and injected from the run's thread id, so
  the LLM cannot read or overwrite another run's workspace.
* Results are converted to plain text (the original returned raw CallToolResult objects).
* Each call opens a short-lived session against a stateless server, which survives
  sandbox restarts without reconnect logic.
"""

# No `from __future__ import annotations` here: LangChain detects the injected
# `config: RunnableConfig` parameter from the live annotation.
import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

import httpx2
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import BaseTool, StructuredTool
from mcp.client.session import ClientSession
from mcp.client.streamable_http import streamable_http_client
from pydantic import Field, create_model

from agent_workspace.metrics import TOOL_CALLS

logger = logging.getLogger(__name__)

HIDDEN_ARGS = frozenset({"session_id"})
_JSON_TYPES: dict[str, type] = {"string": str, "integer": int, "number": float, "boolean": bool}


class SandboxClient:
    def __init__(self, url: str, token: str, *, timeout_s: float) -> None:
        self.url, self._token, self._timeout_s = url, token, timeout_s

    @asynccontextmanager
    async def _session(self) -> AsyncIterator[ClientSession]:
        timeout = httpx2.Timeout(30.0, read=self._timeout_s)
        headers = {"Authorization": f"Bearer {self._token}"}
        async with httpx2.AsyncClient(headers=headers, timeout=timeout) as http:
            async with streamable_http_client(self.url, http_client=http) as (read, write):
                async with ClientSession(read, write) as session:
                    await session.initialize()
                    yield session

    async def list_tools(self) -> list[Any]:
        async with self._session() as s:
            return list((await s.list_tools()).tools)

    async def call_tool(self, name: str, arguments: dict[str, Any]) -> str:
        async with self._session() as s:
            result = await s.call_tool(name, arguments)
        text = "\n".join(getattr(block, "text", "") for block in result.content if getattr(block, "type", "") == "text")
        return f"Tool error: {text}" if result.is_error else text


def _args_model(tool_name: str, schema: dict[str, Any]):
    props: dict[str, Any] = schema.get("properties", {})
    required = set(schema.get("required", []))
    fields: dict[str, Any] = {}
    for name, spec in props.items():
        if name in HIDDEN_ARGS:
            continue
        py_type = _JSON_TYPES.get(spec.get("type", ""), Any)
        default = ... if name in required else spec.get("default")
        fields[name] = (py_type, Field(default, description=spec.get("description") or spec.get("title")))
    return create_model(f"{tool_name}_args", **fields)


def to_langchain_tool(client: SandboxClient, mcp_tool: Any) -> BaseTool:
    name: str = mcp_tool.name

    async def _call(config: RunnableConfig, **kwargs: Any) -> str:
        thread_id = (config.get("configurable") or {}).get("thread_id")
        if not thread_id:
            raise RuntimeError("Sandbox tools require a thread_id in the run config.")
        try:
            out = await client.call_tool(name, {**kwargs, "session_id": thread_id})
        except Exception as exc:
            TOOL_CALLS.labels(tool=name, outcome="transport_error").inc()
            logger.exception("Sandbox call %s failed", name)
            return f"Sandbox unavailable ({type(exc).__name__}); try again shortly."
        TOOL_CALLS.labels(tool=name, outcome="error" if out.startswith(("Tool error", "Rejected")) else "ok").inc()
        return out

    return StructuredTool.from_function(
        coroutine=_call,
        name=name,
        description=mcp_tool.description or name,
        args_schema=_args_model(name, mcp_tool.input_schema),
    )


async def load_sandbox_tools(client: SandboxClient) -> list[BaseTool]:
    return [to_langchain_tool(client, t) for t in await client.list_tools()]
