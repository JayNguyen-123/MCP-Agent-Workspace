"""Structured logging with async-safe per-node / per-run context.

The original swapped the *global* LogRecord factory inside a context manager, which is
not safe under asyncio (concurrent nodes overwrote each other's factory) and its
``%(node_name)s`` format string raised KeyError for every record emitted outside a
node, including all third-party library logs.
"""

from __future__ import annotations

import contextvars
import json
import logging
import sys
from collections.abc import Iterator
from contextlib import contextmanager

node_var: contextvars.ContextVar[str] = contextvars.ContextVar("node", default="-")
run_var: contextvars.ContextVar[str] = contextvars.ContextVar("run_id", default="-")


class ContextFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        record.node = node_var.get()
        record.run_id = run_var.get()
        return True


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "ts": self.formatTime(record, "%Y-%m-%dT%H:%M:%S%z"),
            "level": record.levelname,
            "logger": record.name,
            "node": getattr(record, "node", "-"),
            "run_id": getattr(record, "run_id", "-"),
            "msg": record.getMessage(),
        }
        if record.exc_info:
            payload["exc"] = self.formatException(record.exc_info)
        return json.dumps(payload, default=str)


def configure_logging(level: str = "INFO", json_output: bool = True) -> None:
    handler = logging.StreamHandler(sys.stdout)
    handler.addFilter(ContextFilter())
    handler.setFormatter(
        JsonFormatter()
        if json_output
        else logging.Formatter("%(asctime)s %(levelname)s [%(run_id)s/%(node)s] %(name)s: %(message)s")
    )
    root = logging.getLogger()
    root.handlers[:] = [handler]
    root.setLevel(level.upper())


@contextmanager
def log_context(*, node: str | None = None, run_id: str | None = None) -> Iterator[None]:
    tokens = []
    if node is not None:
        tokens.append((node_var, node_var.set(node)))
    if run_id is not None:
        tokens.append((run_var, run_var.set(run_id)))
    try:
        yield
    finally:
        for var, token in reversed(tokens):
            var.reset(token)
