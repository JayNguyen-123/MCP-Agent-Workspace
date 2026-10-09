"""Prometheus instrumentation.

Labels are deliberately low-cardinality. The original labelled a gauge by ``thread_id``
(unbounded series growth) and never reset it, so the loop alert fired forever once tripped.
"""

from __future__ import annotations

import logging

from prometheus_client import Counter, Histogram, start_http_server

logger = logging.getLogger(__name__)

RUNS_TOTAL = Counter("agent_runs_total", "Agent runs by terminal or paused outcome", ["outcome"])
NODE_LATENCY = Histogram(
    "agent_node_latency_seconds",
    "Latency of each graph node",
    ["node"],
    buckets=(0.1, 0.25, 0.5, 1, 2.5, 5, 10, 20, 40, 80, 160),
)
RUNTIME_FAILURES = Counter(
    "agent_runtime_failures_total", "Unhandled failures by component and exception type", ["component", "failure_type"]
)
TOKENS_TOTAL = Counter("agent_token_consumption_total", "LLM tokens consumed", ["node", "token_type"])
CIRCUIT_BREAKER_TRIPS = Counter("agent_circuit_breaker_trips_total", "Runs terminated by the hop limit")
RUN_HOPS = Histogram("agent_run_hops", "Agent hops used per run", buckets=(1, 2, 3, 5, 8, 12, 15, 20, 30, 50))
TOOL_CALLS = Counter("agent_tool_calls_total", "Sandbox tool invocations", ["tool", "outcome"])
APPROVAL_DECISIONS = Counter("agent_approval_decisions_total", "Human approval decisions", ["decision"])
CACHE_LOOKUPS = Counter("agent_semantic_cache_lookups_total", "Semantic cache lookups", ["result"])


def record_usage(node: str, message: object) -> None:
    usage = getattr(message, "usage_metadata", None) or {}
    if usage:
        TOKENS_TOTAL.labels(node=node, token_type="input").inc(usage.get("input_tokens", 0))
        TOKENS_TOTAL.labels(node=node, token_type="output").inc(usage.get("output_tokens", 0))


def start_metrics_server(port: int) -> None:
    """Expose /metrics on a dedicated port that is *not* published outside the cluster."""
    try:
        start_http_server(port)
        logger.info("Prometheus metrics listening on :%d", port)
    except OSError:
        logger.exception("Could not start metrics server on port %d", port)
