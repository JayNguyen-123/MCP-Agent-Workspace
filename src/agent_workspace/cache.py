"""Opt-in semantic cache of *final answers* for completed, review-passed runs.

Changes from the original:
* It is actually wired in (the original module was never imported).
* It caches whole-task answers, not individual tool outputs. Returning a cached
  execution result for a merely *similar* prompt silently answers the wrong question
  (e.g. "sum to 10" vs "sum to 100" embed almost identically).
* The default distance threshold is far stricter (0.05 vs 0.15), and the cache is off
  unless SEMANTIC_CACHE_ENABLED=true.
* Async, pooled connections instead of a new blocking connection per call, and a
  non-blocking embedding client.
"""

from __future__ import annotations

import logging

from openai import AsyncOpenAI
from psycopg_pool import AsyncConnectionPool

from agent_workspace.metrics import CACHE_LOOKUPS

logger = logging.getLogger(__name__)


def _vector_literal(values: list[float]) -> str:
    return "[" + ",".join(f"{v:.7g}" for v in values) + "]"


class SemanticCache:
    def __init__(self, pool: AsyncConnectionPool, client: AsyncOpenAI, *, model: str, max_distance: float) -> None:
        self._pool, self._client, self._model, self._max_distance = pool, client, model, max_distance

    async def _embed(self, text: str) -> str:
        resp = await self._client.embeddings.create(input=[text], model=self._model)
        return _vector_literal(resp.data[0].embedding)

    async def lookup(self, task: str) -> str | None:
        try:
            vec = await self._embed(task)
            async with self._pool.connection() as conn:
                cur = await conn.execute(
                    """
                    SELECT answer, embedding <=> %(v)s::vector AS distance
                    FROM task_result_cache
                    ORDER BY embedding <=> %(v)s::vector
                    LIMIT 1
                    """,
                    {"v": vec},
                )
                row = await cur.fetchone()
        except Exception:
            CACHE_LOOKUPS.labels(result="error").inc()
            logger.warning("Semantic cache lookup failed; continuing without cache", exc_info=True)
            return None
        if row and row[1] <= self._max_distance:
            CACHE_LOOKUPS.labels(result="hit").inc()
            return row[0]
        CACHE_LOOKUPS.labels(result="miss").inc()
        return None

    async def store(self, task: str, answer: str) -> None:
        try:
            vec = await self._embed(task)
            async with self._pool.connection() as conn:
                await conn.execute(
                    "INSERT INTO task_result_cache (task, embedding, answer) VALUES (%s, %s::vector, %s)",
                    (task, vec, answer),
                )
        except Exception:
            logger.warning("Semantic cache write failed", exc_info=True)
