"""Production composition root. ``uvicorn agent_workspace.main:app``."""

from __future__ import annotations

import asyncio
import logging
from collections.abc import AsyncIterator
from contextlib import AsyncExitStack, asynccontextmanager
from datetime import timedelta

from langchain_anthropic import ChatAnthropic
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from psycopg.rows import dict_row
from psycopg_pool import AsyncConnectionPool
from redis.asyncio import Redis

from agent_workspace.api import Container
from agent_workspace.config import Settings
from agent_workspace.graph import AgentModels, build_graph
from agent_workspace.metrics import start_metrics_server
from agent_workspace.runner import AgentService
from agent_workspace.sandbox_tools import SandboxClient, load_sandbox_tools
from agent_workspace.security import LoginThrottle, OperatorDirectory, RevocationStore, TokenService
from agent_workspace.store import Store, make_engine

logger = logging.getLogger(__name__)


def _chat(settings: Settings, model: str, temperature: float) -> ChatAnthropic:
    return ChatAnthropic(
        model=model,
        temperature=temperature,
        max_tokens=settings.llm_max_tokens,
        timeout=settings.llm_timeout_s,
        max_retries=settings.llm_max_retries,
        api_key=settings.anthropic_api_key,
    )


async def _load_tools_with_retry(client: SandboxClient, attempts: int = 10):
    delay = 1.0
    for attempt in range(1, attempts + 1):
        try:
            tools = await load_sandbox_tools(client)
            logger.info("Loaded %d sandbox tools: %s", len(tools), [t.name for t in tools])
            return tools
        except Exception:
            if attempt == attempts:
                raise
            logger.warning("Sandbox not ready (attempt %d/%d); retrying in %.0fs", attempt, attempts, delay)
            await asyncio.sleep(delay)
            delay = min(delay * 2, 15)
    raise RuntimeError("unreachable")


def container_factory(settings: Settings):
    @asynccontextmanager
    async def factory() -> AsyncIterator[Container]:
        async with AsyncExitStack() as stack:
            engine = make_engine(settings.database_url.get_secret_value(), pool_size=settings.db_pool_size)
            stack.push_async_callback(engine.dispose)
            store = Store(engine)
            if settings.auto_create_schema:
                await store.create_schema()
            orphaned = await store.fail_orphaned_runs(older_than=timedelta(seconds=settings.run_timeout_s + 60))
            if orphaned:
                logger.warning("Marked %d orphaned runs as failed after restart", orphaned)

            redis = Redis.from_url(settings.redis_url.get_secret_value(), decode_responses=True)
            stack.push_async_callback(redis.aclose)

            # Durable checkpoints: paused (awaiting-approval) runs survive restarts and are
            # resumable from any replica. The original used in-process MemorySaver.
            pool = AsyncConnectionPool(
                settings.psycopg_dsn,
                max_size=settings.db_pool_size,
                kwargs={"autocommit": True, "prepare_threshold": 0, "row_factory": dict_row},
                open=False,
            )
            await pool.open()
            stack.push_async_callback(pool.close)
            checkpointer = AsyncPostgresSaver(pool)
            await checkpointer.setup()

            sandbox = SandboxClient(
                settings.sandbox_mcp_url,
                settings.sandbox_auth_token.get_secret_value(),
                timeout_s=settings.sandbox_call_timeout_s,
            )
            tools = await _load_tools_with_retry(sandbox)
            models = AgentModels(
                supervisor=_chat(settings, settings.supervisor_model, 0.0),
                coder=_chat(settings, settings.coder_model, 0.2),
                reviewer=_chat(settings, settings.reviewer_model, 0.0),
            )
            graph = build_graph(models, tools, max_hops=settings.max_hops, checkpointer=checkpointer)

            cache = None
            if settings.semantic_cache_enabled:
                from openai import AsyncOpenAI

                from agent_workspace.cache import SemanticCache

                cache = SemanticCache(
                    pool,
                    AsyncOpenAI(api_key=settings.openai_api_key.get_secret_value()),
                    model=settings.embedding_model,
                    max_distance=settings.semantic_cache_max_distance,
                )

            service = AgentService(
                graph,
                store,
                max_hops=settings.max_hops,
                max_concurrent_runs=settings.max_concurrent_runs,
                run_timeout_s=settings.run_timeout_s,
                cache=cache,
            )
            stack.push_async_callback(service.shutdown)

            start_metrics_server(settings.metrics_port)
            yield Container(
                settings=settings,
                store=store,
                service=service,
                directory=OperatorDirectory(settings.operators_json.get_secret_value()),
                tokens=TokenService(
                    settings.jwt_secret_key.get_secret_value(),
                    settings.jwt_issuer,
                    settings.jwt_audience,
                    timedelta(minutes=settings.access_token_ttl_minutes),
                ),
                revocations=RevocationStore(redis),
                throttle=LoginThrottle(redis, settings.login_max_failures, settings.login_lockout_seconds),
                ping_redis=redis.ping,
                allow_self_approval=settings.allow_self_approval,
            )

    return factory
