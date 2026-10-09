-- Runs once, on first start of an empty data volume.
-- Application tables (agent_runs, approval_requests) are created by the API on startup;
-- LangGraph checkpoint tables are created by AsyncPostgresSaver.setup().
-- For schema changes after go-live, adopt Alembic migrations instead of create_all.

CREATE EXTENSION IF NOT EXISTS vector;

-- Opt-in semantic cache of final answers (SEMANTIC_CACHE_ENABLED=true).
CREATE TABLE IF NOT EXISTS task_result_cache (
    id          BIGSERIAL PRIMARY KEY,
    task        TEXT        NOT NULL,
    embedding   VECTOR(1536) NOT NULL,          -- text-embedding-3-small
    answer      TEXT        NOT NULL,
    created_at  TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS task_result_cache_hnsw
    ON task_result_cache USING hnsw (embedding vector_cosine_ops);
