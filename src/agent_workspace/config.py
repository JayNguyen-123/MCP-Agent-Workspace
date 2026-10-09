"""Typed, validated application settings loaded from the environment (12-factor)."""

from __future__ import annotations

from functools import lru_cache
from typing import Literal

from pydantic import Field, SecretStr, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

_INSECURE_DEFAULTS = {"change-me", "secret", "super-secret-agent-signing-key"}


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    environment: Literal["development", "test", "production"] = "production"
    log_level: str = "INFO"
    log_json: bool = True

    # --- Storage -----------------------------------------------------------
    database_url: SecretStr = SecretStr("postgresql+psycopg://agent:agent@localhost:5432/agent_db")
    redis_url: SecretStr = SecretStr("redis://localhost:6379/0")
    auto_create_schema: bool = True
    db_pool_size: int = 10

    # --- Auth --------------------------------------------------------------
    jwt_secret_key: SecretStr = Field(default=SecretStr(""))
    jwt_issuer: str = "agent-workspace"
    jwt_audience: str = "agent-workspace-api"
    access_token_ttl_minutes: int = Field(default=15, ge=1, le=24 * 60)
    # JSON object: {"alice": {"password_hash": "scrypt$...", "role": "approver"}, ...}
    # Generate hashes with:  python -m agent_workspace.security hash-password
    operators_json: SecretStr = SecretStr("{}")
    login_max_failures: int = 5
    login_lockout_seconds: int = 900
    # Four-eyes rule: an approver may not approve installs requested by their own run.
    allow_self_approval: bool = False

    # --- LLM ---------------------------------------------------------------
    anthropic_api_key: SecretStr = SecretStr("")
    supervisor_model: str = "claude-sonnet-5-5"
    coder_model: str = "claude-sonnet-5-5"
    reviewer_model: str = "claude-sonnet-5-5"
    llm_timeout_s: float = 120.0
    llm_max_retries: int = 3
    llm_max_tokens: int = 4096

    # --- Orchestration guardrails ---------------------------------------------
    max_hops: int = Field(default=25, ge=1, le=200)  # each supervisor/coder/reviewer turn is one hop
    max_concurrent_runs: int = Field(default=8, ge=1)
    run_timeout_s: float = 900.0
    max_task_chars: int = 8000

    # --- Sandbox MCP server -----------------------------------------------------
    sandbox_mcp_url: str = "http://sandbox:9000/mcp"
    sandbox_auth_token: SecretStr = SecretStr("")
    sandbox_call_timeout_s: float = 240.0

    # --- Semantic cache (opt-in) -------------------------------------------------
    semantic_cache_enabled: bool = False
    semantic_cache_max_distance: float = Field(default=0.05, ge=0.0, le=1.0)
    openai_api_key: SecretStr = SecretStr("")
    embedding_model: str = "text-embedding-3-small"

    # --- HTTP --------------------------------------------------------------
    cors_allow_origins: list[str] = Field(default_factory=list)
    metrics_port: int = 8001

    @field_validator("database_url")
    @classmethod
    def _normalise_db_driver(cls, v: SecretStr) -> SecretStr:
        raw = v.get_secret_value()
        if raw.startswith("postgresql://"):
            raw = "postgresql+psycopg://" + raw.removeprefix("postgresql://")
        return SecretStr(raw)

    @model_validator(mode="after")
    def _fail_closed_in_production(self) -> Settings:
        if self.environment != "production":
            return self
        secret = self.jwt_secret_key.get_secret_value()
        problems = []
        if len(secret) < 32 or secret in _INSECURE_DEFAULTS:
            problems.append("JWT_SECRET_KEY must be a random value of at least 32 characters")
        if len(self.sandbox_auth_token.get_secret_value()) < 32:
            problems.append("SANDBOX_AUTH_TOKEN must be a random value of at least 32 characters")
        if not self.anthropic_api_key.get_secret_value():
            problems.append("ANTHROPIC_API_KEY is required")
        if self.semantic_cache_enabled and not self.openai_api_key.get_secret_value():
            problems.append("OPENAI_API_KEY is required when SEMANTIC_CACHE_ENABLED=true")
        if problems:
            raise ValueError("Refusing to start with insecure configuration: " + "; ".join(problems))
        return self

    @property
    def psycopg_dsn(self) -> str:
        """Plain libpq DSN for psycopg / the LangGraph checkpointer."""
        return self.database_url.get_secret_value().replace("postgresql+psycopg://", "postgresql://", 1)


@lru_cache
def get_settings() -> Settings:
    return Settings()
