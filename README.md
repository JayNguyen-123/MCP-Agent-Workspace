# Agent Workspace

A hierarchical multi-agent coding team (Supervisor → Coder ⇄ Tools → Reviewer) built on
**LangGraph**, with code execution in an isolated **MCP** sandbox, **human approval** for
package installs, and a **FastAPI** control plane with an operator dashboard.

See [`REVIEW.md`](REVIEW.md) for the production-readiness review that produced this version.

## Architecture

```
               ┌──────────── frontend ────────────┐
  browser ──►  ui (Streamlit) ──► agent-service (FastAPI + LangGraph) ──► Anthropic API   [egress]
                                    │      │      │
                         [data]     │      │      │  [sandbox: internal, no internet]
                    postgres ◄──────┘      │      └──► sandbox (MCP over HTTP, bearer auth)
                    (runs, approvals,      │              per-run dirs · rlimits · no secrets
                     checkpoints, cache)   │
                    redis ◄────────────────┘ (token revocation, login throttling)
                                    │
                         [metrics]  └──► prometheus ──► grafana ──► Slack
```

| Component | Path | Responsibility |
|---|---|---|
| Graph | `src/agent_workspace/graph.py` | Supervisor/coder/reviewer routing, approval gate (`interrupt()`), hop-limit circuit breaker |
| Runner | `src/agent_workspace/runner.py` | Background execution, bounded concurrency, timeouts, DB state sync |
| API | `src/agent_workspace/api.py` | Auth, runs, approvals, audit trail, health |
| Security | `src/agent_workspace/security.py` | scrypt passwords, JWTs, Redis revocation (fail-closed), login throttling |
| Store | `src/agent_workspace/store.py` | Async SQL; atomic conditional state transitions |
| Sandbox | `src/sandbox_server/` | MCP server: write/lint/run Python, gated package installs |
| UI | `ui/app.py` | Sign-in, submit tasks, watch runs, approve/deny, audit trail |
| Evals | `evals/run_evals.py` | LangSmith regression evals (manual/nightly) |

### Run lifecycle

`queued → running → (awaiting_approval ⇄ running)* → completed | terminated | failed`

When the coder requests `install_project_dependency`, the graph pauses *before* executing
it. The checkpoint is stored in Postgres, so the pause survives restarts and any replica
can resume it. An approver (not the run's owner — four-eyes rule) approves or denies; on
denial the coder receives an error tool result and must adapt.

## Quick start

```bash
uv lock                                   # generate and commit uv.lock (builds are --frozen)
cp .env.example .env                      # fill in every value
uv run --extra api python -m agent_workspace.security hash-password   # for OPERATORS_JSON
docker compose up --build -d
```

| Service | URL (bound to localhost only) |
|---|---|
| Dashboard | http://localhost:8501 |
| API | http://localhost:8000 (`/docs` only when `ENVIRONMENT != production`) |
| Grafana | http://localhost:3000 |
| Prometheus | http://localhost:9090 |

**Package installs:** the sandbox has no internet access by design. To allow approved
installs, set `SANDBOX_PIP_INDEX_URL` to an internal PyPI mirror reachable on the
`sandbox` network, and restrict requests with `SANDBOX_PACKAGE_ALLOWLIST`.

## API

| Method | Path | Role | |
|---|---|---|---|
| POST | `/auth/token` | — | `{username, password}` → bearer token (15 min) |
| POST | `/auth/logout` | any | Revokes the current token immediately |
| GET | `/auth/me` | any | |
| POST | `/agent/runs` | operator | `{task}` → 202 with server-generated run id |
| GET | `/agent/runs`, `/agent/runs/{id}` | operator | Own runs; approvers see all |
| GET | `/agent/approvals?status=PENDING` | approver | |
| POST | `/agent/approvals/{id}/decision` | approver | `{decision: APPROVED\|DENIED, reason?}`; 409 if already decided |
| GET | `/agent/audit-trail` | approver | Every decision, who made it, when, why |
| GET | `/healthz`, `/readyz` | — | Liveness / readiness (DB + Redis) |

Prometheus metrics are served on port 8001, which is not published outside the stack.

## Development

```bash
uv sync --all-extras --group dev
uv run ruff check . && uv run ruff format --check .
uv run pytest            # hermetic: SQLite + fakes, no services or API keys
```

Run the sandbox locally:
`SANDBOX_AUTH_TOKEN=$(openssl rand -hex 32) SANDBOX_ALLOWED_HOSTS='127.0.0.1:*' uv run --extra sandbox python -m sandbox_server.server`

Evals (paid API calls; needs a running sandbox): `uv run --extra api --group evals python evals/run_evals.py`

## Configuration

All settings are environment variables (see `src/agent_workspace/config.py` and
`.env.example`). With `ENVIRONMENT=production` the API refuses to start if
`JWT_SECRET_KEY` or `SANDBOX_AUTH_TOKEN` is weak or `ANTHROPIC_API_KEY` is missing.

Key guardrails: `MAX_HOPS` (25), `RUN_TIMEOUT_S` (900), `MAX_CONCURRENT_RUNS` (8),
`SANDBOX_TIMEOUT_S` (10), `SANDBOX_MEMORY_BYTES` (512 MiB), `SANDBOX_MAX_OUTPUT_BYTES` (16 KiB).

## Operations

* **Shutdown:** `docker compose down` keeps all volumes (runs, audit trail, checkpoints).
  `docker compose down -v` deletes them.
* **Schema changes:** tables are created on startup (`AUTO_CREATE_SCHEMA`). Adopt Alembic
  before the first schema change in production.
* **Scaling:** API replicas are stateless apart from in-flight runs; approvals and
  checkpoints live in Postgres, so any replica can resume a paused run. A run whose
  replica died mid-execution is marked `failed` by the next replica that starts after
  the run timeout has elapsed.
