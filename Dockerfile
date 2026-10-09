# syntax=docker/dockerfile:1.7
# One Dockerfile, three images:
#   docker build --target api     -t agent-workspace/api .
#   docker build --target sandbox -t agent-workspace/sandbox .
#   docker build --target ui      -t agent-workspace/ui .
#
# Requires a committed uv.lock (run `uv lock`); builds are --frozen for reproducibility.

ARG PYTHON_VERSION=3.12

FROM ghcr.io/astral-sh/uv:0.11 AS uv

# --------------------------------------------------------------------------- base
FROM python:${PYTHON_VERSION}-slim-bookworm AS base
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never \
    UV_PROJECT_ENVIRONMENT=/opt/venv \
    PATH="/opt/venv/bin:${PATH}"
RUN groupadd --system --gid 10001 app \
 && useradd --system --uid 10001 --gid app --home-dir /home/app --create-home --shell /usr/sbin/nologin app

# -------------------------------------------------------------------- builders
FROM base AS builder
COPY --from=uv /uv /usr/local/bin/uv
WORKDIR /build
COPY pyproject.toml uv.lock README.md ./

FROM builder AS build-api
RUN --mount=type=cache,target=/root/.cache/uv uv sync --frozen --no-dev --no-install-project --extra api
COPY src ./src
RUN --mount=type=cache,target=/root/.cache/uv uv sync --frozen --no-dev --no-editable --extra api

FROM builder AS build-sandbox
RUN --mount=type=cache,target=/root/.cache/uv uv sync --frozen --no-dev --no-install-project --extra sandbox
COPY src ./src
RUN --mount=type=cache,target=/root/.cache/uv uv sync --frozen --no-dev --no-editable --extra sandbox

FROM builder AS build-ui
RUN --mount=type=cache,target=/root/.cache/uv uv sync --frozen --no-dev --no-install-project --extra ui

# ------------------------------------------------------------------------- api
FROM base AS api
COPY --from=build-api /opt/venv /opt/venv
USER app
EXPOSE 8000 8001
HEALTHCHECK --interval=15s --timeout=3s --start-period=30s --retries=3 \
  CMD python -c "import urllib.request,sys; sys.exit(urllib.request.urlopen('http://127.0.0.1:8000/healthz', timeout=2).status != 200)"
CMD ["uvicorn", "agent_workspace.main:app", "--host", "0.0.0.0", "--port", "8000", \
     "--proxy-headers", "--forwarded-allow-ips", "*", "--no-server-header", "--timeout-graceful-shutdown", "30"]

# --------------------------------------------------------------------- sandbox
FROM base AS sandbox
COPY --from=build-sandbox /opt/venv /opt/venv
# Agent code runs in a *separate* interpreter environment so that packages the agent
# installs can never shadow or tamper with the MCP server's own dependencies.
RUN python -m venv /opt/sandbox-venv \
 && mkdir -p /workspace \
 && chown -R app:app /opt/sandbox-venv /workspace
ENV SANDBOX_PYTHON=/opt/sandbox-venv/bin/python \
    SANDBOX_WORKSPACE=/workspace \
    SANDBOX_PORT=9000
USER app
EXPOSE 9000
HEALTHCHECK --interval=15s --timeout=3s --start-period=10s --retries=3 \
  CMD python -c "import urllib.request,sys; sys.exit(urllib.request.urlopen('http://127.0.0.1:9000/healthz', timeout=2).status != 200)"
CMD ["python", "-m", "sandbox_server.server"]

# -------------------------------------------------------------------------- ui
FROM base AS ui
COPY --from=build-ui /opt/venv /opt/venv
WORKDIR /app
COPY --chown=app:app ui/app.py ./app.py
USER app
EXPOSE 8501
HEALTHCHECK --interval=15s --timeout=3s --start-period=20s --retries=3 \
  CMD python -c "import urllib.request,sys; sys.exit(urllib.request.urlopen('http://127.0.0.1:8501/_stcore/health', timeout=2).status != 200)"
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0", \
     "--server.headless=true", "--browser.gatherUsageStats=false"]
