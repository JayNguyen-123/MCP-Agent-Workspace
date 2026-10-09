"""ASGI entrypoint: ``uvicorn agent_workspace.main:app``."""

from agent_workspace.api import create_app
from agent_workspace.bootstrap import container_factory
from agent_workspace.config import get_settings
from agent_workspace.logging_setup import configure_logging

settings = get_settings()
configure_logging(settings.log_level, settings.log_json)
app = create_app(container_factory(settings), settings=settings)
