"""
app/observability/langsmith.py
LangSmith tracing setup for observability.
Configures traces, token usage monitoring, and evaluation metrics.
"""

import os

from app.config import get_settings
from app.utils.logging import get_logger

logger = get_logger(__name__)


def setup_langsmith() -> None:
    """
    Configure LangSmith tracing environment variables.

    If LANGSMITH_API_KEY is not set, tracing is silently disabled.
    This allows the app to run without LangSmith in demo mode.
    """
    settings = get_settings()

    if not settings.tracing_enabled:
        logger.info("LangSmith tracing disabled (no API key or tracing flag off)")
        os.environ["LANGCHAIN_TRACING_V2"] = "false"
        return

    os.environ["LANGCHAIN_API_KEY"]      = settings.langsmith_api_key
    os.environ["LANGCHAIN_PROJECT"]      = settings.langsmith_project
    os.environ["LANGCHAIN_TRACING_V2"]   = "true"
    os.environ["LANGCHAIN_ENDPOINT"]     = settings.langchain_endpoint

    logger.info(
        "LangSmith tracing enabled",
        project=settings.langsmith_project,
        endpoint=settings.langchain_endpoint,
    )
