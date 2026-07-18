"""
app/utils/logging.py
Structured logging setup using structlog with PrintLoggerFactory.
Compatible with structlog v26+ — avoids stdlib-only processors.
"""

import logging
import sys
from typing import Any

import structlog
from structlog.types import EventDict, WrappedLogger

from app.config import get_settings


def add_app_context(
    logger: WrappedLogger,
    method_name: str,
    event_dict: EventDict,
) -> EventDict:
    """Add application-level context fields to every log entry."""
    settings = get_settings()
    event_dict["app"] = "intelliresearch"
    event_dict["env"] = settings.environment
    return event_dict


def setup_logging() -> None:
    """
    Configure structlog for structured logging.

    Uses PrintLoggerFactory (not stdlib) so we exclude any stdlib-only
    processors like add_logger_name that require a logger.name attribute.

    Development: human-readable coloured console output.
    Production:  JSON output for log aggregation (Datadog, CloudWatch).
    """
    settings = get_settings()
    log_level = getattr(logging, settings.log_level.upper(), logging.INFO)

    shared_processors: list[Any] = [
        structlog.contextvars.merge_contextvars,
        add_app_context,
        structlog.stdlib.add_log_level,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
    ]

    if settings.is_production:
        renderer: Any = structlog.processors.JSONRenderer()
    else:
        renderer = structlog.dev.ConsoleRenderer(colors=True)

    structlog.configure(
        processors=[*shared_processors, renderer],
        wrapper_class=structlog.make_filtering_bound_logger(log_level),
        context_class=dict,
        logger_factory=structlog.PrintLoggerFactory(sys.stdout),
        cache_logger_on_first_use=True,
    )

    # Keep stdlib logging at WARNING so third-party libs don't flood output
    logging.basicConfig(
        format="%(levelname)s: %(message)s",
        stream=sys.stdout,
        level=logging.WARNING,
    )


def get_logger(name: str) -> structlog.BoundLogger:
    """
    Get a named structlog logger.

    Args:
        name: Logger name (pass __name__ from the calling module).

    Returns:
        A bound structlog logger.

    Example:
        logger = get_logger(__name__)
        logger.info("Agent started", agent="planner")
    """
    return structlog.get_logger(name)
