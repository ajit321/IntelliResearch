"""
app/security/guardrails.py
Security guardrails: prompt injection detection, content safety,
and query sanitisation.
"""

import re
from typing import NamedTuple

import bleach

from app.config import get_settings
from app.utils.logging import get_logger

logger = get_logger(__name__)
settings = get_settings()

# ── Prompt Injection Patterns ─────────────────────────────────────────────────

# Patterns indicative of prompt injection attacks
_INJECTION_PATTERNS: list[re.Pattern] = [
    re.compile(r"ignore\s+(all\s+)?previous\s+instructions?", re.IGNORECASE),
    re.compile(r"disregard\s+(all\s+)?prior\s+instructions?", re.IGNORECASE),
    re.compile(r"you\s+are\s+now\s+[a-z]+", re.IGNORECASE),
    re.compile(r"act\s+as\s+(if\s+you\s+are|a)\s+", re.IGNORECASE),
    re.compile(r"jailbreak", re.IGNORECASE),
    re.compile(r"DAN\s+mode", re.IGNORECASE),
    re.compile(r"override\s+system\s+prompt", re.IGNORECASE),
    re.compile(r"<\s*system\s*>", re.IGNORECASE),
    re.compile(r"\[SYSTEM\]", re.IGNORECASE),
    re.compile(r"reveal\s+your\s+(system\s+)?prompt", re.IGNORECASE),
]

# Patterns for potentially harmful content requests
_HARMFUL_PATTERNS: list[re.Pattern] = [
    re.compile(r"how\s+to\s+(make|build|create)\s+(bomb|weapon|malware|virus)", re.IGNORECASE),
    re.compile(r"(synthesize|manufacture)\s+(drug|chemical\s+weapon)", re.IGNORECASE),
]


class ValidationResult(NamedTuple):
    """Result of a guardrail validation check."""
    is_safe: bool
    reason: str


def check_prompt_injection(query: str) -> ValidationResult:
    """
    Check a query for prompt injection attack patterns.

    Args:
        query: User-submitted query string.

    Returns:
        ValidationResult with is_safe flag and reason string.
    """
    for pattern in _INJECTION_PATTERNS:
        if pattern.search(query):
            logger.warning(
                "Prompt injection attempt detected",
                pattern=pattern.pattern[:40],
                query_snippet=query[:60],
            )
            return ValidationResult(
                is_safe=False,
                reason="Potential prompt injection detected. Query blocked.",
            )
    return ValidationResult(is_safe=True, reason="")


def check_harmful_content(query: str) -> ValidationResult:
    """
    Check a query for requests for harmful content.

    Args:
        query: User-submitted query string.

    Returns:
        ValidationResult with is_safe flag and reason string.
    """
    for pattern in _HARMFUL_PATTERNS:
        if pattern.search(query):
            logger.warning(
                "Harmful content request detected",
                query_snippet=query[:60],
            )
            return ValidationResult(
                is_safe=False,
                reason="Query contains potentially harmful content and was blocked.",
            )
    return ValidationResult(is_safe=True, reason="")


def sanitise_query(query: str) -> str:
    """
    Sanitise a user query for safe processing.

    - Strips HTML tags to prevent XSS
    - Trims whitespace
    - Enforces max length from settings

    Args:
        query: Raw user query string.

    Returns:
        Sanitised query string.
    """
    # Strip HTML
    clean = bleach.clean(query, tags=[], attributes={}, strip=True)
    # Trim and enforce length
    clean = clean.strip()[: settings.max_query_length]
    logger.debug("Query sanitised", original_length=len(query), clean_length=len(clean))
    return clean


def validate_query(query: str) -> ValidationResult:
    """
    Run all guardrail checks on a query.

    Args:
        query: User-submitted query string (pre-sanitised).

    Returns:
        ValidationResult — if is_safe=False, request should be rejected.
    """
    if not query.strip():
        return ValidationResult(is_safe=False, reason="Query cannot be empty.")

    injection_check = check_prompt_injection(query)
    if not injection_check.is_safe:
        return injection_check

    harmful_check = check_harmful_content(query)
    if not harmful_check.is_safe:
        return harmful_check

    return ValidationResult(is_safe=True, reason="Query validated successfully.")
