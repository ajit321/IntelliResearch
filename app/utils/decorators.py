"""
app/utils/decorators.py
Production-grade decorators: circuit breaker, retry with backoff,
execution timer, idempotency guard, and structured error wrapper.
"""

import asyncio
import functools
import hashlib
import time
from collections.abc import Callable
from enum import Enum
from typing import Any, TypeVar

from app.utils.logging import get_logger

logger = get_logger(__name__)

F = TypeVar("F", bound=Callable[..., Any])


# ── Circuit Breaker ───────────────────────────────────────────────────────────

class CircuitState(Enum):
    """Possible states of a circuit breaker."""
    CLOSED = "closed"       # Normal — requests flow through
    OPEN = "open"           # Tripped — requests fail fast
    HALF_OPEN = "half_open" # Testing — one request allowed through


class CircuitBreaker:
    """
    Stateful circuit breaker implementation (Ralph Loop pattern).

    Prevents cascading failures by stopping requests to a failing service
    and automatically retrying after a timeout period.
    """

    def __init__(self, failure_threshold: int = 3, timeout: int = 30) -> None:
        """
        Initialise the circuit breaker.

        Args:
            failure_threshold: Number of failures before tripping open.
            timeout: Seconds before attempting recovery (HALF_OPEN state).
        """
        self.failure_threshold = failure_threshold
        self.timeout = timeout
        self.failure_count: int = 0
        self.last_failure_time: float = 0.0
        self.state: CircuitState = CircuitState.CLOSED

    def record_failure(self) -> None:
        """Record a failure and trip the breaker if threshold is reached."""
        self.failure_count += 1
        self.last_failure_time = time.monotonic()
        if self.failure_count >= self.failure_threshold:
            self.state = CircuitState.OPEN
            logger.warning(
                "Circuit breaker tripped",
                failures=self.failure_count,
                timeout=self.timeout,
            )

    def record_success(self) -> None:
        """Record a success and reset the breaker to closed."""
        self.failure_count = 0
        self.state = CircuitState.CLOSED

    def can_attempt(self) -> bool:
        """Return True if a request should be attempted."""
        if self.state == CircuitState.CLOSED:
            return True
        if self.state == CircuitState.OPEN:
            elapsed = time.monotonic() - self.last_failure_time
            if elapsed >= self.timeout:
                self.state = CircuitState.HALF_OPEN
                return True
            return False
        # HALF_OPEN: allow one attempt
        return True


# Module-level registry of circuit breakers (one per service name)
_breakers: dict[str, CircuitBreaker] = {}


def circuit_breaker(service_name: str, failure_threshold: int = 3, timeout: int = 30) -> Callable[[F], F]:
    """
    Decorator that wraps an async function with a circuit breaker.

    Args:
        service_name: Unique name for this service/circuit.
        failure_threshold: Failures before opening the circuit.
        timeout: Seconds before attempting recovery.

    Returns:
        Decorated async function that fails fast when circuit is open.

    Example:
        @circuit_breaker("arxiv_api", failure_threshold=3, timeout=30)
        async def fetch_arxiv(query: str) -> list[dict]:
            ...
    """
    if service_name not in _breakers:
        _breakers[service_name] = CircuitBreaker(failure_threshold, timeout)
    breaker = _breakers[service_name]

    def decorator(func: F) -> F:
        @functools.wraps(func)
        async def wrapper(*args: Any, **kwargs: Any) -> Any:
            if not breaker.can_attempt():
                logger.error(
                    "Circuit open — skipping call",
                    service=service_name,
                    state=breaker.state.value,
                )
                raise RuntimeError(
                    f"Circuit breaker OPEN for '{service_name}'. "
                    f"Retry after {breaker.timeout}s."
                )
            try:
                result = await func(*args, **kwargs)
                breaker.record_success()
                return result
            except Exception as exc:
                breaker.record_failure()
                logger.error(
                    "Service call failed",
                    service=service_name,
                    error=str(exc),
                    failures=breaker.failure_count,
                )
                raise

        return wrapper  # type: ignore[return-value]

    return decorator


# ── Retry with Exponential Backoff ───────────────────────────────────────────

def retry(
    max_attempts: int = 3,
    delay: float = 1.0,
    backoff: float = 2.0,
    exceptions: tuple[type[Exception], ...] = (Exception,),
) -> Callable[[F], F]:
    """
    Decorator that retries an async function on failure with exponential backoff.

    Args:
        max_attempts: Maximum number of total attempts.
        delay: Initial delay in seconds before first retry.
        backoff: Multiplier applied to delay on each retry.
        exceptions: Tuple of exception types to catch and retry.

    Example:
        @retry(max_attempts=3, delay=1.0, backoff=2.0)
        async def call_llm(prompt: str) -> str:
            ...
    """
    def decorator(func: F) -> F:
        @functools.wraps(func)
        async def wrapper(*args: Any, **kwargs: Any) -> Any:
            current_delay = delay
            for attempt in range(1, max_attempts + 1):
                try:
                    return await func(*args, **kwargs)
                except exceptions as exc:
                    if attempt == max_attempts:
                        logger.error(
                            "All retry attempts exhausted",
                            function=func.__name__,
                            attempts=max_attempts,
                            error=str(exc),
                        )
                        raise
                    logger.warning(
                        "Retrying after failure",
                        function=func.__name__,
                        attempt=attempt,
                        delay=current_delay,
                        error=str(exc),
                    )
                    await asyncio.sleep(current_delay)
                    current_delay *= backoff

        return wrapper  # type: ignore[return-value]

    return decorator


# ── Execution Timer ───────────────────────────────────────────────────────────

def timer(func: F) -> F:
    """
    Decorator that logs the execution time of an async function.

    Example:
        @timer
        async def run_analysis(state: ResearchState) -> dict:
            ...
    """
    @functools.wraps(func)
    async def wrapper(*args: Any, **kwargs: Any) -> Any:
        start = time.monotonic()
        result = await func(*args, **kwargs)
        elapsed = round(time.monotonic() - start, 3)
        logger.info(
            "Function completed",
            function=func.__name__,
            duration_seconds=elapsed,
        )
        return result

    return wrapper  # type: ignore[return-value]


# ── Idempotency Guard ─────────────────────────────────────────────────────────

_idempotency_cache: dict[str, Any] = {}


def idempotent(key_fn: Callable[..., str]) -> Callable[[F], F]:
    """
    Decorator that prevents duplicate execution for the same logical request.

    Args:
        key_fn: Function that derives the idempotency key from call arguments.

    Example:
        @idempotent(key_fn=lambda query, depth: f"{query}:{depth}")
        async def start_research(query: str, depth: str) -> dict:
            ...
    """
    def decorator(func: F) -> F:
        @functools.wraps(func)
        async def wrapper(*args: Any, **kwargs: Any) -> Any:
            raw_key = key_fn(*args, **kwargs)
            idem_key = hashlib.sha256(raw_key.encode()).hexdigest()

            if idem_key in _idempotency_cache:
                logger.info(
                    "Idempotent cache hit — returning cached result",
                    function=func.__name__,
                    key_hash=idem_key[:8],
                )
                return _idempotency_cache[idem_key]

            result = await func(*args, **kwargs)
            _idempotency_cache[idem_key] = result
            logger.info(
                "Idempotency key stored",
                function=func.__name__,
                key_hash=idem_key[:8],
            )
            return result

        return wrapper  # type: ignore[return-value]

    return decorator
