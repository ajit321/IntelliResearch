"""
app/agents/market_agent.py
Market Agent — retrieves market and industry data (Wikipedia + mock market data).
In production: wire to financial APIs (Alpha Vantage, Polygon.io, etc.)
"""

from datetime import datetime, timezone
from typing import Any

import httpx

from app.agents.state import ResearchState, SourceDoc, AgentEvent
from app.utils.logging import get_logger
from app.utils.decorators import circuit_breaker, timer

logger = get_logger(__name__)


@circuit_breaker("wikipedia_api", failure_threshold=3, timeout=30)
async def _fetch_wikipedia(query: str) -> SourceDoc | None:
    """
    Fetch a Wikipedia summary for industry/market context.

    Args:
        query: Topic to look up.

    Returns:
        SourceDoc from Wikipedia or None if not found.
    """
    encoded = query.replace(" ", "_")
    url = f"https://en.wikipedia.org/api/rest_v1/page/summary/{encoded}"

    async with httpx.AsyncClient(timeout=10.0) as client:
        response = await client.get(
            url,
            headers={"User-Agent": "IntelliResearch/1.0"},
        )

    if response.status_code != 200:
        logger.warning("Wikipedia lookup failed", query=query[:40], status=response.status_code)
        return None

    data = response.json()
    return SourceDoc(
        title=data.get("title", query),
        url=data.get("content_urls", {}).get("desktop", {}).get("page", ""),
        content=data.get("extract", "")[:2000],
        source_type="wikipedia",
        confidence=0.85,
        published_date="",
    )


@timer
async def market_agent_node(state: ResearchState) -> dict[str, Any]:
    """
    Market Agent node — retrieves market/industry context.

    Currently uses Wikipedia for market context.
    Production extension: wire to Alpha Vantage, Yahoo Finance, Polygon.io.

    Args:
        state: Current pipeline state.

    Returns:
        State update dict with sources, raw_texts, and events.
    """
    session_id  = state["session_id"]
    agent_tasks = state.get("agent_tasks", {})
    queries: list[str] = agent_tasks.get("market_agent", [state["query"]])

    logger.info("Market Agent started", session_id=session_id, queries=len(queries))

    start_event: AgentEvent = {
        "agent":     "market_agent",
        "action":    "started",
        "message":   "Retrieving market and industry context...",
        "timestamp": _now(),
    }

    all_sources: list[SourceDoc] = []
    for q in queries:
        try:
            doc = await _fetch_wikipedia(q)
            if doc:
                all_sources.append(doc)
        except Exception as exc:
            logger.error("Market lookup failed", query=q[:40], error=str(exc))

    done_event: AgentEvent = {
        "agent":     "market_agent",
        "action":    "completed",
        "message":   f"Retrieved {len(all_sources)} market/industry document(s)",
        "timestamp": _now(),
    }

    logger.info("Market Agent completed", docs_found=len(all_sources))

    return {
        "sources":   all_sources,
        "raw_texts": [s["content"] for s in all_sources],
        "events":    [start_event, done_event],
        "progress":  28,
    }


def _now() -> str:
    """Return current UTC timestamp as ISO string."""
    return datetime.now(timezone.utc).isoformat()
