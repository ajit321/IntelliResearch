"""
app/agents/news_agent.py
News Agent — retrieves recent web and news articles via SerpAPI.
Falls back to mock data if SerpAPI key is not configured.
"""

from datetime import datetime, timezone
from typing import Any

import httpx

from app.agents.state import ResearchState, SourceDoc, AgentEvent
from app.config import get_settings
from app.utils.logging import get_logger
from app.utils.decorators import circuit_breaker, timer

logger = get_logger(__name__)
settings = get_settings()


@circuit_breaker("serpapi", failure_threshold=3, timeout=60)
async def _fetch_web_results(query: str, max_results: int = 5) -> list[SourceDoc]:
    """
    Fetch web search results via SerpAPI.

    Falls back to mock data if SERPAPI_KEY is not set — enabling demo mode.

    Args:
        query: Search query string.
        max_results: Maximum results to return.

    Returns:
        List of SourceDoc from web search results.
    """
    if not settings.serpapi_key:
        logger.warning("SerpAPI key missing — using mock news data for demo")
        return _mock_news_results(query)

    async with httpx.AsyncClient(timeout=20.0) as client:
        response = await client.get(
            "https://serpapi.com/search",
            params={
                "q":       query,
                "api_key": settings.serpapi_key,
                "num":     max_results,
                "output":  "json",
                "tbm":     "nws",  # News search
            },
        )
        response.raise_for_status()
        data = response.json()

    results: list[SourceDoc] = []
    for item in data.get("news_results", [])[:max_results]:
        results.append(
            SourceDoc(
                title=item.get("title", ""),
                url=item.get("link", ""),
                content=item.get("snippet", "")[:1000],
                source_type="news",
                confidence=0.72,  # News gets slightly lower base confidence
                published_date=item.get("date", ""),
            )
        )

    logger.info("SerpAPI news results fetched", count=len(results), query=query[:40])
    return results


def _mock_news_results(query: str) -> list[SourceDoc]:
    """
    Return mock news results for demo/test mode.

    Args:
        query: Original search query (used in mock content).

    Returns:
        List of fake SourceDoc entries for demo purposes.
    """
    return [
        SourceDoc(
            title=f"Latest developments in: {query[:50]}",
            url="https://example.com/news/demo",
            content=(
                f"[DEMO MODE] This is a simulated news article about '{query}'. "
                "In production with a SerpAPI key, real news articles would appear here. "
                "Recent analysis suggests significant developments in this area, "
                "with experts from leading institutions weighing in on the implications."
            ),
            source_type="news",
            confidence=0.60,
            published_date=datetime.now(timezone.utc).isoformat(),
        )
    ]


@timer
async def news_agent_node(state: ResearchState) -> dict[str, Any]:
    """
    News Agent node — retrieves current news and web articles.

    Args:
        state: Current pipeline state.

    Returns:
        State update dict with sources, raw_texts, and events.
    """
    session_id  = state["session_id"]
    agent_tasks = state.get("agent_tasks", {})
    queries: list[str] = agent_tasks.get("news_agent", [state["query"]])

    _DEPTH_MAX = {"quick": 3, "deep": 5, "expert": 8}
    max_results = _DEPTH_MAX.get(state.get("depth", "deep"), 5)

    logger.info("News Agent started", session_id=session_id, queries=len(queries))

    start_event: AgentEvent = {
        "agent":     "news_agent",
        "action":    "started",
        "message":   f"Searching news sources for {len(queries)} query(ies)...",
        "timestamp": _now(),
    }

    all_sources: list[SourceDoc] = []
    for q in queries:
        try:
            results = await _fetch_web_results(q, max_results=max(1, max_results // len(queries)))
            all_sources.extend(results)
        except Exception as exc:
            logger.error("News fetch failed", query=q[:40], error=str(exc))

    done_event: AgentEvent = {
        "agent":     "news_agent",
        "action":    "completed",
        "message":   f"Retrieved {len(all_sources)} news articles",
        "timestamp": _now(),
    }

    logger.info("News Agent completed", articles_found=len(all_sources))

    return {
        "sources":   all_sources,
        "raw_texts": [s["content"] for s in all_sources],
        "events":    [start_event, done_event],
        "progress":  28,
    }


def _now() -> str:
    """Return current UTC timestamp as ISO string."""
    return datetime.now(timezone.utc).isoformat()
