"""
app/agents/paper_agent.py
Paper Agent — retrieves academic research papers from arXiv.
Uses circuit breaker for resilience and indexes results into FAISS.
"""

import asyncio
from datetime import datetime, timezone
from typing import Any

import arxiv

from app.agents.state import ResearchState, SourceDoc, AgentEvent
from app.config import get_settings
from app.utils.logging import get_logger
from app.utils.decorators import circuit_breaker, timer

logger = get_logger(__name__)
settings = get_settings()

# Depth → max papers to retrieve
_DEPTH_MAP: dict[str, int] = {
    "quick":  3,
    "deep":   6,
    "expert": 10,
}


@circuit_breaker("arxiv_api", failure_threshold=3, timeout=30)
async def _fetch_arxiv_papers(
    query: str,
    max_results: int,
) -> list[SourceDoc]:
    """
    Fetch papers from arXiv API for a given query.

    Args:
        query: Search query string.
        max_results: Maximum number of papers to retrieve.

    Returns:
        List of SourceDoc dicts with paper metadata and abstract.

    Raises:
        RuntimeError: Propagated from circuit breaker on repeated failure.
    """
    results: list[SourceDoc] = []

    # Run synchronous arxiv client in thread pool to avoid blocking event loop
    search = arxiv.Search(
        query=query,
        max_results=max_results,
        sort_by=arxiv.SortCriterion.Relevance,
    )
    client = arxiv.Client()

    # arxiv.Client is synchronous — run in executor
    loop = asyncio.get_event_loop()
    papers = await loop.run_in_executor(
        None, lambda: list(client.results(search))
    )

    for paper in papers:
        results.append(
            SourceDoc(
                title=paper.title,
                url=paper.entry_id,
                content=paper.summary[:2000],  # Cap at 2000 chars
                source_type="arxiv",
                confidence=0.92,               # Academic papers get high base confidence
                published_date=paper.published.isoformat() if paper.published else "",
            )
        )
        logger.debug(
            "arXiv paper retrieved",
            title=paper.title[:60],
            url=paper.entry_id,
        )

    return results


@timer
async def paper_agent_node(state: ResearchState) -> dict[str, Any]:
    """
    Paper Agent node — retrieves academic papers for all assigned sub-questions.

    Runs queries in parallel using asyncio.gather for efficiency.

    Args:
        state: Current pipeline state.

    Returns:
        State update dict with sources, raw_texts, and events.
    """
    session_id = state["session_id"]
    depth      = state.get("depth", "deep")
    max_papers = _DEPTH_MAP.get(depth, _DEPTH_MAP["deep"])

    # Get tasks assigned to this agent by the planner
    agent_tasks = state.get("agent_tasks", {})
    queries: list[str] = agent_tasks.get("paper_agent", [state["query"]])

    logger.info(
        "Paper Agent started",
        session_id=session_id,
        queries=len(queries),
        max_papers=max_papers,
    )

    start_event: AgentEvent = {
        "agent":     "paper_agent",
        "action":    "started",
        "message":   f"Searching arXiv for {len(queries)} sub-question(s)...",
        "timestamp": _now(),
    }

    # Fetch all queries in parallel
    fetch_tasks = [
        _fetch_arxiv_papers(q, max_results=max(1, max_papers // len(queries)))
        for q in queries
    ]

    results_nested = await asyncio.gather(*fetch_tasks, return_exceptions=True)

    all_sources: list[SourceDoc] = []
    for result in results_nested:
        if isinstance(result, Exception):
            logger.error("arXiv fetch error", error=str(result))
        else:
            all_sources.extend(result)

    # Deduplicate by URL
    seen: set[str] = set()
    unique_sources = [s for s in all_sources if not (s["url"] in seen or seen.add(s["url"]))]  # type: ignore[func-returns-value]

    done_event: AgentEvent = {
        "agent":     "paper_agent",
        "action":    "completed",
        "message":   f"Retrieved {len(unique_sources)} academic papers from arXiv",
        "timestamp": _now(),
    }

    logger.info(
        "Paper Agent completed",
        papers_found=len(unique_sources),
        session_id=session_id,
    )

    return {
        "sources":   unique_sources,
        "raw_texts": [s["content"] for s in unique_sources],
        "events":    [start_event, done_event],
        "progress":  25,
    }


def _now() -> str:
    """Return current UTC timestamp as ISO string."""
    return datetime.now(timezone.utc).isoformat()
