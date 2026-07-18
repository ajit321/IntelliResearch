"""
app/agents/citation_agent.py
Citation & Validation Agent — verifies claims, assigns verdicts,
and formats citations for the final report.
"""

from datetime import datetime, timezone
from typing import Any

from pydantic import BaseModel, Field

from app.agents.state import ResearchState, AgentEvent, FactCheckResult, Citation
from app.utils.llm import call_llm_structured, get_model_for_task
from app.utils.logging import get_logger
from app.utils.decorators import timer

logger = get_logger(__name__)


class CitationOutput(BaseModel):
    """Structured output from the Citation Agent."""

    fact_checks: list[dict] = Field(
        ...,
        description=(
            "4-6 specific claims fact-checked. Each: "
            "{claim, verdict, confidence, evidence}"
        ),
    )
    credibility_notes: list[str] = Field(
        default_factory=list,
        description="Notes on source credibility and potential biases",
    )


CITATION_SYSTEM = """You are a rigorous fact-checker and citation specialist.

Your task:
1. FACT-CHECK: Extract 4-6 specific, concrete factual claims from the research summaries.
   For each claim:
   - verdict: "verified" | "disputed" | "unverified"
   - confidence: 0.0-1.0
   - evidence: Which source(s) support or refute this claim

2. CREDIBILITY: Note any concerns about source credibility, potential bias,
   or conflicts of interest.

Base ALL verdicts on the provided source evidence. Do not use external knowledge."""


@timer
async def citation_agent_node(state: ResearchState) -> dict[str, Any]:
    """
    Citation & Validation Agent node — verifies claims and formats citations.

    Args:
        state: Current pipeline state with summaries and sources.

    Returns:
        State update dict with fact_checks, citations, and events.
    """
    session_id = state["session_id"]
    sources    = state.get("sources", [])
    summaries  = state.get("summaries", [])

    logger.info(
        "Citation Agent started",
        session_id=session_id,
        sources=len(sources),
    )

    start_event: AgentEvent = {
        "agent":     "citation_agent",
        "action":    "started",
        "message":   f"Verifying claims against {len(sources)} sources...",
        "timestamp": _now(),
    }

    source_titles = [s["title"] for s in sources[:10]]
    summary_text  = "\n\n".join(summaries)

    prompt = f"""Research topic: {state['query']}

Summary to fact-check:
{summary_text[:3000]}

Available sources for verification:
{chr(10).join([f'- {t}' for t in source_titles])}

Fact-check the key claims and note credibility concerns."""

    result: CitationOutput | None = await call_llm_structured(
        prompt=prompt,
        system=CITATION_SYSTEM,
        output_model=CitationOutput,
        model=get_model_for_task("analysis"),
    )

    if result is None:
        result = CitationOutput(
            fact_checks=[{
                "claim": "Main research findings",
                "verdict": "unverified",
                "confidence": 0.5,
                "evidence": "Unable to complete verification.",
            }],
            credibility_notes=[],
        )

    # Map to typed FactCheckResult dicts
    fact_checks: list[FactCheckResult] = [
        FactCheckResult(
            claim=fc.get("claim", ""),
            verdict=fc.get("verdict", "unverified"),
            confidence=float(fc.get("confidence", 0.5)),
            evidence=fc.get("evidence", ""),
        )
        for fc in result.fact_checks
    ]

    # Format citations from sources
    citations: list[Citation] = [
        Citation(
            index=i + 1,
            title=s["title"],
            url=s["url"],
            source_type=s["source_type"],
            authors="",  # Not available from all sources
            published_date=s.get("published_date", ""),
        )
        for i, s in enumerate(sources[:20])
    ]

    verified = sum(1 for f in fact_checks if f["verdict"] == "verified")
    disputed = sum(1 for f in fact_checks if f["verdict"] == "disputed")

    done_event: AgentEvent = {
        "agent":     "citation_agent",
        "action":    "completed",
        "message":   f"{verified} verified, {disputed} disputed of {len(fact_checks)} claims",
        "timestamp": _now(),
    }

    logger.info(
        "Citation Agent completed",
        fact_checks=len(fact_checks),
        verified=verified,
        disputed=disputed,
        citations=len(citations),
    )

    return {
        "fact_checks": fact_checks,
        "citations":   citations,
        "events":      [start_event, done_event],
        "progress":    75,
    }


def _now() -> str:
    """Return current UTC timestamp as ISO string."""
    return datetime.now(timezone.utc).isoformat()
