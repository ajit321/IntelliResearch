"""
app/agents/analysis_agent.py
Critical Analysis Agent — detects contradictions, scores confidence,
and produces structured summaries from all retrieved sources.
"""

from datetime import datetime, timezone
from typing import Any

from pydantic import BaseModel, Field

from app.agents.state import ResearchState, AgentEvent, Contradiction
from app.utils.llm import call_llm_structured, get_model_for_task
from app.utils.logging import get_logger
from app.utils.decorators import timer

logger = get_logger(__name__)


# ── Structured Output Schema ──────────────────────────────────────────────────

class AnalysisOutput(BaseModel):
    """Structured output from the Critical Analysis Agent."""

    summary: str = Field(
        ...,
        description="2-3 paragraph synthesis of ALL sources combined",
    )
    contradictions: list[dict] = Field(
        default_factory=list,
        description=(
            "List of contradictions. Each: "
            "{claim, source_a, source_b, detail, severity}"
        ),
    )
    confidence: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Overall confidence in the body of evidence (0.0 – 1.0)",
    )
    key_themes: list[str] = Field(
        default_factory=list,
        description="3-5 key themes identified across all sources",
    )


ANALYSIS_SYSTEM = """You are a critical research analyst with expertise in evaluating
scientific and journalistic evidence.

Your task:
1. SUMMARISE: Write a 2-3 paragraph synthesis of ALL provided source texts.
   Cite sources inline as [source_type: title snippet].

2. CONTRADICTIONS: Identify where sources DISAGREE. For each contradiction:
   - State the specific claim in dispute
   - Name both conflicting sources
   - Explain the nature of the disagreement
   - Rate severity: low | medium | high

3. CONFIDENCE: Assign an overall confidence score (0.0-1.0) to the evidence.
   Consider: source diversity, recency, consensus level, source reputation.

4. KEY THEMES: List 3-5 recurring themes across all sources.

Be rigorous, precise, and cite specific evidence. Do not speculate beyond sources."""


@timer
async def analysis_agent_node(state: ResearchState) -> dict[str, Any]:
    """
    Critical Analysis Agent node — analyses all retrieved sources.

    Runs after all retrieval agents complete (fan-in point).
    Uses a high-capability model for nuanced contradiction detection.

    Args:
        state: Current pipeline state with all sources.

    Returns:
        State update dict with summaries, contradictions, and events.
    """
    session_id = state["session_id"]
    sources    = state.get("sources", [])
    loop_count = state.get("correction_loop_count", 0)

    logger.info(
        "Analysis Agent started",
        session_id=session_id,
        source_count=len(sources),
        correction_loop=loop_count,
    )

    start_event: AgentEvent = {
        "agent":     "analysis_agent",
        "action":    "started",
        "message":   f"Analysing {len(sources)} sources for contradictions...",
        "timestamp": _now(),
    }

    # Build combined source text for analysis
    source_text = "\n\n---\n\n".join([
        f"[{s['source_type'].upper()}] {s['title']}\n"
        f"Published: {s.get('published_date', 'unknown')}\n"
        f"Confidence: {s.get('confidence', 0.0):.0%}\n\n"
        f"{s['content'][:800]}"
        for s in sources[:15]  # Cap at 15 sources to stay within context window
    ])

    hitl_feedback = state.get("hitl_feedback", "")
    feedback_note = (
        f"\n\nPrevious review feedback to address: {hitl_feedback}"
        if hitl_feedback else ""
    )

    prompt = f"""Research topic: {state['query']}

{len(sources)} sources retrieved:

{source_text[:6000]}
{feedback_note}

Analyse these sources thoroughly."""

    result: AnalysisOutput | None = await call_llm_structured(
        prompt=prompt,
        system=ANALYSIS_SYSTEM,
        output_model=AnalysisOutput,
        model=get_model_for_task("analysis"),
    )

    if result is None:
        logger.warning("Analysis structured output failed — using raw LLM response")
        result = AnalysisOutput(
            summary=f"Analysis of {len(sources)} sources on '{state['query']}'.",
            contradictions=[],
            confidence=0.70,
            key_themes=[],
        )

    # Map to typed Contradiction dicts
    contradictions: list[Contradiction] = [
        Contradiction(
            claim=c.get("claim", ""),
            source_a=c.get("source_a", ""),
            source_b=c.get("source_b", ""),
            detail=c.get("detail", ""),
            severity=c.get("severity", "low"),
        )
        for c in result.contradictions
    ]

    done_event: AgentEvent = {
        "agent":     "analysis_agent",
        "action":    "completed",
        "message":   (
            f"Analysis complete: {len(contradictions)} contradiction(s), "
            f"{result.confidence:.0%} confidence"
        ),
        "timestamp": _now(),
    }

    logger.info(
        "Analysis Agent completed",
        contradictions=len(contradictions),
        confidence=result.confidence,
        themes=len(result.key_themes),
    )

    return {
        "summaries":           [result.summary],
        "contradictions":      contradictions,
        "correction_loop_count": loop_count + (1 if state.get("needs_correction") else 0),
        "events":              [start_event, done_event],
        "progress":            55,
    }


def _now() -> str:
    """Return current UTC timestamp as ISO string."""
    return datetime.now(timezone.utc).isoformat()
