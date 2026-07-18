"""
app/agents/report_builder.py
Report Builder Agent — compiles all agent outputs into a structured research report.
"""

from datetime import datetime, timezone
from typing import Any

from app.agents.state import ResearchState, AgentEvent
from app.utils.llm import call_llm, get_model_for_task
from app.utils.logging import get_logger
from app.utils.decorators import timer

logger = get_logger(__name__)

REPORT_SYSTEM = """You are a professional research report writer.
Compile the provided data into a polished, structured executive research report.

Format the report with these sections (use Markdown):
## Executive Summary
## Key Findings
## Contradictions & Disputes
## Emerging Hypotheses & Trends  
## Methodology & Sources
## Conclusion
## References

Rules:
- Cite sources inline as [1], [2], etc.
- Be analytical, precise, and objective
- Highlight key statistics and evidence
- Note any important caveats or limitations
- Target: 700-1000 words of high-quality prose"""


@timer
async def report_builder_node(state: ResearchState) -> dict[str, Any]:
    """
    Report Builder Agent node — assembles the final research report.

    Incorporates HITL feedback if a previous report was rejected.

    Args:
        state: Complete pipeline state from all previous agents.

    Returns:
        State update dict with the final report object.
    """
    session_id    = state["session_id"]
    sources       = state.get("sources", [])
    summaries     = state.get("summaries", [])
    contradictions = state.get("contradictions", [])
    hypotheses    = state.get("hypotheses", [])
    trends        = state.get("trends", [])
    fact_checks   = state.get("fact_checks", [])
    citations     = state.get("citations", [])
    hitl_feedback = state.get("hitl_feedback", "")

    logger.info(
        "Report Builder started",
        session_id=session_id,
        sources=len(sources),
        contradictions=len(contradictions),
    )

    start_event: AgentEvent = {
        "agent":     "report_builder",
        "action":    "started",
        "message":   "Compiling all findings into structured report...",
        "timestamp": _now(),
    }

    citations_text = "\n".join([
        f"[{c['index']}] {c['title']} ({c['source_type']}) — {c['url']}"
        for c in citations[:15]
    ])

    feedback_note = (
        f"\n\nREVISION REQUIRED — Human reviewer feedback:\n{hitl_feedback}\n"
        "Please address all feedback points in this revised report."
        if hitl_feedback else ""
    )

    prompt = f"""Research Query: {state['query']}

=== ANALYSIS SUMMARIES ===
{chr(10).join(summaries)[:2500]}

=== CONTRADICTIONS ===
{chr(10).join([f"• [{c['severity'].upper()}] {c['claim']}: {c['source_a']} vs {c['source_b']}" for c in contradictions]) or 'None detected'}

=== TRENDS ===
{chr(10).join([f'• {t}' for t in trends]) or 'None identified'}

=== HYPOTHESES ===
{chr(10).join([f'• {h["text"]}' for h in hypotheses]) or 'None generated'}

=== FACT-CHECK RESULTS ===
{chr(10).join([f'• [{f["verdict"].upper()}] {f["claim"]}' for f in fact_checks]) or 'Not performed'}

=== CITATIONS ===
{citations_text or 'No citations available'}
{feedback_note}

Write a comprehensive, professional research report."""

    report_text = await call_llm(
        prompt=prompt,
        system=REPORT_SYSTEM,
        model=get_model_for_task("analysis"),
    )

    # Build structured report object for frontend consumption
    report: dict[str, Any] = {
        "query":            state["query"],
        "generated_at":     _now(),
        "session_id":       session_id,
        "full_text":        report_text,
        "executive_summary": summaries[0][:600] if summaries else "",
        "sources":          sources,
        "contradictions":   contradictions,
        "hypotheses":       [
            {
                "text":           h["text"],
                "reasoning_chain": h["reasoning_chain"],
                "confidence":     h["confidence"],
            }
            for h in hypotheses
        ],
        "trends":      trends,
        "fact_checks": fact_checks,
        "citations":   citations,
        "stats": {
            "total_sources":        len(sources),
            "contradictions_found": len(contradictions),
            "hypotheses_generated": len(hypotheses),
            "claims_verified":      sum(1 for f in fact_checks if f["verdict"] == "verified"),
            "claims_disputed":      sum(1 for f in fact_checks if f["verdict"] == "disputed"),
        },
    }

    done_event: AgentEvent = {
        "agent":     "report_builder",
        "action":    "completed",
        "message":   "Research report compiled successfully",
        "timestamp": _now(),
    }

    logger.info(
        "Report Builder completed",
        report_chars=len(report_text),
        session_id=session_id,
    )

    return {
        "report": report,
        "events": [start_event, done_event],
        "progress": 88,
    }


def _now() -> str:
    """Return current UTC timestamp as ISO string."""
    return datetime.now(timezone.utc).isoformat()
