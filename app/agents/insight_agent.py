"""
app/agents/insight_agent.py
Insight Generation Agent — produces hypotheses, trends, and reasoning chains.
"""

from datetime import datetime, timezone
from typing import Any

from pydantic import BaseModel, Field

from app.agents.state import ResearchState, AgentEvent, Hypothesis
from app.utils.llm import call_llm_structured, get_model_for_task
from app.utils.logging import get_logger
from app.utils.decorators import timer

logger = get_logger(__name__)


class InsightOutput(BaseModel):
    """Structured output from the Insight Generation Agent."""

    hypotheses: list[dict] = Field(
        ...,
        description=(
            "2-3 research hypotheses. Each: "
            "{text, reasoning_chain: [step1, step2, ...], confidence}"
        ),
    )
    trends: list[str] = Field(
        ...,
        description="3-5 emerging trends identified in the research",
    )
    knowledge_gaps: list[str] = Field(
        default_factory=list,
        description="Areas where evidence is weak or missing",
    )


INSIGHT_SYSTEM = """You are a research insight specialist skilled in synthesising
complex information into actionable hypotheses.

Your task:
1. HYPOTHESES: Generate 2-3 novel, testable research hypotheses that the data suggests.
   For each, provide a multi-hop reasoning chain showing how you arrived at it:
   [observation → inference → hypothesis → prediction]

2. TRENDS: Identify 3-5 emerging trends from the evidence.
   Each trend should be specific and evidence-grounded.

3. KNOWLEDGE GAPS: Identify areas where evidence is weak, contradictory, or missing.
   These represent opportunities for further research.

Be analytically rigorous. Ground everything in the provided evidence.
Speculate only where clearly labelled as such."""


@timer
async def insight_agent_node(state: ResearchState) -> dict[str, Any]:
    """
    Insight Agent node — generates hypotheses and trends from analysis.

    Args:
        state: Current pipeline state with summaries and contradictions.

    Returns:
        State update dict with hypotheses, trends, and events.
    """
    session_id    = state["session_id"]
    summaries     = state.get("summaries", [])
    contradictions = state.get("contradictions", [])

    logger.info("Insight Agent started", session_id=session_id)

    start_event: AgentEvent = {
        "agent":     "insight_agent",
        "action":    "started",
        "message":   "Generating hypotheses and identifying trends...",
        "timestamp": _now(),
    }

    summary_text = "\n\n".join(summaries)
    contra_text  = "\n".join([
        f"• {c['claim']}: {c['source_a']} vs {c['source_b']}"
        for c in contradictions
    ])

    source_types = list(set(s["source_type"] for s in state.get("sources", [])))

    prompt = f"""Research topic: {state['query']}

Analysis summaries:
{summary_text[:3000]}

Identified contradictions:
{contra_text or 'None detected'}

Source types consulted: {source_types}

Generate hypotheses, trends, and knowledge gaps from this evidence."""

    result: InsightOutput | None = await call_llm_structured(
        prompt=prompt,
        system=INSIGHT_SYSTEM,
        output_model=InsightOutput,
        model=get_model_for_task("reasoning"),
    )

    if result is None:
        result = InsightOutput(
            hypotheses=[{
                "text": "Further investigation required",
                "reasoning_chain": ["Insufficient evidence for strong hypotheses"],
                "confidence": 0.5,
            }],
            trends=["Ongoing research activity in this domain"],
            knowledge_gaps=["More primary sources needed"],
        )

    # Map to typed Hypothesis dicts
    typed_hypotheses: list[Hypothesis] = [
        Hypothesis(
            text=h.get("text", ""),
            reasoning_chain=h.get("reasoning_chain", []),
            confidence=float(h.get("confidence", 0.75)),
        )
        for h in result.hypotheses
    ]

    done_event: AgentEvent = {
        "agent":     "insight_agent",
        "action":    "completed",
        "message":   f"Generated {len(typed_hypotheses)} hypothesis(es), {len(result.trends)} trends",
        "timestamp": _now(),
    }

    logger.info(
        "Insight Agent completed",
        hypotheses=len(typed_hypotheses),
        trends=len(result.trends),
    )

    return {
        "hypotheses": typed_hypotheses,
        "trends":     result.trends,
        "events":     [start_event, done_event],
        "progress":   65,
    }


def _now() -> str:
    """Return current UTC timestamp as ISO string."""
    return datetime.now(timezone.utc).isoformat()
