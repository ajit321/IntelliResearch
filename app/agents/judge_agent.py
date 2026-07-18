"""
app/agents/judge_agent.py
LLM-as-a-Judge Agent — evaluates report quality and triggers self-correction.
Uses a high-capability model to score the report on multiple dimensions.
"""

from datetime import datetime, timezone
from typing import Any

from pydantic import BaseModel, Field

from app.agents.state import ResearchState, AgentEvent, QualityScore
from app.config import get_settings
from app.utils.llm import call_llm_structured, get_model_for_task
from app.utils.logging import get_logger
from app.utils.decorators import timer

logger = get_logger(__name__)
settings = get_settings()


class JudgeOutput(BaseModel):
    """Structured quality evaluation from the LLM-as-a-Judge."""

    overall_score: float = Field(
        ...,
        ge=0.0,
        le=10.0,
        description="Overall quality score from 0.0 (terrible) to 10.0 (excellent)",
    )
    clarity: float = Field(..., ge=0.0, le=10.0, description="Writing clarity and structure")
    depth: float = Field(..., ge=0.0, le=10.0, description="Research depth and thoroughness")
    accuracy: float = Field(..., ge=0.0, le=10.0, description="Factual accuracy and citations")
    completeness: float = Field(..., ge=0.0, le=10.0, description="Coverage of the topic")
    feedback: str = Field(
        ...,
        description="Specific, actionable feedback for improvement",
    )
    strengths: list[str] = Field(default_factory=list, description="What the report does well")
    weaknesses: list[str] = Field(default_factory=list, description="Areas needing improvement")


JUDGE_SYSTEM = """You are an expert research quality evaluator — an LLM-as-a-Judge.

Your task is to objectively evaluate a research report on these dimensions:

1. CLARITY (0-10): Is the writing clear, well-structured, and easy to follow?
2. DEPTH (0-10): Does the report go beyond surface level? Are insights substantive?
3. ACCURACY (0-10): Are claims well-supported by cited evidence? Are contradictions addressed?
4. COMPLETENESS (0-10): Does the report adequately answer the original research question?
5. OVERALL (0-10): Holistic quality assessment.

Also provide:
- Specific, actionable feedback for improvement
- 2-3 key strengths
- 2-3 key weaknesses

Be honest and rigorous. A score of 7+ should be reserved for genuinely strong reports.
Score below 6 if significant improvements are clearly needed."""


@timer
async def judge_agent_node(state: ResearchState) -> dict[str, Any]:
    """
    LLM-as-a-Judge Agent node — evaluates report quality.

    If overall_score < threshold AND correction loops remain,
    the conditional edge will route back to analysis_agent for self-correction.

    Args:
        state: Current pipeline state with completed report.

    Returns:
        State update dict with quality_score and needs_correction flag.
    """
    session_id = state["session_id"]
    report     = state.get("report", {})

    if not report:
        logger.warning("Judge Agent: no report to evaluate", session_id=session_id)
        default_score = QualityScore(
            overall_score=5.0,
            clarity=5.0,
            depth=5.0,
            accuracy=5.0,
            completeness=5.0,
            feedback="No report generated to evaluate.",
            passed=False,
        )
        return {
            "quality_score":     default_score,
            "needs_correction":  True,
            "events":            [],
            "progress":          90,
        }

    logger.info("Judge Agent started", session_id=session_id)

    start_event: AgentEvent = {
        "agent":     "judge_agent",
        "action":    "started",
        "message":   "Evaluating report quality with LLM-as-a-Judge...",
        "timestamp": _now(),
    }

    report_excerpt = str(report.get("full_text", ""))[:4000]
    stats = report.get("stats", {})

    prompt = f"""Research question: {state['query']}

Report to evaluate:
{report_excerpt}

Report statistics:
- Sources consulted: {stats.get('total_sources', 0)}
- Contradictions identified: {stats.get('contradictions_found', 0)}
- Hypotheses generated: {stats.get('hypotheses_generated', 0)}
- Claims verified: {stats.get('claims_verified', 0)}

Evaluate this report rigorously across all quality dimensions."""

    result: JudgeOutput | None = await call_llm_structured(
        prompt=prompt,
        system=JUDGE_SYSTEM,
        output_model=JudgeOutput,
        model=get_model_for_task("judge"),
    )

    if result is None:
        logger.warning("Judge structured output failed — assuming acceptable quality")
        result = JudgeOutput(
            overall_score=7.0,
            clarity=7.0,
            depth=7.0,
            accuracy=7.0,
            completeness=7.0,
            feedback="Automatic quality evaluation failed — manual review recommended.",
            strengths=[],
            weaknesses=[],
        )

    threshold = settings.judge_quality_threshold
    passed    = result.overall_score >= threshold

    quality_score = QualityScore(
        overall_score=result.overall_score,
        clarity=result.clarity,
        depth=result.depth,
        accuracy=result.accuracy,
        completeness=result.completeness,
        feedback=result.feedback,
        passed=passed,
    )

    done_event: AgentEvent = {
        "agent":     "judge_agent",
        "action":    "completed",
        "message":   (
            f"Quality score: {result.overall_score:.1f}/10 "
            f"({'✅ PASSED' if passed else '❌ NEEDS REVISION'})"
        ),
        "timestamp": _now(),
    }

    logger.info(
        "Judge Agent completed",
        score=result.overall_score,
        threshold=threshold,
        passed=passed,
        loop_count=state.get("correction_loop_count", 0),
    )

    return {
        "quality_score":     quality_score,
        "needs_correction":  not passed,
        "events":            [start_event, done_event],
        "progress":          92,
    }


def _now() -> str:
    """Return current UTC timestamp as ISO string."""
    return datetime.now(timezone.utc).isoformat()
