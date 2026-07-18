"""
app/agents/planner.py
Planner Agent — understands the user goal, breaks it into sub-questions,
and creates an execution plan assigning tasks to specialist agents.
"""

import json
from typing import Any

from pydantic import BaseModel, Field

from app.agents.state import ResearchState, AgentEvent
from app.utils.llm import call_llm_structured, get_model_for_task
from app.utils.logging import get_logger
from app.utils.decorators import timer

logger = get_logger(__name__)


# ── Structured Output Schema ──────────────────────────────────────────────────

class ResearchPlan(BaseModel):
    """Structured research plan output from the Planner Agent."""

    sub_questions: list[str] = Field(
        ...,
        description="3-5 focused sub-questions that decompose the research query",
    )
    agent_assignments: dict[str, list[str]] = Field(
        ...,
        description=(
            "Map of agent name to list of sub-questions assigned to it. "
            "Keys: paper_agent, news_agent, market_agent, user_docs_agent"
        ),
    )
    reasoning: str = Field(
        ...,
        description="Brief explanation of the research strategy",
    )
    estimated_depth: str = Field(
        ...,
        description="Recommended research depth: quick | deep | expert",
    )


PLANNER_SYSTEM = """You are a senior research strategist and planning agent.

Your job is to:
1. Understand the user's research goal deeply
2. Decompose it into 3-5 focused sub-questions
3. Assign each sub-question to the most appropriate specialist agent:
   - paper_agent: academic papers, scientific literature, arXiv
   - news_agent: current events, news articles, recent developments
   - market_agent: market data, industry reports, financial trends
   - user_docs_agent: user-uploaded documents and personal knowledge base

Be specific and actionable. Sub-questions should be narrow enough for
efficient retrieval but broad enough to capture the full picture."""


@timer
async def planner_node(state: ResearchState) -> dict[str, Any]:
    """
    Planner Agent node — creates a research plan from the user query.

    Emits events for real-time frontend streaming.
    Uses structured output to ensure reliable JSON parsing.

    Args:
        state: Current pipeline state with query and depth.

    Returns:
        State update dict with research_plan, agent_tasks, and events.
    """
    session_id = state["session_id"]
    query      = state["query"]
    depth      = state.get("depth", "deep")

    logger.info("Planner Agent started", session_id=session_id, query=query[:60])

    event: AgentEvent = {
        "agent":     "planner",
        "action":    "started",
        "message":   f"Planning research strategy for: {query[:60]}...",
        "timestamp": _now(),
    }

    prompt = f"""Research query: {query}
Requested depth: {depth}

Create a structured research plan with sub-questions and agent assignments.
Consider what types of sources would be most valuable for this topic."""

    plan: ResearchPlan | None = await call_llm_structured(
        prompt=prompt,
        system=PLANNER_SYSTEM,
        output_model=ResearchPlan,
        model=get_model_for_task("analysis"),
    )

    if plan is None:
        # Graceful fallback if structured parsing fails
        logger.warning("Planner structured output failed — using fallback plan")
        plan = ResearchPlan(
            sub_questions=[query],
            agent_assignments={
                "paper_agent":     [query],
                "news_agent":      [query],
                "market_agent":    [query],
                "user_docs_agent": [query],
            },
            reasoning="Fallback: querying all agents with the original query.",
            estimated_depth=depth,
        )

    done_event: AgentEvent = {
        "agent":     "planner",
        "action":    "completed",
        "message":   f"Research plan ready: {len(plan.sub_questions)} sub-questions",
        "timestamp": _now(),
    }

    logger.info(
        "Planner Agent completed",
        sub_questions=len(plan.sub_questions),
        agents_assigned=list(plan.agent_assignments.keys()),
    )

    return {
        "research_plan": plan.sub_questions,
        "agent_tasks":   plan.agent_assignments,
        "events":        [event, done_event],
        "progress":      10,
    }


def _now() -> str:
    """Return current UTC timestamp as ISO string."""
    from datetime import datetime, timezone
    return datetime.now(timezone.utc).isoformat()
