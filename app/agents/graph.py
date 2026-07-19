"""
app/agents/graph.py
LangGraph StateGraph definition for IntelliResearch.

Implements:
- All agent nodes wired with typed edges
- Parallel fan-out for Paper/News/Market/UserDocs agents
- Conditional edges for self-correction loop
- Human-in-the-loop interrupt before report delivery
- MemorySaver for checkpoint persistence and time travel
"""

from typing import Any, Awaitable, Callable

from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver

from app.agents.state import ResearchState
from app.agents.planner import planner_node
from app.agents.paper_agent import paper_agent_node
from app.agents.news_agent import news_agent_node
from app.agents.market_agent import market_agent_node
from app.agents.user_docs_agent import user_docs_agent_node
from app.agents.analysis_agent import analysis_agent_node
from app.agents.insight_agent import insight_agent_node
from app.agents.citation_agent import citation_agent_node
from app.agents.report_builder import report_builder_node
from app.agents.judge_agent import judge_agent_node
from app.config import get_settings
from app.utils.logging import get_logger

logger = get_logger(__name__)
settings = get_settings()


# ── Conditional Edge Functions ────────────────────────────────────────────────

def should_self_correct(state: ResearchState) -> str:
    """
    Conditional edge: decide whether to re-run analysis or proceed to HITL.

    Routes to "analysis_agent" if the judge score is below threshold
    AND we haven't exceeded the max correction loop count.
    Otherwise routes to "hitl_review" for human approval.

    Args:
        state: Current pipeline state.

    Returns:
        Next node name: "analysis_agent" or "hitl_review".
    """
    score = state.get("quality_score")
    loop_count = state.get("correction_loop_count", 0)
    max_loops = settings.max_self_correction_loops

    if (
        score is not None
        and not score.get("passed", True)
        and loop_count < max_loops
    ):
        logger.info(
            "Quality below threshold — triggering self-correction",
            score=score.get("overall_score"),
            loop=loop_count,
            max_loops=max_loops,
        )
        return "analysis_agent"

    logger.info(
        "Quality check passed — proceeding to HITL review",
        score=score.get("overall_score") if score else "N/A",
    )
    return "hitl_review"


def after_hitl(state: ResearchState) -> str:
    """
    Conditional edge: after human review, either finalize or re-run.

    Args:
        state: Current pipeline state.

    Returns:
        Next node name: "report_builder" (re-run) or END.
    """
    if state.get("hitl_approved", True):
        return END
    # Human rejected — rebuild report with their feedback
    logger.info("HITL rejected — rebuilding report", feedback=state.get("hitl_feedback"))
    return "report_builder"


# ── Graph Builder ─────────────────────────────────────────────────────────────

def build_graph() -> StateGraph:
    """
    Build and compile the full multi-agent research StateGraph.

    Graph topology:
        planner
            ├── paper_agent    ┐
            ├── news_agent     ├── parallel fan-out
            ├── market_agent   │
            └── user_docs      ┘
                └── analysis_agent
                    └── insight_agent
                        └── citation_agent
                            └── report_builder
                                └── judge_agent
                                    ├── (score < threshold) → analysis_agent (loop)
                                    └── (score OK) → hitl_review
                                                        ├── (approved) → END
                                                        └── (rejected) → report_builder

    Returns:
        Compiled LangGraph application with MemorySaver checkpoints.
    """
    graph = StateGraph(ResearchState)

    # ── Register all nodes ────────────────────────────────────
    graph.add_node("planner",          planner_node)
    graph.add_node("paper_agent",      paper_agent_node)
    graph.add_node("news_agent",       news_agent_node)
    graph.add_node("market_agent",     market_agent_node)
    graph.add_node("user_docs_agent",  user_docs_agent_node)
    graph.add_node("analysis_agent",   analysis_agent_node)
    graph.add_node("insight_agent",    insight_agent_node)
    graph.add_node("citation_agent",   citation_agent_node)
    graph.add_node("report_builder",   report_builder_node)
    graph.add_node("judge_agent",      judge_agent_node)

    # HITL node: interrupt_before causes LangGraph to pause here
    # and wait for human input before continuing.
    graph.add_node("hitl_review", _hitl_passthrough_node)

    # ── Entry point ───────────────────────────────────────────
    graph.set_entry_point("planner")

    # ── Planner → parallel retrieval fan-out ─────────────────
    # All four retrieval agents run concurrently
    graph.add_edge("planner",      "paper_agent")
    graph.add_edge("planner",      "news_agent")
    graph.add_edge("planner",      "market_agent")
    graph.add_edge("planner",      "user_docs_agent")

    # ── Retrieval → Analysis (all four must complete first) ──
    graph.add_edge(
        ["paper_agent", "news_agent", "market_agent", "user_docs_agent"],
        "analysis_agent",
    )

    # ── Linear analysis pipeline ──────────────────────────────
    graph.add_edge("analysis_agent",  "insight_agent")
    graph.add_edge("insight_agent",   "citation_agent")
    graph.add_edge("citation_agent",  "report_builder")
    graph.add_edge("report_builder",  "judge_agent")

    # ── Conditional: self-correction or HITL ─────────────────
    graph.add_conditional_edges(
        "judge_agent",
        should_self_correct,
        {
            "analysis_agent": "analysis_agent",
            "hitl_review":    "hitl_review",
        },
    )

    # ── Conditional: HITL approval or re-build ───────────────
    graph.add_conditional_edges(
        "hitl_review",
        after_hitl,
        {
            "report_builder": "report_builder",
            END: END,
        },
    )

    return graph


async def _hitl_passthrough_node(state: ResearchState) -> dict:
    """
    HITL passthrough node — pauses the graph here.

    LangGraph's interrupt_before mechanism stops execution at this node.
    The graph resumes when the API receives human approval/feedback.

    Args:
        state: Current pipeline state.

    Returns:
        Unchanged state dict (human input is injected externally).
    """
    logger.info(
        "Graph paused at HITL review",
        session_id=state.get("session_id"),
        approved=state.get("hitl_approved"),
    )
    return {}


# ── Compiled Graph Singleton ──────────────────────────────────────────────────

# MemorySaver persists checkpoints in-memory for the hackathon.
# In production, swap with SqliteSaver or PostgresSaver for durability.
_memory_saver = MemorySaver()

# interrupt_before=["hitl_review"] pauses the graph at the HITL node
research_graph = build_graph().compile(
    checkpointer=_memory_saver,
    interrupt_before=["hitl_review"],
)


async def run_research(
    session_id: str,
    query: str,
    depth: str = "deep",
    user_id: str = "anonymous",
    uploaded_docs: list[str] | None = None,
    on_update: Callable[[dict[str, Any]], Awaitable[None]] | None = None,
) -> dict:
    """
    Execute the full multi-agent research pipeline.

    Args:
        session_id: Unique session identifier for checkpoint keying.
        query: The user's research query.
        depth: Research depth — quick | deep | expert.
        user_id: Authenticated user ID from Clerk.
        uploaded_docs: List of raw text from user-uploaded documents.
        on_update: Optional async callback invoked for each graph node update.

    Returns:
        Final pipeline state dict containing the completed report.
    """
    initial_state: ResearchState = {
        "query": query,
        "depth": depth,
        "session_id": session_id,
        "user_id": user_id,
        "research_plan": [],
        "agent_tasks": {},
        "sources": [],
        "raw_texts": uploaded_docs or [],
        "summaries": [],
        "contradictions": [],
        "hypotheses": [],
        "trends": [],
        "fact_checks": [],
        "citations": [],
        "report": None,
        "quality_score": None,
        "correction_loop_count": 0,
        "needs_correction": False,
        "hitl_approved": False,
        "hitl_feedback": "",
        "events": [],
        "progress": 0,
        "error": None,
        "cache_hit": False,
    }

    config = {"configurable": {"thread_id": session_id}}

    logger.info(
        "Research pipeline started",
        session_id=session_id,
        query=query[:60],
        depth=depth,
        user_id=user_id,
    )

    async for update in research_graph.astream(
        initial_state,
        config=config,
        stream_mode="updates",
    ):
        if on_update is not None:
            await on_update(update)

    snapshot = await research_graph.aget_state(config)
    return dict(snapshot.values)


async def resume_after_hitl(
    session_id: str,
    approved: bool,
    feedback: str = "",
) -> dict:
    """
    Resume the graph after human-in-the-loop review.

    Args:
        session_id: Session to resume.
        approved: True if human approved the report.
        feedback: Optional reviewer feedback for re-generation.

    Returns:
        Final pipeline state after HITL resolution.
    """
    config = {"configurable": {"thread_id": session_id}}

    # Update state with human decision
    await research_graph.aupdate_state(
        config,
        {"hitl_approved": approved, "hitl_feedback": feedback},
    )

    # Resume execution from the HITL node
    final_state = await research_graph.ainvoke(None, config=config)
    logger.info(
        "Graph resumed after HITL",
        session_id=session_id,
        approved=approved,
    )
    return final_state
