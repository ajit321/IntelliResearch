"""
tests/test_agents.py
Unit tests for agent nodes using mocked LLM calls.
"""

import pytest
from unittest.mock import patch

from app.agents.state import ResearchState
from app.config import get_settings


def _make_state(**overrides) -> ResearchState:
    """Build a minimal ResearchState for testing."""
    base: ResearchState = {
        "query":                "Test query about AI",
        "depth":                "quick",
        "session_id":           "test-session-123",
        "user_id":              "user-test",
        "research_plan":        [],
        "agent_tasks":          {},
        "sources":              [],
        "raw_texts":            [],
        "summaries":            [],
        "contradictions":       [],
        "hypotheses":           [],
        "trends":               [],
        "fact_checks":          [],
        "citations":            [],
        "report":               None,
        "quality_score":        None,
        "correction_loop_count": 0,
        "needs_correction":     False,
        "hitl_approved":        False,
        "hitl_feedback":        "",
        "events":               [],
        "progress":             0,
        "error":                None,
        "cache_hit":            False,
    }
    base.update(overrides)
    return base


def test_all_agent_tasks_use_configured_model() -> None:
    """Task routing must not silently select retired provider model IDs."""
    from app.utils.llm import get_model_for_task

    configured_model = get_settings().llm_model
    for task_type in ("retrieval", "analysis", "reasoning", "judge", "unknown"):
        assert get_model_for_task(task_type) == configured_model


class TestPlannerNode:
    """Tests for the Planner Agent node."""

    @pytest.mark.asyncio
    async def test_planner_returns_plan(self) -> None:
        """Planner should return research_plan and agent_tasks."""
        from app.agents.planner import planner_node

        with patch("app.agents.planner.call_llm_structured") as mock_llm:
            from app.agents.planner import ResearchPlan
            mock_llm.return_value = ResearchPlan(
                sub_questions=["What is AI?", "What are the trends?"],
                agent_assignments={
                    "paper_agent": ["What is AI?"],
                    "news_agent": ["What are the trends?"],
                    "market_agent": [],
                    "user_docs_agent": [],
                },
                reasoning="Test reasoning",
                estimated_depth="quick",
            )

            state = _make_state()
            result = await planner_node(state)

        assert "research_plan" in result
        assert len(result["research_plan"]) == 2
        assert "agent_tasks" in result
        assert result["progress"] == 10

    @pytest.mark.asyncio
    async def test_planner_fallback_on_llm_failure(self) -> None:
        """Planner should use fallback plan if LLM returns None."""
        from app.agents.planner import planner_node

        with patch("app.agents.planner.call_llm_structured", return_value=None):
            state  = _make_state()
            result = await planner_node(state)

        assert "research_plan" in result
        assert len(result["research_plan"]) >= 1  # Fallback uses original query


class TestCitationAgent:
    """Tests for the Citation Agent node."""

    @pytest.mark.asyncio
    async def test_citation_agent_formats_citations(self) -> None:
        """Citation agent should produce formatted citations from sources."""
        from app.agents.citation_agent import citation_agent_node
        from app.agents.citation_agent import CitationOutput

        mock_output = CitationOutput(
            fact_checks=[{
                "claim": "AI is transforming research",
                "verdict": "verified",
                "confidence": 0.90,
                "evidence": "Multiple papers confirm this",
            }],
            credibility_notes=[],
        )

        with patch("app.agents.citation_agent.call_llm_structured", return_value=mock_output):
            state = _make_state(
                summaries=["AI is transforming research rapidly."],
                sources=[{
                    "title": "AI Research 2024",
                    "url": "https://arxiv.org/1234",
                    "content": "...",
                    "source_type": "arxiv",
                    "confidence": 0.92,
                    "published_date": "2024-01-01",
                }],
            )
            result = await citation_agent_node(state)

        assert "fact_checks" in result
        assert len(result["fact_checks"]) == 1
        assert result["fact_checks"][0]["verdict"] == "verified"
        assert "citations" in result
        assert len(result["citations"]) == 1


class TestJudgeAgent:
    """Tests for the LLM-as-a-Judge Agent."""

    @pytest.mark.asyncio
    async def test_judge_passes_high_quality_report(self) -> None:
        """Judge should pass a high-quality report."""
        from app.agents.judge_agent import judge_agent_node, JudgeOutput

        mock_output = JudgeOutput(
            overall_score=8.5,
            clarity=8.0,
            depth=9.0,
            accuracy=8.5,
            completeness=8.5,
            feedback="Excellent report with strong evidence.",
            strengths=["Well cited", "Clear structure"],
            weaknesses=[],
        )

        with patch("app.agents.judge_agent.call_llm_structured", return_value=mock_output):
            state = _make_state(
                report={
                    "full_text": "Comprehensive research report...",
                    "stats": {"total_sources": 5},
                }
            )
            result = await judge_agent_node(state)

        assert result["quality_score"]["passed"] is True
        assert result["quality_score"]["overall_score"] == 8.5
        assert result["needs_correction"] is False

    @pytest.mark.asyncio
    async def test_judge_fails_low_quality_report(self) -> None:
        """Judge should flag a low-quality report for self-correction."""
        from app.agents.judge_agent import judge_agent_node, JudgeOutput

        mock_output = JudgeOutput(
            overall_score=4.0,
            clarity=4.0,
            depth=3.5,
            accuracy=4.5,
            completeness=4.0,
            feedback="Report lacks depth and proper citations.",
            strengths=[],
            weaknesses=["Poor citation", "Shallow analysis"],
        )

        with patch("app.agents.judge_agent.call_llm_structured", return_value=mock_output):
            state = _make_state(
                report={"full_text": "Thin report...", "stats": {}},
            )
            result = await judge_agent_node(state)

        assert result["quality_score"]["passed"] is False
        assert result["needs_correction"] is True
