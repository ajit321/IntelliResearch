"""Smoke tests for the Streamlit research workspace states."""

from pathlib import Path

from streamlit.testing.v1 import AppTest


APP_PATH = Path(__file__).parents[1] / "frontend" / "streamlit_app.py"


def test_frontend_composer_renders_without_exceptions() -> None:
    """The initial research composer should render cleanly."""
    app = AppTest.from_file(str(APP_PATH)).run(timeout=30)

    assert not app.exception
    assert any(button.label == "Start research →" for button in app.button)


def test_frontend_report_workspace_renders_without_exceptions() -> None:
    """Evidence, report, quality, and review states should render together."""
    app = AppTest.from_file(str(APP_PATH)).run(timeout=30)
    app.session_state["progress"] = 92
    app.session_state["awaiting_hitl"] = True
    app.session_state["sources"] = [{
        "title": "Example source",
        "url": "https://example.com/research",
        "content": "Evidence summary for visual state testing.",
        "source_type": "news",
        "confidence": 0.86,
        "published_date": "2026-01-01",
    }]
    app.session_state["report"] = {
        "query": "Example research question",
        "executive_summary": "A concise evidence-backed executive summary.",
        "full_text": "# Research brief\n\nA verified research narrative.",
        "stats": {
            "total_sources": 1,
            "contradictions_found": 0,
            "hypotheses_generated": 2,
            "claims_verified": 3,
        },
    }
    app.session_state["quality_score"] = {
        "overall_score": 8.4,
        "clarity": 8.5,
        "depth": 8.2,
        "accuracy": 8.6,
        "completeness": 8.3,
        "feedback": "Strong synthesis with clear evidence.",
        "passed": True,
    }
    app.run(timeout=30)

    assert not app.exception
    assert len(app.tabs) == 5
