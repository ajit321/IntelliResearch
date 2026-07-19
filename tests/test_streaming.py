"""Regression tests for research progress streaming."""

from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest
from fastapi.testclient import TestClient

from app.main import app


@pytest.mark.asyncio
async def test_graph_runner_forwards_node_updates() -> None:
    """The graph runner should invoke its callback for every streamed update."""
    from app.agents.graph import run_research

    updates = [
        {"planner": {"events": [{"agent": "planner"}], "progress": 10}},
        {"report_builder": {"report": {"full_text": "Draft"}, "progress": 88}},
    ]

    class FakeGraph:
        async def astream(self, *_args, **_kwargs):
            for update in updates:
                yield update

        async def aget_state(self, _config):
            return SimpleNamespace(values={"report": {"full_text": "Draft"}})

    received: list[dict] = []

    async def receive(update: dict) -> None:
        received.append(update)

    with patch("app.agents.graph.research_graph", new=FakeGraph()):
        result = await run_research(
            session_id="stream-test",
            query="Test streaming",
            depth="quick",
            on_update=receive,
        )

    assert received == updates
    assert result["report"]["full_text"] == "Draft"


def test_api_streams_progress_and_hitl_terminal_message() -> None:
    """Starting a run should produce progress followed by a terminal draft."""
    update = {
        "planner": {
            "events": [{
                "agent": "planner",
                "action": "completed",
                "message": "Research plan ready",
                "timestamp": "2026-01-01T00:00:00Z",
            }],
            "progress": 10,
        }
    }
    final_values = {
        "report": {"full_text": "Completed draft"},
        "quality_score": {"overall_score": 8.0, "passed": True},
        "sources": [],
        "progress": 92,
    }

    async def fake_run_research(**kwargs):
        await kwargs["on_update"](update)
        return final_values

    snapshot = SimpleNamespace(values=final_values, next=("hitl_review",))

    with (
        patch("app.main.run_research", new=fake_run_research),
        patch("app.main.research_graph.aget_state", new=AsyncMock(return_value=snapshot)),
        TestClient(app) as client,
    ):
        create_response = client.post(
            "/api/v1/research",
            json={"query": "Benefits of E20 fuel", "depth": "quick"},
        )
        assert create_response.status_code == 201
        session_id = create_response.json()["session_id"]

        with client.websocket_connect(f"/api/v1/ws/{session_id}") as websocket:
            run_response = client.post(
                f"/api/v1/research/{session_id}/run",
                data={"query": "Benefits of E20 fuel", "depth": "quick"},
            )
            assert run_response.status_code == 202

            progress_message = websocket.receive_json()
            terminal_message = websocket.receive_json()

    assert progress_message["type"] == "state_update"
    assert progress_message["node"] == "planner"
    assert progress_message["progress"] == 10
    assert terminal_message["type"] == "hitl_required"
    assert terminal_message["report"]["full_text"] == "Completed draft"
