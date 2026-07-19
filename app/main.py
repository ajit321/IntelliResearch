"""
app/main.py
IntelliResearch FastAPI application.
Provides REST API for research sessions, HITL approvals, and WebSocket streaming.
"""

import asyncio
import uuid
from contextlib import asynccontextmanager
from typing import Annotated, Any

from fastapi import FastAPI, File, Form, HTTPException, UploadFile, WebSocket, WebSocketDisconnect, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address

from app.auth.clerk import CurrentUser
from app.agents.graph import run_research, resume_after_hitl, research_graph
from app.config import get_settings
from app.observability.langsmith import setup_langsmith
from app.security.guardrails import sanitise_query, validate_query
from app.tools.document_loader import extract_text_from_upload
from app.utils.logging import get_logger, setup_logging

# ── Initialise ────────────────────────────────────────────────────────────────
setup_logging()
logger   = get_logger(__name__)
settings = get_settings()

# Active session event queues for WebSocket streaming
_session_queues: dict[str, asyncio.Queue] = {}


async def _publish_graph_update(session_id: str, update: dict[str, Any]) -> None:
    """Publish serialisable node output to a session's WebSocket queue."""
    queue = _session_queues.get(session_id)
    if queue is None:
        return

    for node, node_update in update.items():
        if not isinstance(node_update, dict):
            continue

        payload: dict[str, Any] = {
            "type": "state_update",
            "node": node,
            "events": node_update.get("events", []),
            "sources": node_update.get("sources", []),
        }
        for key in ("progress", "report", "quality_score"):
            if key in node_update:
                payload[key] = node_update[key]

        await queue.put(payload)

# Rate limiter (slowapi)
_limiter = Limiter(key_func=get_remote_address)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan — startup and shutdown logic."""
    logger.info(
        "IntelliResearch API starting",
        environment=settings.environment,
        auth_enabled=settings.auth_enabled,
        tracing_enabled=settings.tracing_enabled,
    )
    setup_langsmith()
    yield
    logger.info("IntelliResearch API shutting down")


# ── FastAPI App ───────────────────────────────────────────────────────────────
app = FastAPI(
    title="IntelliResearch API",
    description="Multi-Agent AI Research Platform — LangGraph + OpenRouter + FAISS",
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

app.state.limiter = _limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)  # type: ignore[arg-type]

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins_list,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Request / Response Schemas ────────────────────────────────────────────────

class ResearchRequest(BaseModel):
    """Request body for starting a research session."""
    query: str = Field(..., min_length=3, max_length=5000, description="Research query")
    depth: str = Field(default="deep", pattern="^(quick|deep|expert)$")


class SessionResponse(BaseModel):
    """Response body for a created research session."""
    session_id: str
    ws_url: str
    message: str


class HITLRequest(BaseModel):
    """Request body for HITL (Human-in-the-Loop) approval/rejection."""
    approved: bool = Field(..., description="True to approve, False to reject and revise")
    feedback: str  = Field(default="", description="Optional feedback for revision")


class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    version: str
    environment: str
    auth_enabled: bool
    tracing_enabled: bool


# ── Routes ────────────────────────────────────────────────────────────────────

@app.get("/health", response_model=HealthResponse, tags=["System"])
async def health_check() -> HealthResponse:
    """Return API health and configuration status."""
    return HealthResponse(
        status="ok",
        version="1.0.0",
        environment=settings.environment,
        auth_enabled=settings.auth_enabled,
        tracing_enabled=settings.tracing_enabled,
    )


@app.get("/", tags=["System"])
async def root() -> dict[str, Any]:
    """Return API metadata."""
    return {
        "name":    "IntelliResearch",
        "version": "1.0.0",
        "agents":  [
            "planner", "paper_agent", "news_agent", "market_agent",
            "user_docs_agent", "analysis_agent", "insight_agent",
            "citation_agent", "report_builder", "judge_agent",
        ],
    }


@app.post("/api/v1/research", response_model=SessionResponse, status_code=status.HTTP_201_CREATED, tags=["Research"])
async def create_research_session(
    request: ResearchRequest,
    current_user: CurrentUser,
) -> SessionResponse:
    """
    Create a new research session.

    Validates the query through security guardrails, creates a session ID,
    and returns a WebSocket URL for streaming agent events.

    Args:
        request: Research query and depth settings.
        current_user: Authenticated user (from Clerk JWT or bypass).

    Returns:
        Session metadata including WebSocket URL.

    Raises:
        HTTPException 400: If query fails security validation.
    """
    clean_query = sanitise_query(request.query)
    validation  = validate_query(clean_query)

    if not validation.is_safe:
        logger.warning(
            "Query rejected by guardrails",
            user_id=current_user.user_id,
            reason=validation.reason,
        )
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=validation.reason,
        )

    session_id = str(uuid.uuid4())
    _session_queues[session_id] = asyncio.Queue()

    ws_url = f"ws://localhost:{settings.backend_port}/api/v1/ws/{session_id}"

    logger.info(
        "Research session created",
        session_id=session_id,
        user_id=current_user.user_id,
        query=clean_query[:60],
        depth=request.depth,
    )

    return SessionResponse(
        session_id=session_id,
        ws_url=ws_url,
        message=f"Session ready. Connect to WebSocket then POST to /api/v1/research/{session_id}/run",
    )


@app.post("/api/v1/research/{session_id}/run", status_code=status.HTTP_202_ACCEPTED, tags=["Research"])
async def run_research_session(
    session_id: str,
    current_user: CurrentUser,
    query: Annotated[str, Form(min_length=3, max_length=5000)],
    depth: Annotated[str, Form(pattern="^(quick|deep|expert)$")] = "deep",
    documents: list[UploadFile] = File(default=[]),
) -> dict[str, str]:
    """
    Start the agent graph for an existing session.

    Optionally accepts uploaded documents (PDF, DOCX, TXT) for RAG retrieval.

    Args:
        session_id: Session ID from create_research_session.
        query: Research query submitted as multipart form data.
        depth: Requested research depth.
        current_user: Authenticated user.
        documents: Optional list of uploaded files.

    Returns:
        Acknowledgement dict.

    Raises:
        HTTPException 404: If session_id doesn't exist.
    """
    if session_id not in _session_queues:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Session '{session_id}' not found. Call POST /api/v1/research first.",
        )

    # Extract text from uploaded documents
    uploaded_texts: list[str] = []
    for doc in documents:
        text = await extract_text_from_upload(doc)
        if text:
            uploaded_texts.append(text)
            logger.info("Document processed", filename=doc.filename, chars=len(text))

    async def _run_graph() -> None:
        """Background task that runs the full agent graph."""
        try:
            await run_research(
                session_id=session_id,
                query=sanitise_query(query),
                depth=depth,
                user_id=current_user.user_id,
                uploaded_docs=uploaded_texts,
                on_update=lambda update: _publish_graph_update(session_id, update),
            )

            final_state = await research_graph.aget_state(
                {"configurable": {"thread_id": session_id}}
            )
            values = dict(final_state.values)
            q = _session_queues.get(session_id)
            if q is not None:
                await q.put({
                    "type": "hitl_required" if final_state.next else "report_ready",
                    "report": values.get("report"),
                    "quality_score": values.get("quality_score"),
                    "sources": values.get("sources", []),
                    "progress": values.get("progress", 100),
                })
        except Exception as exc:
            q = _session_queues.get(session_id)
            if q is not None:
                await q.put({
                    "type": "fatal_error",
                    "error": str(exc),
                    "message": "Research failed. Check the backend terminal for details.",
                })
            logger.error("Graph execution failed", session_id=session_id, error=str(exc))

    asyncio.create_task(_run_graph())

    return {"status": "started", "session_id": session_id}


@app.post("/api/v1/research/{session_id}/hitl", tags=["Research"])
async def submit_hitl_decision(
    session_id: str,
    request: HITLRequest,
    current_user: CurrentUser,
) -> dict[str, Any]:
    """
    Submit a Human-in-the-Loop (HITL) approval or rejection.

    Called after the graph pauses at the hitl_review node.
    Resumes graph execution with the human's decision.

    Args:
        session_id: Session to resume.
        request: Approval status and optional feedback.
        current_user: Authenticated user (must be the session owner).

    Returns:
        Resumed graph final state summary.
    """
    if session_id not in _session_queues:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Session '{session_id}' not found.",
        )

    logger.info(
        "HITL decision received",
        session_id=session_id,
        user_id=current_user.user_id,
        approved=request.approved,
    )

    final_state = await resume_after_hitl(
        session_id=session_id,
        approved=request.approved,
        feedback=request.feedback,
    )

    return {
        "status":   "finalised" if request.approved else "awaiting_review",
        "approved": request.approved,
        "report":   final_state.get("report"),
        "quality_score": final_state.get("quality_score"),
    }


@app.get("/api/v1/research/{session_id}/state", tags=["Research"])
async def get_session_state(
    session_id: str,
    current_user: CurrentUser,
) -> dict[str, Any]:
    """
    Get the current state of a research session (for time travel / inspection).

    Args:
        session_id: Session to inspect.
        current_user: Authenticated user.

    Returns:
        Current LangGraph checkpoint state.
    """
    try:
        config = {"configurable": {"thread_id": session_id}}
        state  = research_graph.get_state(config)
        return {
            "session_id": session_id,
            "progress":   state.values.get("progress", 0),
            "events":     state.values.get("events", []),
            "report":     state.values.get("report"),
        }
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Session state not found: {exc}",
        ) from exc


# ── WebSocket ─────────────────────────────────────────────────────────────────

@app.websocket("/api/v1/ws/{session_id}")
async def websocket_stream(ws: WebSocket, session_id: str) -> None:
    """
    WebSocket endpoint for real-time agent event streaming.

    Streams structured events from the agent pipeline to the frontend.
    Events include: agent_start, agent_done, log, progress, report_ready.

    Args:
        ws: WebSocket connection.
        session_id: Session to stream events for.
    """
    await ws.accept()

    if session_id not in _session_queues:
        _session_queues[session_id] = asyncio.Queue()

    queue = _session_queues[session_id]

    try:
        while True:
            try:
                event = await asyncio.wait_for(queue.get(), timeout=120.0)
                await ws.send_json(event)
                if event.get("type") in ("hitl_required", "report_ready", "fatal_error"):
                    break
            except asyncio.TimeoutError:
                # Send keepalive ping every 120s to prevent timeout
                await ws.send_json({"type": "ping"})
    except WebSocketDisconnect:
        logger.info("WebSocket client disconnected", session_id=session_id)
    finally:
        _session_queues.pop(session_id, None)
