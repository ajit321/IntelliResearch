"""
app/agents/user_docs_agent.py
User Docs Agent — performs RAG retrieval over user-uploaded documents.
Uses FAISS vector index with HuggingFace sentence-transformers.
"""

from datetime import datetime, timezone
from typing import Any

from app.agents.state import ResearchState, SourceDoc, AgentEvent
from app.rag.vectorstore import FAISSVectorStore
from app.rag.retriever import retrieve_relevant_chunks
from app.utils.logging import get_logger
from app.utils.decorators import timer

logger = get_logger(__name__)


@timer
async def user_docs_agent_node(state: ResearchState) -> dict[str, Any]:
    """
    User Docs Agent node — indexes and retrieves from user-uploaded documents.

    If the user uploaded documents, this agent:
    1. Indexes them into a session-scoped FAISS vector store
    2. Retrieves the most semantically relevant chunks for each sub-question
    3. Returns them as SourceDoc entries for downstream analysis

    If no documents were uploaded, returns empty state gracefully.

    Args:
        state: Current pipeline state (includes raw_texts from upload).

    Returns:
        State update dict with sources, raw_texts, and events.
    """
    session_id  = state["session_id"]
    raw_texts   = state.get("raw_texts", [])
    agent_tasks = state.get("agent_tasks", {})
    queries: list[str] = agent_tasks.get("user_docs_agent", [state["query"]])

    logger.info(
        "User Docs Agent started",
        session_id=session_id,
        doc_count=len(raw_texts),
        queries=len(queries),
    )

    start_event: AgentEvent = {
        "agent":     "user_docs_agent",
        "action":    "started",
        "message":   f"Processing {len(raw_texts)} uploaded document(s)...",
        "timestamp": _now(),
    }

    if not raw_texts:
        logger.info("No user documents uploaded — skipping RAG retrieval")
        done_event: AgentEvent = {
            "agent":     "user_docs_agent",
            "action":    "skipped",
            "message":   "No uploaded documents — agent skipped",
            "timestamp": _now(),
        }
        return {
            "sources": [],
            "events":  [start_event, done_event],
            "progress": 28,
        }

    # Build or update session-scoped FAISS index
    vector_store = FAISSVectorStore(session_id=session_id)
    await vector_store.add_texts_async(raw_texts)
    logger.info("Documents indexed into FAISS", doc_count=len(raw_texts))

    # Retrieve relevant chunks for each sub-question
    all_sources: list[SourceDoc] = []
    for query in queries:
        chunks = await retrieve_relevant_chunks(vector_store, query)
        for i, chunk in enumerate(chunks):
            all_sources.append(
                SourceDoc(
                    title=f"User Document — Chunk {i + 1}",
                    url="user_upload",
                    content=chunk,
                    source_type="user_doc",
                    confidence=0.88,
                    published_date="",
                )
            )

    done_event = {
        "agent":     "user_docs_agent",
        "action":    "completed",
        "message":   f"Retrieved {len(all_sources)} relevant chunk(s) from uploaded docs",
        "timestamp": _now(),
    }

    logger.info("User Docs Agent completed", chunks_retrieved=len(all_sources))

    return {
        "sources": all_sources,
        "events":  [start_event, done_event],
        "progress": 30,
    }


def _now() -> str:
    """Return current UTC timestamp as ISO string."""
    return datetime.now(timezone.utc).isoformat()
