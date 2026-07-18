"""
app/rag/retriever.py
RAG retriever — convenience wrapper over FAISSVectorStore.
"""

from app.rag.vectorstore import FAISSVectorStore
from app.utils.logging import get_logger

logger = get_logger(__name__)


async def retrieve_relevant_chunks(
    vector_store: FAISSVectorStore,
    query: str,
    k: int = 5,
) -> list[str]:
    """
    Retrieve the most semantically relevant document chunks for a query.

    Args:
        vector_store: Initialised FAISSVectorStore for the session.
        query: Search query string.
        k: Number of chunks to return.

    Returns:
        List of relevant text chunks, empty if no documents indexed.
    """
    try:
        chunks = await vector_store.similarity_search_async(query, k=k)
        logger.info(
            "RAG retrieval complete",
            query=query[:40],
            chunks_returned=len(chunks),
        )
        return chunks
    except Exception as exc:
        logger.error("RAG retrieval failed", query=query[:40], error=str(exc))
        return []
