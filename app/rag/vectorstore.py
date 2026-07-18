"""
app/rag/vectorstore.py
FAISS vector store wrapper with async support.
Manages session-scoped document indices for user-uploaded documents.
"""

import asyncio
from pathlib import Path
from typing import Any

from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.config import get_settings
from app.utils.logging import get_logger

logger = get_logger(__name__)
settings = get_settings()

# Shared embedding model (loaded once, reused)
_embedding_model: HuggingFaceEmbeddings | None = None


def _get_embeddings() -> HuggingFaceEmbeddings:
    """Lazily load the HuggingFace embedding model (singleton)."""
    global _embedding_model
    if _embedding_model is None:
        logger.info("Loading embedding model", model=settings.embedding_model)
        _embedding_model = HuggingFaceEmbeddings(
            model_name=settings.embedding_model,
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True},
        )
    return _embedding_model


class FAISSVectorStore:
    """
    Session-scoped FAISS vector store for document indexing and retrieval.

    Each research session gets its own in-memory FAISS index.
    Supports async text addition and similarity search.
    """

    def __init__(self, session_id: str) -> None:
        """
        Initialise a FAISS store for a session.

        Args:
            session_id: Unique session identifier.
        """
        self.session_id = session_id
        self._index: FAISS | None = None
        self._embeddings = _get_embeddings()
        self._splitter = RecursiveCharacterTextSplitter(
            chunk_size=settings.chunk_size,
            chunk_overlap=settings.chunk_overlap,
            separators=["\n\n", "\n", " ", ""],
        )
        logger.info("FAISSVectorStore initialised", session_id=session_id)

    async def add_texts_async(self, texts: list[str]) -> None:
        """
        Split and index a list of raw texts asynchronously.

        Runs the blocking FAISS operations in a thread pool executor
        to avoid blocking the event loop.

        Args:
            texts: List of raw document text strings.
        """
        if not texts:
            return

        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, self._add_texts_sync, texts)

    def _add_texts_sync(self, texts: list[str]) -> None:
        """
        Synchronous text indexing — runs in executor thread.

        Args:
            texts: Raw text strings to chunk and index.
        """
        chunks: list[str] = []
        for text in texts:
            if text.strip():
                chunks.extend(self._splitter.split_text(text))

        if not chunks:
            logger.warning("No chunks after splitting", session_id=self.session_id)
            return

        if self._index is None:
            self._index = FAISS.from_texts(chunks, embedding=self._embeddings)
            logger.info(
                "FAISS index created",
                session_id=self.session_id,
                chunks=len(chunks),
            )
        else:
            self._index.add_texts(chunks)
            logger.info(
                "Chunks added to FAISS index",
                session_id=self.session_id,
                new_chunks=len(chunks),
            )

    async def similarity_search_async(
        self,
        query: str,
        k: int | None = None,
    ) -> list[str]:
        """
        Perform async semantic similarity search.

        Args:
            query: Search query string.
            k: Number of results to return (defaults to settings.top_k_retrieval).

        Returns:
            List of relevant text chunks.
        """
        if self._index is None:
            logger.warning("FAISS index empty — no documents indexed yet")
            return []

        top_k = k or settings.top_k_retrieval
        loop  = asyncio.get_event_loop()
        docs  = await loop.run_in_executor(
            None,
            lambda: self._index.similarity_search(query, k=top_k),
        )

        results = [doc.page_content for doc in docs]
        logger.debug(
            "FAISS similarity search",
            query=query[:40],
            results=len(results),
        )
        return results

    def save_to_disk(self) -> None:
        """Persist the FAISS index to disk for checkpoint recovery."""
        if self._index is None:
            return
        save_path = Path(settings.faiss_index_path) / self.session_id
        save_path.mkdir(parents=True, exist_ok=True)
        self._index.save_local(str(save_path))
        logger.info("FAISS index saved", path=str(save_path))

    def load_from_disk(self) -> bool:
        """
        Load a persisted FAISS index from disk.

        Returns:
            True if loaded successfully, False if no saved index exists.
        """
        load_path = Path(settings.faiss_index_path) / self.session_id
        if not load_path.exists():
            return False
        self._index = FAISS.load_local(
            str(load_path),
            embeddings=self._embeddings,
            allow_dangerous_deserialization=True,
        )
        logger.info("FAISS index loaded from disk", session_id=self.session_id)
        return True
