"""Busca densa via ChromaDB.

Wrapper isolado sobre PropertyVectorStore. Função pública: `dense_search`.
"""
from __future__ import annotations

from typing import Any

from langchain_core.documents import Document

from src.rag.embeddings import load_vectorstore


def dense_search(
    query: str,
    k: int = 10,
    vectorstore: Any | None = None,
) -> list[Document]:
    """Retorna os top-k chunks semanticamente mais próximos da query."""
    vs = vectorstore or load_vectorstore()
    return vs.similarity_search(query, k=k)
