"""Orquestrador de retrieval híbrido.

Pipeline: dense_search + bm25_search → rrf_merge → rerank (opcional).
Função pública: `hybrid_search(query, top_k)`.
"""
from __future__ import annotations

from typing import Any

from langchain_core.documents import Document

from src.config import get_settings
from src.rag.dense import dense_search
from src.rag.fusion import rrf_merge
from src.rag.lexical import bm25_search


def hybrid_search(
    query: str,
    top_k: int | None = None,
    *,
    candidate_k: int = 20,
    enable_rerank: bool = True,
    vectorstore: Any | None = None,
) -> list[Document]:
    """Retrieval híbrido com fusão RRF e rerank opcional.

    Args:
        query: pergunta do usuário.
        top_k: tamanho do retorno final. Default = settings.rerank_top_k.
        candidate_k: chunks por subsistema antes da fusão.
        enable_rerank: se True, aplica cross-encoder no topo. Caro na 1ª chamada.
        vectorstore: instância PropertyVectorStore (opcional, lazy load).
    """
    s = get_settings()
    final_k = top_k or s.rerank_top_k

    dense = dense_search(query, k=candidate_k, vectorstore=vectorstore)
    lexical = bm25_search(query, k=candidate_k, vectorstore=vectorstore)
    merged = rrf_merge([dense, lexical], top_n=candidate_k)

    if not enable_rerank or not merged:
        return merged[:final_k]

    from src.rag.rerank import rerank
    return rerank(query, merged, top_n=final_k)
