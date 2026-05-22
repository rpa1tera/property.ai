"""Reranking via cross-encoder multilíngue.

Modelo: BAAI/bge-reranker-v2-m3 (CPU-friendly, suporta PT-BR).
Carga lazy via lru_cache — primeiro `rerank()` paga o download.

Função pública: `rerank(query, docs, top_n) -> list[Document]`.
A função retorna documentos com metadado `rerank_score` adicionado.
"""
from __future__ import annotations

from functools import lru_cache

from langchain_core.documents import Document

_DEFAULT_MODEL = "BAAI/bge-reranker-v2-m3"


@lru_cache(maxsize=1)
def _get_reranker(model_name: str = _DEFAULT_MODEL):
    """Importa e instancia o cross-encoder. Cache de processo."""
    from sentence_transformers import CrossEncoder
    return CrossEncoder(model_name, max_length=512)


def rerank(
    query: str,
    docs: list[Document],
    top_n: int = 5,
    model_name: str = _DEFAULT_MODEL,
) -> list[Document]:
    """Reordena `docs` por relevância à `query` via cross-encoder.

    Retorna top-n documentos com `metadata["rerank_score"]` preenchido.
    Empty list-in → empty list-out (sem inicializar o modelo).
    """
    if not docs:
        return []

    model = _get_reranker(model_name)
    pairs = [(query, d.page_content) for d in docs]
    scores = model.predict(pairs)

    scored = sorted(zip(docs, scores), key=lambda x: float(x[1]), reverse=True)
    out: list[Document] = []
    for doc, score in scored[:top_n]:
        new_meta = dict(doc.metadata)
        new_meta["rerank_score"] = float(score)
        out.append(Document(page_content=doc.page_content, metadata=new_meta))
    return out
