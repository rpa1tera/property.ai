"""Reciprocal Rank Fusion (RRF).

Algoritmo determinístico (sem LLM) para combinar múltiplas listas ranqueadas.
Cormack et al., 2009: score(d) = Σ 1 / (k + rank(d, list_i))
"""
from __future__ import annotations

from langchain_core.documents import Document


def _doc_key(doc: Document) -> str:
    """Identificador estável para deduplicação entre listas."""
    src = str(doc.metadata.get("fonte", ""))
    page = str(doc.metadata.get("pagina", ""))
    head = doc.page_content[:120]
    return f"{src}|{page}|{head}"


def rrf_merge(
    results_lists: list[list[Document]],
    k: int = 60,
    top_n: int | None = None,
) -> list[Document]:
    """Combina N listas ranqueadas via Reciprocal Rank Fusion.

    Args:
        results_lists: cada lista é um ranking (índice 0 = top-1).
        k: constante RRF (default 60, padrão da literatura).
        top_n: trunca o resultado final. None = retorna todos.
    """
    scores: dict[str, float] = {}
    by_key: dict[str, Document] = {}

    for ranking in results_lists:
        for rank, doc in enumerate(ranking):
            key = _doc_key(doc)
            scores[key] = scores.get(key, 0.0) + 1.0 / (k + rank + 1)
            by_key.setdefault(key, doc)

    ordered = sorted(scores.items(), key=lambda kv: (-kv[1], kv[0]))
    merged = [by_key[key] for key, _ in ordered]
    return merged[:top_n] if top_n else merged
