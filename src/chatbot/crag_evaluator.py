"""Avaliação de suficiência do contexto recuperado.

Sprint 3: a versão LLM ("SUFICIENTE/INSUFICIENTE") foi substituída por threshold
sobre o `rerank_score` (preenchido por src.rag.rerank). Economiza 1 LLM call/turno.

A assinatura `evaluate_sufficiency(question, docs, llm=None)` é preservada porque
testes legados fazem patch dela. O parâmetro `llm` é ignorado neste caminho.
"""
from __future__ import annotations

from typing import Any

from langchain_core.documents import Document

_DEFAULT_THRESHOLD = 0.0  # bge-reranker-v2-m3 emite logits; >0 indica relevância.


def docs_to_context(docs: list[Document]) -> str:
    return "\n\n---\n\n".join(
        f"[Fonte: {doc.metadata.get('fonte', 'N/A')} | "
        f"Tipo: {doc.metadata.get('tipo', 'N/A')}]\n{doc.page_content}"
        for doc in docs
    )


def evaluate_sufficiency(
    question: str,
    docs: list[Document],
    llm: Any | None = None,
    threshold: float = _DEFAULT_THRESHOLD,
) -> bool:
    """Retorna True se houver pelo menos 1 chunk com rerank_score ≥ threshold.

    Fallback: se nenhum doc tem `rerank_score` em metadata (rerank desabilitado),
    considera suficiente quando há ao menos 1 doc — delega o filtro ao retrieval.
    """
    if not docs:
        return False

    scored = [d for d in docs if "rerank_score" in d.metadata]
    if not scored:
        return True  # sem score: confia no retrieval.

    top = max(float(d.metadata["rerank_score"]) for d in scored)
    return top >= threshold
