from langchain_core.documents import Document

from src.chatbot.crag_evaluator import docs_to_context, evaluate_sufficiency


def _doc(content, score=None):
    meta = {"fonte": "x.pdf", "tipo": "apolice"}
    if score is not None:
        meta["rerank_score"] = score
    return Document(page_content=content, metadata=meta)


def test_empty_docs_is_insufficient():
    assert evaluate_sufficiency("q", []) is False


def test_threshold_positive_score_is_sufficient():
    docs = [_doc("relevante", score=0.5), _doc("ruido", score=-2.0)]
    assert evaluate_sufficiency("q", docs) is True


def test_threshold_all_negative_is_insufficient():
    docs = [_doc("a", score=-1.0), _doc("b", score=-3.5)]
    assert evaluate_sufficiency("q", docs, threshold=0.0) is False


def test_no_score_falls_back_to_trust_retrieval():
    docs = [_doc("sem score, sem rerank")]
    # Sem rerank_score em metadata: confia no retrieval (não escala).
    assert evaluate_sufficiency("q", docs) is True


def test_docs_to_context_includes_source():
    out = docs_to_context([_doc("X", score=1.0)])
    assert "x.pdf" in out
    assert "X" in out
