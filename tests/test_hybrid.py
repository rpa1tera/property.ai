from unittest.mock import patch

from langchain_core.documents import Document

from src.rag.hybrid import hybrid_search


def _doc(text, fonte="a.pdf"):
    return Document(page_content=text, metadata={"fonte": fonte})


def test_hybrid_combines_dense_and_lexical_via_rrf_without_rerank():
    dense = [_doc("franquia detalhada", "doc1.pdf"), _doc("outro", "doc2.pdf")]
    lexical = [_doc("franquia detalhada", "doc1.pdf"), _doc("sinistro", "doc3.pdf")]

    with patch("src.rag.hybrid.dense_search", return_value=dense), \
         patch("src.rag.hybrid.bm25_search", return_value=lexical):
        out = hybrid_search("franquia", enable_rerank=False, top_k=3)

    # franquia detalhada aparece em ambos → topo
    assert out[0].page_content == "franquia detalhada"
    assert len(out) <= 3


def test_hybrid_calls_rerank_when_enabled():
    dense = [_doc("A"), _doc("B")]
    lexical = [_doc("B"), _doc("C")]

    with patch("src.rag.hybrid.dense_search", return_value=dense), \
         patch("src.rag.hybrid.bm25_search", return_value=lexical), \
         patch("src.rag.rerank.rerank", return_value=[_doc("B")]) as mock_rerank:
        out = hybrid_search("q", enable_rerank=True, top_k=5)

    mock_rerank.assert_called_once()
    assert len(out) == 1
    assert out[0].page_content == "B"


def test_hybrid_empty_when_both_sources_empty():
    with patch("src.rag.hybrid.dense_search", return_value=[]), \
         patch("src.rag.hybrid.bm25_search", return_value=[]):
        out = hybrid_search("q", enable_rerank=False)
    assert out == []
