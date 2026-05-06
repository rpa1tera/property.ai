from unittest.mock import MagicMock, patch

from langchain_core.documents import Document

from src.rag.retriever import search


def _mock_doc(content: str = "Cobertura de incêndio está incluída.", tipo: str = "apolice") -> Document:
    return Document(page_content=content, metadata={"tipo": tipo, "fonte": "test.pdf"})


def test_search_returns_documents():
    doc = _mock_doc()
    with patch("src.rag.retriever.load_vectorstore") as mock_vs_factory:
        mock_vs = MagicMock()
        mock_vs.as_retriever.return_value.invoke.return_value = [doc]
        mock_vs_factory.return_value = mock_vs

        results = search("cobertura incêndio")

    assert len(results) == 1
    assert isinstance(results[0], Document)
    assert "incêndio" in results[0].page_content


def test_search_empty_result():
    with patch("src.rag.retriever.load_vectorstore") as mock_vs_factory:
        mock_vs = MagicMock()
        mock_vs.as_retriever.return_value.invoke.return_value = []
        mock_vs_factory.return_value = mock_vs

        results = search("pergunta sem nenhuma resposta na base")

    assert results == []


def test_search_respects_top_k():
    docs = [_mock_doc(f"doc {i}") for i in range(10)]
    with patch("src.rag.retriever.load_vectorstore") as mock_vs_factory:
        mock_vs = MagicMock()
        mock_vs.as_retriever.return_value.invoke.return_value = docs[:3]
        mock_vs_factory.return_value = mock_vs

        results = search("cobertura", top_k=3)

    assert len(results) == 3
