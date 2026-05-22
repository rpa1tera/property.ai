from unittest.mock import MagicMock, patch

from langchain_core.documents import Document

from src.rag import rerank as rerank_mod


def _docs():
    return [
        Document(page_content="texto A", metadata={"fonte": "a.pdf"}),
        Document(page_content="texto B", metadata={"fonte": "b.pdf"}),
        Document(page_content="texto C", metadata={"fonte": "c.pdf"}),
    ]


def test_rerank_empty_short_circuits():
    # Não deve tentar carregar o modelo se docs estiver vazio.
    with patch.object(rerank_mod, "_get_reranker") as mock:
        out = rerank_mod.rerank("qualquer", [], top_n=3)
    assert out == []
    mock.assert_not_called()


def test_rerank_reorders_by_score_and_adds_metadata():
    fake_model = MagicMock()
    # Score do segundo doc é o maior → deve ir para topo.
    fake_model.predict.return_value = [0.1, 0.9, 0.5]
    with patch.object(rerank_mod, "_get_reranker", return_value=fake_model):
        out = rerank_mod.rerank("query", _docs(), top_n=3)

    assert [d.page_content for d in out] == ["texto B", "texto C", "texto A"]
    assert out[0].metadata["rerank_score"] == 0.9
    assert all("rerank_score" in d.metadata for d in out)


def test_rerank_top_n_truncates():
    fake_model = MagicMock()
    fake_model.predict.return_value = [0.1, 0.9, 0.5]
    with patch.object(rerank_mod, "_get_reranker", return_value=fake_model):
        out = rerank_mod.rerank("query", _docs(), top_n=2)
    assert len(out) == 2
    assert out[0].page_content == "texto B"
