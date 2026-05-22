from langchain_core.documents import Document

from src.rag.lexical import BM25Index, bm25_search, tokenize


def test_tokenize_strips_accents_and_lowers():
    assert tokenize("Cláusula Quarta") == ["clausula", "quarta"]


def test_tokenize_splits_on_punctuation():
    assert tokenize("seguro, incêndio. dano!") == ["seguro", "incendio", "dano"]


def _make_index(texts):
    from rank_bm25 import BM25Okapi
    docs = [Document(page_content=t, metadata={"fonte": "x.pdf"}) for t in texts]
    bm25 = BM25Okapi([tokenize(t) for t in texts])
    return BM25Index(bm25=bm25, documents=docs)


def test_bm25_finds_exact_term():
    index = _make_index([
        "A franquia é a participação obrigatória.",
        "Incêndio é uma cobertura básica.",
        "Vistoria de sinistro é obrigatória.",
    ])
    out = bm25_search("franquia", index=index, k=3)
    assert out
    assert "franquia" in out[0].page_content.lower()


def test_bm25_returns_empty_for_unknown_terms():
    index = _make_index(["Texto sobre seguros."])
    out = bm25_search("xyzqwerty", index=index, k=5)
    assert out == []


def test_bm25_respects_accent_insensitivity():
    # BM25 precisa de N>>df para gerar IDF > 0; usamos corpus maior com 1 doc relevante.
    index = _make_index([
        "Apólice tem vigência de 12 meses.",
        "Outro texto qualquer A.",
        "Outro texto qualquer B.",
        "Mais um texto C.",
        "Mais um texto D.",
    ])
    out = bm25_search("apolice", index=index, k=2)
    assert out
    assert "Apólice" in out[0].page_content
