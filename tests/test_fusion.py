from langchain_core.documents import Document

from src.rag.fusion import rrf_merge


def _doc(text, fonte="a.pdf", pagina=1):
    return Document(page_content=text, metadata={"fonte": fonte, "pagina": pagina})


def test_rrf_merges_two_lists_promoting_common_documents():
    a = _doc("franquia explicada", "doc1.pdf")
    b = _doc("cobertura de incêndio", "doc2.pdf")
    c = _doc("vistoria sinistro", "doc3.pdf")

    # B aparece em rank 0 de ambas; A só rank 1 e 2 → B deve vencer.
    dense = [b, a, c]
    lexical = [b, c, a]

    merged = rrf_merge([dense, lexical])
    assert merged[0].page_content == "cobertura de incêndio"


def test_rrf_deterministic_same_input_same_order():
    docs = [_doc(f"chunk {i}", f"d{i}.pdf") for i in range(5)]
    out_a = rrf_merge([docs, docs[::-1]])
    out_b = rrf_merge([docs, docs[::-1]])
    assert [d.page_content for d in out_a] == [d.page_content for d in out_b]


def test_rrf_empty_lists():
    assert rrf_merge([]) == []
    assert rrf_merge([[], []]) == []


def test_rrf_top_n_truncates():
    docs = [_doc(f"c{i}", f"d{i}.pdf") for i in range(10)]
    merged = rrf_merge([docs], top_n=3)
    assert len(merged) == 3


def test_rrf_dedups_by_content_and_source():
    a = _doc("mesmo texto", "a.pdf")
    a_dup = _doc("mesmo texto", "a.pdf")
    b = _doc("outro", "b.pdf")
    merged = rrf_merge([[a, b], [a_dup, b]])
    assert len(merged) == 2  # a/a_dup colapsam
