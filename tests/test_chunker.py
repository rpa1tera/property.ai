from langchain_core.documents import Document

from src.ingestion.chunker import chunk_documents


def test_structural_split_by_clausula():
    text = (
        "Cláusula 1 - Objeto. Este contrato regula a cobertura de incêndio.\n"
        "Cláusula 2 - Definições. Entende-se por sinistro o evento previsto.\n"
        "Cláusula 3 - Vigência. A apólice tem vigência de 12 meses."
    )
    doc = Document(page_content=text, metadata={"fonte": "x.pdf", "tipo": "apolice"})
    chunks = chunk_documents([doc])
    assert len(chunks) >= 3
    assert all(c.metadata["chunk_strategy"] == "structural" for c in chunks)
    assert any("Cláusula 1" in c.page_content for c in chunks)
    assert any("Cláusula 2" in c.page_content for c in chunks)


def test_structural_split_by_artigo():
    text = (
        "Art. 1 Primeira disposição contratual extensa o suficiente para virar chunk.\n"
        "Art. 2 Segunda disposição contratual também relevante para fins de teste.\n"
        "Art. 3 Terceira disposição final encerrando o documento."
    )
    doc = Document(page_content=text, metadata={"fonte": "y.pdf"})
    chunks = chunk_documents([doc])
    assert all(c.metadata["chunk_strategy"] == "structural" for c in chunks)
    assert len(chunks) >= 3


def test_fallback_recursive_when_no_markers():
    # Texto livre sem marcadores estruturais.
    text = ". ".join([f"Esta é a sentença número {i} do documento de teste" for i in range(60)])
    doc = Document(page_content=text, metadata={"fonte": "manual.pdf"})
    chunks = chunk_documents([doc], chunk_size=200, overlap=20)
    assert len(chunks) >= 2
    assert all(c.metadata["chunk_strategy"] == "recursive" for c in chunks)


def test_metadata_preserved():
    doc = Document(
        page_content="Cláusula 1 - Objeto do contrato.\nCláusula 2 - Definições importantes.",
        metadata={"fonte": "z.pdf", "pagina": 7, "tipo": "apolice"},
    )
    chunks = chunk_documents([doc])
    for c in chunks:
        assert c.metadata["fonte"] == "z.pdf"
        assert c.metadata["pagina"] == 7
        assert c.metadata["tipo"] == "apolice"
        assert "chunk_index" in c.metadata
        assert c.metadata["chunk_strategy"] == "structural"


def test_chunk_index_is_sequential():
    doc = Document(
        page_content="Cláusula 1 A.\nCláusula 2 B.\nCláusula 3 C.",
        metadata={"fonte": "w.pdf"},
    )
    chunks = chunk_documents([doc])
    indices = [c.metadata["chunk_index"] for c in chunks]
    assert indices == list(range(len(chunks)))
