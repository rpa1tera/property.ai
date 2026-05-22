from langchain_core.documents import Document

from src.ingestion.cleaner import clean_documents, clean_text


def test_fix_hyphen_eol():
    assert clean_text("seguro pa-\ntrimonial cobre") == "seguro patrimonial cobre"


def test_soft_line_join():
    # quebra de linha sem pontuação final → vira espaço
    out = clean_text("a primeira parte\ncontinua aqui.")
    assert "primeira parte continua" in out


def test_keeps_hard_line_break_after_punctuation():
    # quebra após '.' deve ser preservada (não vira espaço)
    out = clean_text("Frase final.\nNova frase começa.")
    assert "Frase final." in out


def test_removes_page_number_lines():
    out = clean_text("Texto relevante.\n42\nOutro texto.")
    assert "42" not in out.split()


def test_removes_boilerplate():
    text = "Conteúdo útil aqui.\nTodos os direitos reservados Mapfre.\nMais conteúdo útil."
    out = clean_text(text)
    assert "direitos reservados" not in out.lower()
    assert "Conteúdo útil" in out
    assert "Mais conteúdo útil" in out


def test_collapses_multi_spaces():
    assert "a b" in clean_text("a       b")


def test_clean_documents_drops_too_short():
    docs = [
        Document(page_content="curto", metadata={"fonte": "x"}),
        Document(page_content="este é um documento com cinco palavras pelo menos.", metadata={"fonte": "y"}),
    ]
    out = clean_documents(docs)
    assert len(out) == 1
    assert out[0].metadata["fonte"] == "y"
