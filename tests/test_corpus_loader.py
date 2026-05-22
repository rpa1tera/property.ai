import json
from pathlib import Path

import pytest

from src.ingestion.corpus_loader import _tipo_from_path, load_file


def test_tipo_from_path_faq():
    assert _tipo_from_path(Path("faqs/x.csv")) == "faq"
    assert _tipo_from_path(Path("dataset/faq_property.csv")) == "faq"


def test_tipo_from_path_apolice():
    assert _tipo_from_path(Path("data/CG_RN_96.pdf")) == "apolice"
    assert _tipo_from_path(Path("apolices/x.pdf")) == "apolice"


def test_tipo_from_path_manual():
    assert _tipo_from_path(Path("manuais/x.pdf")) == "manual"
    assert _tipo_from_path(Path("data/Introducao_Resseguro.pdf")) == "manual"


def test_load_csv_valid(tmp_path):
    csv = tmp_path / "faq.csv"
    csv.write_text(
        "pergunta,resposta\n"
        "O que é franquia?,Valor mínimo que o segurado paga em caso de sinistro previsto na apólice.\n"
        "Cobre incêndio?,Sim a apólice patrimonial cobre danos por incêndio raio e explosão conforme condições.\n",
        encoding="utf-8",
    )
    docs = load_file(str(csv))
    assert len(docs) == 2
    assert all(d.metadata["tipo"] == "faq" for d in docs)
    assert all("Pergunta:" in d.page_content for d in docs)


def test_load_csv_drops_short_answers(tmp_path):
    csv = tmp_path / "faq.csv"
    csv.write_text(
        "pergunta,resposta\n"
        "O que é X?,Curto demais.\n"  # < 10 palavras → descartado
        "O que é Y?,Esta resposta tem palavras suficientes para passar pelo filtro de tamanho mínimo.\n",
        encoding="utf-8",
    )
    docs = load_file(str(csv))
    assert len(docs) == 1


def test_load_csv_missing_columns(tmp_path):
    csv = tmp_path / "bad.csv"
    csv.write_text("foo,bar\n1,2\n", encoding="utf-8")
    with pytest.raises(ValueError, match="pergunta.*resposta"):
        load_file(str(csv))


def test_load_json(tmp_path):
    j = tmp_path / "items.json"
    data = [{"q": "x", "a": "y"}, {"q": "z", "a": "w"}]
    j.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
    docs = load_file(str(j))
    assert len(docs) == 2
    assert all("fonte" in d.metadata for d in docs)


def test_unsupported_format(tmp_path):
    bad = tmp_path / "x.docx"
    bad.write_text("foo", encoding="utf-8")
    with pytest.raises(ValueError, match="Formato"):
        load_file(str(bad))
