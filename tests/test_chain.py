from unittest.mock import MagicMock, patch

from langchain_core.documents import Document

from src.chatbot.intents import classify_intent


# --- Intent classification (no mocks needed) ---

def test_intent_cobertura():
    assert classify_intent("Meu seguro cobre enchente?") == "cobertura"
    assert classify_intent("O que está coberto na apólice?") == "cobertura"


def test_intent_franquia():
    assert classify_intent("Qual é a franquia para sinistro de incêndio?") == "franquia"
    assert classify_intent("O que é franquia?") == "franquia"


def test_intent_sinistro():
    assert classify_intent("Como aciono o seguro após um roubo?") == "sinistro"
    assert classify_intent("Quero comunicar um sinistro") == "sinistro"


def test_intent_apolice():
    assert classify_intent("Como obter a 2ª via da minha apólice?") == "apolice_documentos"
    assert classify_intent("Quero renovar meu contrato") == "apolice_documentos"


def test_intent_cancelamento():
    assert classify_intent("Quero cancelar meu seguro") == "cancelamento"
    assert classify_intent("Como faço para encerrar minha apólice?") == "cancelamento"


def test_intent_escalonamento():
    assert classify_intent("Quero falar com um atendente") == "escalonamento"
    assert classify_intent("Me passa para um humano") == "escalonamento"


def test_intent_outros():
    assert classify_intent("Olá, bom dia") == "outros"


# --- Chain answer (with mocks) ---

def test_answer_escalonamento_direct():
    from src.chatbot.chain import answer
    result = answer("Quero falar com um atendente humano")
    assert result["escalated"] is True
    assert result["intent"] == "escalonamento"
    assert result["sources"] == []


def test_answer_insufficient_escalates():
    from src.chatbot.chain import answer
    doc = Document(page_content="Texto irrelevante.", metadata={"tipo": "apolice", "fonte": "test.pdf"})

    with patch("src.chatbot.chain.fused_retrieval", return_value=[doc]), \
         patch("src.chatbot.chain.evaluate_sufficiency", return_value=False), \
         patch("src.chatbot.chain._get_llm", return_value=MagicMock()):
        result = answer("Qual é a cobertura de granizo?")

    assert result["escalated"] is True
    assert "answer" in result


def test_answer_sufficient_returns_response():
    from src.chatbot.chain import answer
    doc = Document(
        page_content="A cobertura de granizo está incluída na modalidade completa.",
        metadata={"tipo": "apolice", "fonte": "condicoes.pdf"},
    )
    mock_llm = MagicMock()
    mock_llm.__or__ = MagicMock(return_value=mock_llm)

    with patch("src.chatbot.chain.fused_retrieval", return_value=[doc]), \
         patch("src.chatbot.chain.evaluate_sufficiency", return_value=True), \
         patch("src.chatbot.chain._get_llm", return_value=MagicMock()), \
         patch("src.chatbot.chain.StrOutputParser") as mock_parser_cls:

        mock_chain = MagicMock()
        mock_chain.invoke.return_value = "A cobertura de granizo está incluída."
        mock_parser_cls.return_value.__ror__ = MagicMock(return_value=mock_chain)

        with patch("src.chatbot.chain._ANSWER_PROMPT.__or__", return_value=mock_chain):
            result = answer("Meu seguro cobre granizo?")

    assert "answer" in result
    assert result["intent"] == "cobertura"
    assert "condicoes.pdf" in result["sources"]
