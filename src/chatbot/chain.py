"""Orquestrador único do property.ai.

Função pública: `answer(question, *, stream, enable_fusion, enable_crag, ...)`.

A UI consome `answer(..., stream=True)` (caminho rápido, sem extra LLM calls).
RAGAS consome `answer(..., enable_fusion=True, enable_crag=True)` (caminho completo).

Os nomes `fused_retrieval`, `evaluate_sufficiency`, `_get_llm`, `_ANSWER_PROMPT`
são mantidos como atributos de módulo porque os testes fazem patch deles.
"""
from __future__ import annotations

from typing import Any, Iterator

from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq

from src.chatbot.crag_evaluator import docs_to_context, evaluate_sufficiency
from src.chatbot.intents import classify_intent
from src.chatbot.rag_fusion import fused_retrieval
from src.config import get_settings
from src.rag.hybrid import hybrid_search

_ANSWER_PROMPT = ChatPromptTemplate.from_template(
    "Você é o property.ai, assistente especializado em seguros patrimoniais (property).\n"
    "Responda APENAS com base nas informações dentro de <chunks>.\n"
    "Se a resposta não estiver nos chunks, diga que vai encaminhar para um especialista.\n"
    "Seja claro, educado e objetivo.\n\n"
    "<chunks>\n{context}\n</chunks>\n\n"
    "<query>\n{question}\n</query>\n\n"
    "Resposta:"
)

_ESCALATION_MSG = (
    "Entendo sua solicitação. Vou encaminhar você para um de nossos especialistas "
    "em seguros patrimoniais. Em breve você receberá atendimento personalizado. "
    "Há algo mais que eu possa registrar antes de transferir?"
)

_NO_INFO_MSG = (
    "Não encontrei informações suficientes na minha base de conhecimento para responder "
    "com precisão a essa pergunta. Vou encaminhar para um especialista que poderá te ajudar melhor."
)


def _get_llm(streaming: bool = False) -> ChatGroq:
    s = get_settings()
    return ChatGroq(
        model=s.groq_model,
        temperature=s.groq_temperature,
        streaming=streaming,
    )


def _unique_sources(docs: list[Document]) -> list[str]:
    seen: set[str] = set()
    sources: list[str] = []
    for doc in docs:
        src = doc.metadata.get("fonte", "N/A")
        if src not in seen:
            seen.add(src)
            sources.append(src)
    return sources


def _retrieve(
    question: str,
    vectorstore: Any,
    llm: ChatGroq | None,
    enable_fusion: bool,
    enable_rerank: bool = True,
) -> list[Document]:
    """Caminho de retrieval.

    - `enable_fusion=True`: RAG-Fusion legado (3 variações via LLM). Opt-in p/ A/B.
    - default: hybrid_search (dense + BM25 + RRF + rerank opcional). Determinístico.
    """
    if enable_fusion:
        return fused_retrieval(question, vectorstore=vectorstore, top_k=5, llm=llm)
    return hybrid_search(
        question,
        vectorstore=vectorstore,
        enable_rerank=enable_rerank,
    )


def _build_escalation(intent: str, sources: list[str] | None = None,
                     docs: list[Document] | None = None, msg: str = _ESCALATION_MSG) -> dict:
    return {
        "answer": msg,
        "stream": None,
        "sources": sources or [],
        "intent": intent,
        "escalated": True,
        "docs": docs or [],
    }


def answer(
    question: str,
    *,
    stream: bool = False,
    enable_fusion: bool = False,
    enable_crag: bool = True,
    enable_rerank: bool = True,
    chat_history: list | None = None,
    vectorstore: Any | None = None,
    llm: ChatGroq | None = None,
) -> dict:
    """Caminho único do property.ai.

    Args:
        question: pergunta do usuário.
        stream: se True, retorna gerador de tokens em `stream` (UI).
        enable_fusion: se True, usa RAG-Fusion (3 variações via LLM).
        enable_crag: se True, valida suficiência via LLM antes de responder.
        vectorstore: instância PropertyVectorStore. Se None, fusion carregará lazy.
        llm: ChatGroq override (testes).

    Returns:
        dict com chaves: answer, stream, sources, intent, escalated, docs.
        - Se stream=False: `answer` contém o texto completo, `stream` é None.
        - Se stream=True e há resposta gerada: `stream` é um Iterator[str],
          `answer` é None (UI consome o iterator e preenche).
        - Em escalation/sem-info: `answer` é mensagem fixa, `stream` é None.
    """
    intent = classify_intent(question)

    if intent == "escalonamento":
        return _build_escalation(intent)

    docs = _retrieve(question, vectorstore, llm, enable_fusion, enable_rerank)

    if not docs:
        return _build_escalation(intent, msg=_NO_INFO_MSG)

    if enable_crag and not evaluate_sufficiency(question, docs, llm=llm):
        return _build_escalation(
            intent,
            sources=_unique_sources(docs),
            docs=docs,
            msg=_NO_INFO_MSG,
        )

    context = docs_to_context(docs)
    sources = _unique_sources(docs)

    if stream:
        streaming_llm = _get_llm(streaming=True)
        return {
            "answer": None,
            "stream": _stream_tokens(streaming_llm, context, question),
            "sources": sources,
            "intent": intent,
            "escalated": False,
            "docs": docs,
        }

    used_llm = llm or _get_llm()
    chain = _ANSWER_PROMPT | used_llm | StrOutputParser()
    response = chain.invoke({"context": context, "question": question})

    return {
        "answer": response,
        "stream": None,
        "sources": sources,
        "intent": intent,
        "escalated": False,
        "docs": docs,
    }


def _stream_tokens(streaming_llm: ChatGroq, context: str, question: str) -> Iterator[str]:
    for chunk in (_ANSWER_PROMPT | streaming_llm).stream(
        {"context": context, "question": question}
    ):
        yield chunk.content
