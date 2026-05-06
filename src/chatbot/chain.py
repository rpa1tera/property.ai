import os
from typing import Any

from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq

from src.chatbot.crag_evaluator import docs_to_context, evaluate_sufficiency
from src.chatbot.intents import classify_intent
from src.chatbot.rag_fusion import fused_retrieval

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


def _get_llm() -> ChatGroq:
    return ChatGroq(
        model=os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile"),
        temperature=0.1,
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


def answer(
    question: str,
    chat_history: list | None = None,
    vectorstore: Any | None = None,
    llm: ChatGroq | None = None,
) -> dict:
    """
    Returns dict: {answer, sources, intent, escalated, docs}
    """
    intent = classify_intent(question)

    if intent == "escalonamento":
        return {
            "answer": _ESCALATION_MSG,
            "sources": [],
            "intent": intent,
            "escalated": True,
            "docs": [],
        }

    llm = llm or _get_llm()
    docs = fused_retrieval(question, vectorstore=vectorstore, top_k=5, llm=llm)

    if not evaluate_sufficiency(question, docs, llm=llm):
        return {
            "answer": _NO_INFO_MSG,
            "sources": _unique_sources(docs),
            "intent": intent,
            "escalated": True,
            "docs": docs,
        }

    context = docs_to_context(docs)
    chain = _ANSWER_PROMPT | llm | StrOutputParser()
    response = chain.invoke({"context": context, "question": question})

    return {
        "answer": response,
        "sources": _unique_sources(docs),
        "intent": intent,
        "escalated": False,
        "docs": docs,
    }
