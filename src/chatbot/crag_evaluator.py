import os

from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq

_EVAL_PROMPT = ChatPromptTemplate.from_template(
    "Você é um avaliador de qualidade de recuperação de informação.\n"
    "Avalie se os chunks abaixo contêm informação EXATA e SUFICIENTE para responder "
    "à pergunta, sem depender de conhecimento externo.\n\n"
    "<chunks>\n{context}\n</chunks>\n\n"
    "<query>\n{question}\n</query>\n\n"
    "Responda apenas: SUFICIENTE ou INSUFICIENTE"
)


def _get_llm() -> ChatGroq:
    return ChatGroq(
        model=os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile"),
        temperature=0,
    )


def docs_to_context(docs: list[Document]) -> str:
    return "\n\n---\n\n".join(
        f"[Fonte: {doc.metadata.get('fonte', 'N/A')} | "
        f"Tipo: {doc.metadata.get('tipo', 'N/A')}]\n{doc.page_content}"
        for doc in docs
    )


def evaluate_sufficiency(
    question: str,
    docs: list[Document],
    llm: ChatGroq | None = None,
) -> bool:
    if not docs:
        return False
    llm = llm or _get_llm()
    chain = _EVAL_PROMPT | llm | StrOutputParser()
    context = docs_to_context(docs)
    result = chain.invoke({"question": question, "context": context})
    upper = result.upper()
    return "SUFICIENTE" in upper and "INSUFICIENTE" not in upper
