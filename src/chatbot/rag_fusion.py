import os
from typing import Any

from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq

from src.rag.embeddings import load_vectorstore

_FUSION_PROMPT = ChatPromptTemplate.from_template(
    "Gere 3 variações semânticas da pergunta abaixo para melhorar a busca em documentos "
    "de seguros patrimoniais. Sem numeração, uma por linha, apenas as variações.\n\n"
    "Pergunta: {question}"
)


def _get_llm() -> ChatGroq:
    return ChatGroq(
        model=os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile"),
        temperature=0,
    )


def generate_variations(question: str, llm: ChatGroq | None = None) -> list[str]:
    llm = llm or _get_llm()
    chain = _FUSION_PROMPT | llm | StrOutputParser()
    raw = chain.invoke({"question": question})
    lines = [ln.strip() for ln in raw.strip().split("\n") if ln.strip()]
    return lines[:3]


def fused_retrieval(
    question: str,
    vectorstore: Any | None = None,
    top_k: int = 5,
    llm: ChatGroq | None = None,
) -> list[Document]:
    if vectorstore is None:
        vectorstore = load_vectorstore()

    variations = generate_variations(question, llm)
    queries = [question] + variations

    seen: set[str] = set()
    fused: list[Document] = []

    for query in queries:
        docs = vectorstore.similarity_search(query, k=top_k)
        for doc in docs:
            fingerprint = doc.page_content[:150]
            if fingerprint not in seen:
                seen.add(fingerprint)
                fused.append(doc)

    return fused[:10]
