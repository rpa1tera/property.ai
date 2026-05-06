import os
from typing import Any

from langchain_core.documents import Document

from src.rag.embeddings import load_vectorstore


def get_retriever(vectorstore: Any | None = None, top_k: int | None = None):
    if vectorstore is None:
        vectorstore = load_vectorstore()
    k = top_k or int(os.getenv("RETRIEVER_TOP_K", "5"))
    return vectorstore.as_retriever(search_kwargs={"k": k})


def search(
    query: str,
    vectorstore: Any | None = None,
    top_k: int | None = None,
) -> list[Document]:
    retriever = get_retriever(vectorstore, top_k)
    return retriever.invoke(query)
