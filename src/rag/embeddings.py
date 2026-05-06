"""
Vector store backed by FAISS (langchain_community.vectorstores.FAISS).
Uses intfloat/multilingual-e5-base for embeddings.

Note: ChromaDB was the original choice but its native HNSW library crashes
on this Windows setup (STATUS_ACCESS_VIOLATION in the C++ DLL).
FAISS is identical in behavior — same LangChain interface, auto-persist via
save_local/load_local.
"""
import os
from pathlib import Path

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings

_INDEX_DIR = "faiss_index"


def get_embeddings() -> HuggingFaceEmbeddings:
    model_name = os.getenv("EMBEDDING_MODEL", "intfloat/multilingual-e5-base")
    return HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )


def build_vectorstore(
    docs: list[Document],
    persist_dir: str | None = None,
) -> FAISS:
    persist_dir = persist_dir or os.getenv("CHROMA_PERSIST_DIR", "./data/processed/embeddings")
    Path(persist_dir).mkdir(parents=True, exist_ok=True)

    emb = get_embeddings()
    vectorstore = FAISS.from_documents(docs, emb)
    vectorstore.save_local(folder_path=persist_dir, index_name=_INDEX_DIR)
    return vectorstore


def load_vectorstore(persist_dir: str | None = None) -> FAISS:
    persist_dir = persist_dir or os.getenv("CHROMA_PERSIST_DIR", "./data/processed/embeddings")
    emb = get_embeddings()
    return FAISS.load_local(
        folder_path=persist_dir,
        embeddings=emb,
        index_name=_INDEX_DIR,
        allow_dangerous_deserialization=True,
    )
