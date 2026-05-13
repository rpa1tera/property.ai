import os
from pathlib import Path

import chromadb
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings

_COLLECTION_NAME = "property_kb"
_BATCH = 100
_ALLOWED_META_TYPES = (str, int, float, bool)


def get_embeddings() -> HuggingFaceEmbeddings:
    model_name = os.getenv("EMBEDDING_MODEL", "intfloat/multilingual-e5-base")
    return HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )


class _HFChromaEF:
    """Passa nosso embedder HuggingFace ao ChromaDB.
    Impede a inicialização do ONNXMiniLM_L6_V2 (default) que falha no Windows.
    """
    def name(self) -> str:
        return "hf-multilingual-e5-base"

    def __init__(self, hf_emb: HuggingFaceEmbeddings) -> None:
        self._hf = hf_emb

    def __call__(self, input: list[str]) -> list[list[float]]:
        return self._hf.embed_documents(list(input))


def _sanitize_metadata(meta: dict) -> dict:
    """Remove None e tipos não suportados; garante dict nunca vazio."""
    clean = {
        k: v for k, v in meta.items()
        if v is not None and isinstance(v, _ALLOWED_META_TYPES)
    }
    return clean if clean else {"_": "ok"}


class PropertyVectorStore:
    """Wrapper sobre chromadb.Collection com interface LangChain."""

    def __init__(self, collection, hf_emb: HuggingFaceEmbeddings) -> None:
        self._col = collection
        self._emb = hf_emb

    def similarity_search(self, query: str, k: int = 5) -> list[Document]:
        q_vec = self._emb.embed_query(query)
        results = self._col.query(
            query_embeddings=[q_vec],
            n_results=min(k, self._col.count()),
            include=["documents", "metadatas"],
        )
        return [
            Document(page_content=text, metadata=meta or {})
            for text, meta in zip(results["documents"][0], results["metadatas"][0])
        ]

    def as_retriever(self, search_kwargs: dict | None = None):
        k = (search_kwargs or {}).get("k", 5)
        vs = self

        class _Retriever:
            def invoke(self_, query: str) -> list[Document]:
                return vs.similarity_search(query, k=k)

        return _Retriever()

    def count(self) -> int:
        return self._col.count()


def _get_client(persist_dir: str) -> chromadb.ClientAPI:
    return chromadb.PersistentClient(path=persist_dir)


def build_vectorstore(
    docs: list[Document],
    persist_dir: str | None = None,
) -> PropertyVectorStore:
    persist_dir = persist_dir or os.getenv("CHROMA_PERSIST_DIR", "./data/processed/embeddings")
    Path(persist_dir).mkdir(parents=True, exist_ok=True)

    hf_emb = get_embeddings()
    chroma_ef = _HFChromaEF(hf_emb)
    client = _get_client(persist_dir)

    try:
        client.delete_collection(_COLLECTION_NAME)
    except Exception:
        pass

    col = client.create_collection(
        _COLLECTION_NAME,
        embedding_function=chroma_ef,
        metadata={"hnsw:space": "cosine"},
    )

    total = len(docs)
    for start in range(0, total, _BATCH):
        batch = docs[start:start + _BATCH]
        texts = [d.page_content for d in batch]
        col.add(
            embeddings=hf_emb.embed_documents(texts),
            documents=texts,
            metadatas=[_sanitize_metadata(d.metadata) for d in batch],
            ids=[f"chunk_{start + j}" for j in range(len(batch))],
        )
        print(f"  indexado {min(start + _BATCH, total)}/{total}", flush=True)

    return PropertyVectorStore(col, hf_emb)


def load_vectorstore(persist_dir: str | None = None) -> PropertyVectorStore:
    persist_dir = persist_dir or os.getenv("CHROMA_PERSIST_DIR", "./data/processed/embeddings")
    hf_emb = get_embeddings()
    chroma_ef = _HFChromaEF(hf_emb)
    client = _get_client(persist_dir)
    col = client.get_collection(_COLLECTION_NAME, embedding_function=chroma_ef)
    return PropertyVectorStore(col, hf_emb)
