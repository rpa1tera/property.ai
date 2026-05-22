import hashlib
from datetime import datetime, timezone
from pathlib import Path

import chromadb
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings

from src.config import get_settings

_BATCH = 100
_ALLOWED_META_TYPES = (str, int, float, bool)


def stable_chunk_id(meta: dict, content: str) -> str:
    """ID determinístico: sha1(fonte|pagina|chunk_index|first_64_chars)[:16].

    Permite reindexação incremental: o mesmo chunk gera o mesmo id em runs
    diferentes, mas mudar o conteúdo gera id novo (porque first_64_chars muda).
    """
    fonte = str(meta.get("fonte", ""))
    pagina = str(meta.get("pagina", ""))
    chunk_index = str(meta.get("chunk_index", ""))
    head = content[:64]
    key = f"{fonte}|{pagina}|{chunk_index}|{head}"
    return hashlib.sha1(key.encode("utf-8")).hexdigest()[:16]


def get_embeddings() -> HuggingFaceEmbeddings:
    s = get_settings()
    return HuggingFaceEmbeddings(
        model_name=s.embedding_model,
        model_kwargs={"device": s.embedding_device},
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


def _enrich_metadata(meta: dict, settings) -> dict:
    """Adiciona campos de versionamento ao metadata do chunk."""
    enriched = dict(meta)
    enriched.setdefault("embedding_model", settings.embedding_model)
    enriched.setdefault("embedding_version", settings.embedding_version)
    enriched.setdefault("indexed_at", datetime.now(timezone.utc).isoformat(timespec="seconds"))
    enriched.setdefault("chunk_strategy", meta.get("chunk_strategy", "recursive"))
    return enriched


def build_vectorstore(
    docs: list[Document],
    persist_dir: str | None = None,
    collection_name: str | None = None,
    rebuild: bool = False,
) -> PropertyVectorStore:
    """Indexa chunks no ChromaDB.

    Args:
        docs: chunks já produzidos por chunk_documents().
        persist_dir / collection_name: override do config.
        rebuild: se True, apaga a coleção antes de recriar (comportamento antigo).
                 Se False (default), usa get_or_create_collection — preserva chunks
                 existentes e faz upsert por id estável.
    """
    s = get_settings()
    persist_dir = persist_dir or s.chroma_persist_dir
    collection_name = collection_name or s.chroma_collection_name
    Path(persist_dir).mkdir(parents=True, exist_ok=True)

    hf_emb = get_embeddings()
    chroma_ef = _HFChromaEF(hf_emb)
    client = _get_client(persist_dir)

    if rebuild:
        try:
            client.delete_collection(collection_name)
        except Exception:
            pass

    col = client.get_or_create_collection(
        collection_name,
        embedding_function=chroma_ef,
        metadata={"hnsw:space": "cosine"},
    )

    total = len(docs)
    for start in range(0, total, _BATCH):
        batch = docs[start:start + _BATCH]
        texts = [d.page_content for d in batch]
        metas = [_enrich_metadata(d.metadata, s) for d in batch]
        ids = [stable_chunk_id(metas[j], texts[j]) for j in range(len(batch))]
        col.upsert(
            embeddings=hf_emb.embed_documents(texts),
            documents=texts,
            metadatas=[_sanitize_metadata(m) for m in metas],
            ids=ids,
        )
        print(f"  indexado {min(start + _BATCH, total)}/{total}", flush=True)

    return PropertyVectorStore(col, hf_emb)


def load_vectorstore(
    persist_dir: str | None = None,
    collection_name: str | None = None,
) -> PropertyVectorStore:
    s = get_settings()
    persist_dir = persist_dir or s.chroma_persist_dir
    collection_name = collection_name or s.chroma_collection_name
    hf_emb = get_embeddings()
    chroma_ef = _HFChromaEF(hf_emb)
    client = _get_client(persist_dir)
    col = client.get_collection(collection_name, embedding_function=chroma_ef)
    return PropertyVectorStore(col, hf_emb)
