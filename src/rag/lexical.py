"""Busca lexical via BM25.

Constrói índice em memória a partir da coleção Chroma. Persiste em pickle.
Função pública: `bm25_search(query, k)`.
"""
from __future__ import annotations

import pickle
import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from langchain_core.documents import Document
from rank_bm25 import BM25Okapi

from src.config import get_settings
from src.rag.embeddings import load_vectorstore

_TOKEN_RE = re.compile(r"\w+", re.UNICODE)


def _normalize(text: str) -> str:
    """Lower + remoção de acentos para casamento robusto em PT-BR."""
    nfkd = unicodedata.normalize("NFKD", text.lower())
    return "".join(c for c in nfkd if not unicodedata.combining(c))


def tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall(_normalize(text))


@dataclass
class BM25Index:
    bm25: BM25Okapi
    documents: list[Document]


def _index_path() -> Path:
    return Path(get_settings().processed_dir) / "bm25.pkl"


def build_index(vectorstore: Any | None = None, persist: bool = True) -> BM25Index:
    """Lê todos os chunks do Chroma e constrói o índice BM25."""
    vs = vectorstore or load_vectorstore()
    col = vs._col  # acesso direto: chromadb.Collection
    raw = col.get(include=["documents", "metadatas"])

    documents = [
        Document(page_content=text, metadata=meta or {})
        for text, meta in zip(raw["documents"], raw["metadatas"])
    ]
    corpus_tokens = [tokenize(d.page_content) for d in documents]
    bm25 = BM25Okapi(corpus_tokens)
    index = BM25Index(bm25=bm25, documents=documents)

    if persist:
        path = _index_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("wb") as f:
            pickle.dump(index, f)

    return index


def load_index() -> BM25Index | None:
    path = _index_path()
    if not path.exists():
        return None
    with path.open("rb") as f:
        return pickle.load(f)


def get_or_build_index(vectorstore: Any | None = None) -> BM25Index:
    idx = load_index()
    if idx is None:
        idx = build_index(vectorstore=vectorstore, persist=True)
    return idx


def bm25_search(
    query: str,
    k: int = 10,
    index: BM25Index | None = None,
    vectorstore: Any | None = None,
) -> list[Document]:
    """Top-k documentos por score BM25."""
    idx = index or get_or_build_index(vectorstore=vectorstore)
    scores = idx.bm25.get_scores(tokenize(query))
    if len(scores) == 0:
        return []
    top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]
    return [idx.documents[i] for i in top_indices if scores[i] > 0]
