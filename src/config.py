"""Configuração centralizada do property.ai.

Fonte única de verdade para variáveis de ambiente e parâmetros operacionais.
Substitui `os.getenv(...)` espalhado pelos módulos.

Uso:
    from src.config import get_settings
    settings = get_settings()
    model = settings.groq_model
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from functools import lru_cache


@dataclass(frozen=True)
class Settings:
    # LLM (Groq)
    groq_model: str = "llama-3.3-70b-versatile"
    groq_model_fast: str = "llama-3.1-8b-instant"
    groq_temperature: float = 0.1

    # Embeddings
    embedding_model: str = "intfloat/multilingual-e5-base"
    embedding_version: str = "v1"
    embedding_device: str = "cpu"

    # Vector store (Chroma)
    chroma_persist_dir: str = "./data/processed/embeddings"
    chroma_collection_name: str = "property_kb"

    # Retrieval
    retriever_top_k: int = 5
    rerank_top_k: int = 5

    # Cache (Sprint 4)
    cache_dir: str = "./data/cache"
    cache_ttl_seconds: int = 86_400  # 24h
    semantic_cache_threshold: float = 0.95

    # Logging (Sprint 6)
    log_level: str = "INFO"
    log_path: str = "./data/logs/turns.jsonl"

    # Paths fixos
    raw_dir: str = "./data/raw"
    processed_dir: str = "./data/processed"
    evaluation_dir: str = "./data/evaluation"


def _get_str(key: str, default: str) -> str:
    val = os.getenv(key)
    return val if val is not None and val != "" else default


def _get_int(key: str, default: int) -> int:
    val = os.getenv(key)
    if val is None or val == "":
        return default
    try:
        return int(val)
    except ValueError:
        return default


def _get_float(key: str, default: float) -> float:
    val = os.getenv(key)
    if val is None or val == "":
        return default
    try:
        return float(val)
    except ValueError:
        return default


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Retorna a instância única de Settings (cache em processo).

    Lê variáveis de ambiente uma única vez. Para forçar releitura
    (ex: testes), chame `get_settings.cache_clear()`.
    """
    defaults = Settings()
    return Settings(
        groq_model=_get_str("GROQ_MODEL", defaults.groq_model),
        groq_model_fast=_get_str("GROQ_MODEL_FAST", defaults.groq_model_fast),
        groq_temperature=_get_float("GROQ_TEMPERATURE", defaults.groq_temperature),
        embedding_model=_get_str("EMBEDDING_MODEL", defaults.embedding_model),
        embedding_version=_get_str("EMBEDDING_VERSION", defaults.embedding_version),
        embedding_device=_get_str("EMBEDDING_DEVICE", defaults.embedding_device),
        chroma_persist_dir=_get_str("CHROMA_PERSIST_DIR", defaults.chroma_persist_dir),
        chroma_collection_name=_get_str("CHROMA_COLLECTION_NAME", defaults.chroma_collection_name),
        retriever_top_k=_get_int("RETRIEVER_TOP_K", defaults.retriever_top_k),
        rerank_top_k=_get_int("RERANK_TOP_K", defaults.rerank_top_k),
        cache_dir=_get_str("CACHE_DIR", defaults.cache_dir),
        cache_ttl_seconds=_get_int("CACHE_TTL_SECONDS", defaults.cache_ttl_seconds),
        semantic_cache_threshold=_get_float(
            "SEMANTIC_CACHE_THRESHOLD", defaults.semantic_cache_threshold
        ),
        log_level=_get_str("LOG_LEVEL", defaults.log_level),
        log_path=_get_str("LOG_PATH", defaults.log_path),
        raw_dir=_get_str("RAW_DIR", defaults.raw_dir),
        processed_dir=_get_str("PROCESSED_DIR", defaults.processed_dir),
        evaluation_dir=_get_str("EVALUATION_DIR", defaults.evaluation_dir),
    )
