"""Chunking estrutural com fallback recursivo.

Apólices e manuais regulatórios são organizados por cláusulas / artigos / itens
numerados. Chunking puramente lexical quebra cláusulas no meio e prejudica
retrieval. Estratégia:

1. Detecta marcadores estruturais via regex (Cláusula, Artigo, Art., 1.1, Item N).
2. Se ≥2 marcadores: corta no marcador, cada cláusula vira um chunk
   (com sub-split se exceder MAX_STRUCTURAL_CHARS).
3. Senão: fallback para RecursiveCharacterTextSplitter (512/64).
"""
from __future__ import annotations

import re

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

_STRUCTURAL_MARKER = re.compile(
    r"(?m)^\s*(?:Cl[áa]usula\s+\w+|Artigo\s+\d+|Art\.\s*\d+|\d+\.\d+(?:\.\d+)?\s|Item\s+\d+)\b"
)

MAX_STRUCTURAL_CHARS = 1024


def _split_structural(text: str) -> list[str] | None:
    """Retorna lista de cláusulas se houver ≥2 marcadores; senão None."""
    matches = list(_STRUCTURAL_MARKER.finditer(text))
    if len(matches) < 2:
        return None
    starts = [m.start() for m in matches] + [len(text)]
    raw = [text[starts[i]:starts[i + 1]].strip() for i in range(len(matches))]
    return [c for c in raw if c]


def _further_split(chunk: str) -> list[str]:
    """Se cláusula passar do máximo, usa splitter recursivo dentro dela."""
    if len(chunk) <= MAX_STRUCTURAL_CHARS:
        return [chunk]
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=MAX_STRUCTURAL_CHARS,
        chunk_overlap=64,
        separators=[". ", "! ", "? ", ";\n", "\n\n", "\n", " ", ""],
    )
    return splitter.split_text(chunk)


def chunk_documents(
    docs: list[Document],
    chunk_size: int = 512,
    overlap: int = 64,
) -> list[Document]:
    recursive = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=overlap,
        separators=[". ", "! ", "? ", ";\n", "\n\n", "\n", " ", ""],
        length_function=len,
    )

    out: list[Document] = []
    for doc in docs:
        structural = _split_structural(doc.page_content)
        if structural is not None:
            pieces: list[str] = []
            for cl in structural:
                pieces.extend(_further_split(cl))
            strategy = "structural"
        else:
            pieces = [c.page_content for c in recursive.split_documents([doc])]
            strategy = "recursive"

        for i, piece in enumerate(pieces):
            meta = dict(doc.metadata)
            meta["chunk_index"] = i
            meta["chunk_strategy"] = strategy
            out.append(Document(page_content=piece, metadata=meta))
    return out
