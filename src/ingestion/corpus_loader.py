import os
from pathlib import Path

import fitz  # PyMuPDF
import pandas as pd
from langchain_core.documents import Document
from langchain_community.document_loaders import BSHTMLLoader


def _tipo_from_path(path: Path) -> str:
    parts = [p.lower() for p in path.parts]
    if any(p in parts for p in ("faqs", "faq")):
        return "faq"
    if any(p in parts for p in ("apolices", "apolice")):
        return "apolice"
    if any(p in parts for p in ("manuais", "manual")):
        return "manual"

    name = path.stem.lower()
    if any(k in name for k in ("faq", "pergunta", "resposta")):
        return "faq"
    if any(k in name for k in ("resseguro", "manual", "introducao", "introdução", "fundacion", "fundação", "guia")):
        return "manual"
    if any(k in name for k in ("cg_", "condicoes", "condições", "apolice", "apólice", "rn_", "clausula")):
        return "apolice"
    return "apolice"


def load_file(file_path: str, tipo: str | None = None) -> list[Document]:
    path = Path(file_path)
    ext = path.suffix.lower()
    resolved_tipo = tipo or _tipo_from_path(path)
    meta_base = {"tipo": resolved_tipo, "fonte": path.name}

    if ext == ".pdf":
        return _load_pdf(path, meta_base)
    elif ext in (".html", ".htm"):
        return _load_html(path, meta_base)
    elif ext == ".csv":
        return _load_csv(path, meta_base)
    elif ext == ".json":
        return _load_json(path, meta_base)
    else:
        raise ValueError(f"Formato não suportado: {ext}")


def _load_pdf(path: Path, meta_base: dict) -> list[Document]:
    docs = []
    with fitz.open(str(path)) as pdf:
        for page_num, page in enumerate(pdf, start=1):
            text = page.get_text()
            if text.strip():
                docs.append(Document(
                    page_content=text,
                    metadata={**meta_base, "pagina": page_num},
                ))
    return docs


def _load_html(path: Path, meta_base: dict) -> list[Document]:
    loader = BSHTMLLoader(str(path), open_encoding="utf-8")
    docs = loader.load()
    for doc in docs:
        doc.metadata.update(meta_base)
    return docs


def _load_csv(path: Path, meta_base: dict) -> list[Document]:
    df = pd.read_csv(path)
    required = {"pergunta", "resposta"}
    if not required.issubset(df.columns):
        raise ValueError(f"CSV deve ter colunas 'pergunta' e 'resposta': {path.name}")
    df = df.dropna(subset=["pergunta", "resposta"])
    df = df.drop_duplicates(subset=["pergunta"])
    df = df[df["resposta"].str.split().str.len() >= 10]
    docs = []
    for _, row in df.iterrows():
        content = f"Pergunta: {row['pergunta']}\nResposta: {row['resposta']}"
        docs.append(Document(page_content=content, metadata={**meta_base}))
    return docs


def _load_json(path: Path, meta_base: dict) -> list[Document]:
    import json
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    items = data if isinstance(data, list) else [data]
    docs = []
    for item in items:
        content = json.dumps(item, ensure_ascii=False) if not isinstance(item, str) else item
        docs.append(Document(page_content=content, metadata={**meta_base}))
    return docs


def load_directory(directory: str, tipo: str | None = None) -> list[Document]:
    supported = {".pdf", ".html", ".htm", ".csv", ".json"}
    docs = []
    for root, _, files in os.walk(directory):
        for fname in sorted(files):
            if Path(fname).suffix.lower() in supported:
                fpath = os.path.join(root, fname)
                docs.extend(load_file(fpath, tipo=tipo))
    return docs
