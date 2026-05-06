"""
Script de ingestão: carrega os PDFs de data/raw/, limpa, chunka e indexa no ChromaDB.

Uso:
    python build_index.py
    python build_index.py --raw-dir data/raw --persist-dir data/processed/embeddings

Execute uma vez antes de iniciar o chatbot. O índice é persistido em persist_dir/.
"""

import argparse
import sys
from pathlib import Path

# Garante que a raiz do projeto está no path
sys.path.insert(0, str(Path(__file__).parent))

from dotenv import load_dotenv
load_dotenv()

from src.ingestion.corpus_loader import load_file
from src.ingestion.cleaner import clean_documents
from src.ingestion.chunker import chunk_documents
from src.rag.embeddings import build_vectorstore

# Mapeamento explícito: filename -> tipo (para PDFs que estão na raiz de raw/)
_TIPO_OVERRIDE: dict[str, str] = {
    "CG_RN_96_V3.2 - MAPFRE SEGUROS.pdf": "apolice",
    "Introduçao ao Resseguro  Fundación MAPFRE.pdf": "manual",
}


def main(raw_dir: str = "data/raw", persist_dir: str = "data/processed/embeddings") -> None:
    raw_path = Path(raw_dir)
    supported = {".pdf", ".html", ".htm", ".csv", ".json"}

    pdf_files = sorted(
        p for p in raw_path.rglob("*")
        if p.suffix.lower() in supported and p.is_file()
    )

    if not pdf_files:
        print(f"[build_index] Nenhum arquivo encontrado em {raw_dir}")
        sys.exit(1)

    print(f"[build_index] {len(pdf_files)} arquivo(s) encontrado(s):")
    all_docs = []

    for fpath in pdf_files:
        tipo = _TIPO_OVERRIDE.get(fpath.name)
        print(f"  Carregando: {fpath.name} (tipo={tipo or 'auto'})")
        docs = load_file(str(fpath), tipo=tipo)
        print(f"    -> {len(docs)} página(s)/entrada(s) extraída(s)")
        all_docs.extend(docs)

    print(f"\n[build_index] Total bruto: {len(all_docs)} documentos")

    cleaned = clean_documents(all_docs)
    print(f"[build_index] Após limpeza: {len(cleaned)} documentos")

    chunks = chunk_documents(cleaned, chunk_size=512, overlap=64)
    print(f"[build_index] Chunks gerados: {len(chunks)}")

    print(f"\n[build_index] Criando embeddings e indexando no ChromaDB ({persist_dir})...")
    print("  (primeira execução baixa ~280MB do modelo multilingual-e5-base)")

    build_vectorstore(chunks, persist_dir=persist_dir)
    print(f"[build_index] Indexação concluída: {len(chunks)} chunks")
    print(f"[build_index] Índice persistido em: {persist_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Constrói o índice ChromaDB a partir dos documentos raw.")
    parser.add_argument("--raw-dir", default="data/raw", help="Diretório com documentos brutos")
    parser.add_argument("--persist-dir", default="data/processed/embeddings", help="Diretório de persistência do ChromaDB")
    args = parser.parse_args()
    main(args.raw_dir, args.persist_dir)
