from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter


def chunk_documents(
    docs: list[Document],
    chunk_size: int = 512,
    overlap: int = 64,
) -> list[Document]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=overlap,
        separators=[". ", "! ", "? ", ";\n", "\n\n", "\n", " ", ""],
        length_function=len,
    )
    chunks = []
    for doc in docs:
        splits = splitter.split_documents([doc])
        for i, chunk in enumerate(splits):
            chunk.metadata["chunk_index"] = i
            chunks.append(chunk)
    return chunks
