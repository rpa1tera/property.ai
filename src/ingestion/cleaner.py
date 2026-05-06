import re

import ftfy
from langchain_core.documents import Document

_HYPHEN_EOL = re.compile(r"(\w+)-\n(\w+)")
_MULTI_SPACE = re.compile(r"[ \t]{2,}")
_CTRL_CHARS = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
_PAGE_NUM_LINE = re.compile(r"^\s*\d{1,4}\s*$", re.MULTILINE)
_SOFT_LINE_JOIN = re.compile(r"(?<![.!?:;])\n(?=[a-záéíóúàâêôãõçüA-ZÁÉÍÓÚÀÂÊÔÃÕÇÜ])")

_BOILERPLATE = [
    "todos os direitos reservados",
    "fundación mapfre",
    "este documento é confidencial",
    "uso exclusivo do corretor",
    "circular susep",
    "processo susep",
]


def clean_text(text: str) -> str:
    text = ftfy.fix_text(text)
    text = _CTRL_CHARS.sub("", text)
    text = _HYPHEN_EOL.sub(r"\1\2", text)
    text = _PAGE_NUM_LINE.sub("", text)
    text = _SOFT_LINE_JOIN.sub(" ", text)
    text = _MULTI_SPACE.sub(" ", text)

    lower = text.lower()
    for phrase in _BOILERPLATE:
        idx = lower.find(phrase)
        while idx != -1:
            line_start = text.rfind("\n", 0, idx)
            line_end = text.find("\n", idx + len(phrase))
            if line_start == -1:
                line_start = 0
            if line_end == -1:
                line_end = len(text)
            text = text[:line_start] + text[line_end:]
            lower = text.lower()
            idx = lower.find(phrase)

    return text.strip()


def clean_documents(docs: list[Document]) -> list[Document]:
    cleaned = []
    for doc in docs:
        text = clean_text(doc.page_content)
        if len(text.split()) >= 5:
            cleaned.append(Document(page_content=text, metadata=doc.metadata))
    return cleaned
