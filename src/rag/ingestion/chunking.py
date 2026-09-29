"""Chunk step: split whatever the parsers didn't already cut to size.

Structure-aware parsers (Docling, tabular, code/JSON) emit `prechunked` docs and
pass through untouched. Free text (plain text, email bodies, OCR output) is
split by tokens, with a small header so each chunk still says where it's from.
"""
from __future__ import annotations

from functools import lru_cache

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

CHUNK_TOKENS = 512
OVERLAP_TOKENS = 64
MIN_CHARS = 20          # drop fragments like a lone page number


@lru_cache
def _splitter() -> RecursiveCharacterTextSplitter:
    return RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        encoding_name="cl100k_base", chunk_size=CHUNK_TOKENS, chunk_overlap=OVERLAP_TOKENS)


def chunk_documents(docs: list[Document]) -> list[Document]:
    out: list[Document] = []
    for doc in docs:
        if doc.metadata.get("prechunked"):
            out.append(doc)
            continue
        header = _header(doc)
        for part in _splitter().split_text(doc.page_content):
            out.append(Document(page_content=f"{header}\n{part}" if header else part,
                                metadata=dict(doc.metadata)))
    return [d for d in out if len(d.page_content.strip()) >= MIN_CHARS]


def _header(doc: Document) -> str:
    m = doc.metadata
    bits = [f"File: {m['filename']}"]
    if m.get("section_path"):
        bits.append(m["section_path"])
    return " | ".join(bits)
