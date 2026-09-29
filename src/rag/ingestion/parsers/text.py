"""Plain text: read with a tolerant decoder; chunked later by the chunk step."""
from __future__ import annotations

from pathlib import Path

from langchain_core.documents import Document

from rag.ingestion.detect import Detection
from rag.ingestion.schema import SourceInfo, make_doc


def read_text(path: Path) -> str:
    raw = path.read_bytes()
    for encoding in ("utf-8-sig", "cp1252"):
        try:
            return raw.decode(encoding)
        except UnicodeDecodeError:
            continue
    return raw.decode("latin-1")   # never fails; last resort


def parse_text(path: Path, det: Detection, src: SourceInfo, ctx) -> list[Document]:
    text = read_text(path).strip()
    return [make_doc(text, src, parser="fallback")] if text else []
