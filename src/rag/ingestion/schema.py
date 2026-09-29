"""The one metadata schema every parser emits, plus stable chunk IDs.

Whatever the file format, a chunk leaves the parsers as a LangChain
`Document` whose metadata has these keys. Downstream code (chunking, indexing,
retrieval filters, citations) relies only on this schema.
"""
from __future__ import annotations

import enum
import hashlib
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone

from langchain_core.documents import Document

# Fixed namespace so the same (tenant, source, content) always maps to the same ID.
_NS = uuid.UUID("6f1c2a52-8d0e-4c1b-9a51-1f3d7e2b9c40")


class FileType(str, enum.Enum):
    pdf = "pdf"
    image = "image"
    docx = "docx"
    pptx = "pptx"
    xlsx = "xlsx"
    csv = "csv"
    html = "html"
    markdown = "markdown"
    json = "json"
    code = "code"
    email = "email"
    text = "text"
    unknown = "unknown"


@dataclass(frozen=True)
class SourceInfo:
    """Identity of the file being ingested; copied into every chunk."""
    tenant: str
    source_uri: str
    filename: str
    sha256: str
    mime: str | None
    file_type: FileType

    @property
    def doc_id(self) -> str:
        # Stable across versions of the same file, so a re-upload replaces it.
        return str(uuid.uuid5(_NS, f"{self.tenant}|{self.source_uri}"))


def make_doc(text: str, src: SourceInfo, *, parser: str, page: int | None = None,
             section_path: str = "", ocr: bool = False, prechunked: bool = False,
             **extra) -> Document:
    return Document(
        page_content=text,
        metadata={
            "doc_id": src.doc_id,
            "tenant": src.tenant,
            "source_uri": src.source_uri,
            "filename": src.filename,
            "sha256": src.sha256,
            "mime": src.mime,
            "file_type": src.file_type.value,
            "parser": parser,
            "page": page,
            "section_path": section_path,
            "ocr": ocr,
            # Parsers that already produce retrieval-sized chunks (Docling,
            # tabular, code) set this so the chunk step leaves them alone.
            "prechunked": prechunked,
            **extra,
        },
    )


def assign_chunk_ids(chunks: list[Document]) -> list[str]:
    """Content-derived IDs: an unchanged chunk keeps its ID when the file is re-ingested.

    Also stamps `chunk_index` and `ingested_at`. Identical text appearing twice
    in one document gets distinct IDs via an occurrence counter.
    """
    now = datetime.now(timezone.utc).isoformat()
    seen: dict[str, int] = {}
    ids = []
    for i, doc in enumerate(chunks):
        m = doc.metadata
        digest = hashlib.sha1(doc.page_content.encode("utf-8")).hexdigest()
        n = seen.get(digest, 0)
        seen[digest] = n + 1
        ids.append(str(uuid.uuid5(_NS, f"{m['tenant']}|{m['source_uri']}|{digest}|{n}")))
        m["chunk_index"] = i
        m["ingested_at"] = now
        m.pop("prechunked", None)
    return ids
