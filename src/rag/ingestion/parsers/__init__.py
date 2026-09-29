"""Parser registry: file type -> function that turns a file into Documents.

Every parser has the signature `parse(path, detection, src, ctx) -> list[Document]`
and emits Documents built with `schema.make_doc`, so the rest of the pipeline
never needs to know which format a chunk came from.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

from langchain_core.documents import Document

from rag.config import Settings
from rag.ingestion.detect import Detection
from rag.ingestion.schema import FileType, SourceInfo


@dataclass
class ParseContext:
    settings: Settings
    depth: int = 0                       # email attachment nesting level
    max_depth: int = 2
    dry_run: bool = False                # count OCR pages but never call the vision model
    stats: dict = field(default_factory=lambda: {"ocr_pages": 0})


Parser = Callable[[Path, Detection, SourceInfo, ParseContext], list[Document]]


class UnsupportedFile(ValueError):
    pass


def route(det: Detection) -> str:
    """Name of the parser node for a detected file (matches the ingestion graph)."""
    if det.file_type is FileType.pdf:
        return "ocr_vision" if det.fully_scanned else "pdf_text"
    return {
        FileType.image: "ocr_vision",
        FileType.docx: "office",
        FileType.pptx: "office",
        FileType.xlsx: "tabular",
        FileType.csv: "tabular",
        FileType.email: "email",
        FileType.html: "web_markup",
        FileType.markdown: "web_markup",
        FileType.json: "code_json",
        FileType.code: "code_json",
        FileType.text: "fallback",
    }.get(det.file_type, "unsupported")


def get_parser(name: str) -> Parser:
    # Imported lazily: Docling and the vision client are heavy to load.
    if name == "pdf_text":
        from rag.ingestion.parsers.docling_parser import parse_pdf
        return parse_pdf
    if name == "ocr_vision":
        from rag.ingestion.parsers.ocr_vision import parse_scanned
        return parse_scanned
    if name == "office":
        from rag.ingestion.parsers.docling_parser import parse_office
        return parse_office
    if name == "web_markup":
        from rag.ingestion.parsers.docling_parser import parse_web
        return parse_web
    if name == "tabular":
        from rag.ingestion.parsers.tabular import parse_tabular
        return parse_tabular
    if name == "email":
        from rag.ingestion.parsers.email_parser import parse_email
        return parse_email
    if name == "code_json":
        from rag.ingestion.parsers.code_json import parse_code_or_json
        return parse_code_or_json
    if name == "fallback":
        from rag.ingestion.parsers.text import parse_text
        return parse_text
    raise UnsupportedFile(f"no parser for route '{name}'")


def parse_file(path: Path, det: Detection, src: SourceInfo, ctx: ParseContext) -> list[Document]:
    """Detect-and-dispatch, used for nested files such as email attachments."""
    return get_parser(route(det))(path, det, src, ctx)
