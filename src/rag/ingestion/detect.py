"""Detect a file's real type from its bytes, and find scanned PDF pages.

Extensions lie (a `.pdf` that is really HTML, a `.txt` that is JSON), so binary
formats are identified by magic bytes first. Text formats have no magic bytes;
for those, content sniffing wins over the extension where it can.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import filetype

from rag.ingestion.schema import FileType

# A page with fewer extractable characters than this has no usable text layer.
MIN_TEXT_CHARS_PER_PAGE = 25

_BINARY_MIME = {
    "application/pdf": FileType.pdf,
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document": FileType.docx,
    "application/vnd.openxmlformats-officedocument.presentationml.presentation": FileType.pptx,
    "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet": FileType.xlsx,
}

_EXT = {
    ".md": FileType.markdown, ".markdown": FileType.markdown,
    ".html": FileType.html, ".htm": FileType.html,
    ".csv": FileType.csv, ".tsv": FileType.csv,
    ".json": FileType.json, ".jsonl": FileType.json,
    ".eml": FileType.email, ".msg": FileType.email,
    ".txt": FileType.text, ".log": FileType.text, ".rst": FileType.text,
    ".docx": FileType.docx, ".pptx": FileType.pptx, ".xlsx": FileType.xlsx,
}

CODE_EXTENSIONS = {
    ".js", ".mjs", ".cjs", ".jsx", ".ts", ".tsx", ".py", ".java", ".go", ".rb", ".rs",
    ".c", ".h", ".cpp", ".hpp", ".cs", ".php", ".kt", ".swift", ".scala", ".sol", ".lua",
    ".sh", ".sql", ".yaml", ".yml",
}


@dataclass
class Detection:
    file_type: FileType
    mime: str | None
    page_count: int | None = None
    scanned_pages: list[int] = field(default_factory=list)   # 1-based page numbers

    @property
    def fully_scanned(self) -> bool:
        return bool(self.page_count) and len(self.scanned_pages) == self.page_count


def detect(path: Path) -> Detection:
    kind = filetype.guess(str(path))
    if kind is not None:
        if kind.mime in _BINARY_MIME:
            ft = _BINARY_MIME[kind.mime]
            if ft is FileType.pdf:
                return _inspect_pdf(path, kind.mime)
            return Detection(ft, kind.mime)
        if kind.mime.startswith("image/"):
            return Detection(FileType.image, kind.mime)
        if kind.mime == "application/zip" and path.suffix.lower() in (".docx", ".pptx", ".xlsx"):
            return Detection(_EXT[path.suffix.lower()], kind.mime)

    ext = path.suffix.lower()
    if ext == ".msg":   # Outlook: OLE container, no reliable magic in `filetype`
        return Detection(FileType.email, "application/vnd.ms-outlook")

    head = _read_head(path)
    if head is None:
        return Detection(FileType.unknown, kind.mime if kind else None)

    sniffed = _sniff_text(head)
    if sniffed is not None:
        return Detection(sniffed, _TEXT_MIME[sniffed])
    if ext in CODE_EXTENSIONS:
        return Detection(FileType.code, "text/plain")
    ft = _EXT.get(ext, FileType.text)
    return Detection(ft, _TEXT_MIME.get(ft, "text/plain"))


_TEXT_MIME = {
    FileType.html: "text/html", FileType.markdown: "text/markdown", FileType.csv: "text/csv",
    FileType.json: "application/json", FileType.email: "message/rfc822", FileType.text: "text/plain",
}


def _read_head(path: Path, size: int = 8192) -> str | None:
    """First bytes as text, or None if the file looks binary."""
    with open(path, "rb") as f:
        raw = f.read(size)
    if b"\x00" in raw:
        return None
    # The cut at `size` can split a multi-byte character, so ignore decode errors.
    return raw.decode("utf-8", errors="ignore")


def _sniff_text(head: str) -> FileType | None:
    s = head.lstrip("﻿ \t\r\n").lower()
    if s.startswith("<!doctype html") or s.startswith("<html"):
        return FileType.html
    lines = head.splitlines()[:30]
    header_names = {ln.split(":", 1)[0].strip().lower() for ln in lines if ":" in ln}
    if {"from", "subject"} <= header_names and ({"to", "date", "mime-version"} & header_names):
        return FileType.email
    if s[:1] in "{[":
        try:
            json.loads(head)
            return FileType.json
        except json.JSONDecodeError:
            pass  # truncated head of a big JSON file, or not JSON: fall back to extension
    return None


def _inspect_pdf(path: Path, mime: str) -> Detection:
    from pypdf import PdfReader

    reader = PdfReader(str(path))
    if reader.is_encrypted:
        reader.decrypt("")   # many "encrypted" PDFs only restrict editing
    scanned = []
    for i, page in enumerate(reader.pages, start=1):
        try:
            text = page.extract_text() or ""
        except Exception:
            text = ""
        if len(text.strip()) < MIN_TEXT_CHARS_PER_PAGE:
            scanned.append(i)
    return Detection(FileType.pdf, mime, page_count=len(reader.pages), scanned_pages=scanned)
