"""Source code (split on functions/classes) and JSON (split on structure)."""
from __future__ import annotations

import json
from pathlib import Path

from langchain_core.documents import Document
from langchain_text_splitters import Language, RecursiveCharacterTextSplitter, RecursiveJsonSplitter

from rag.ingestion.detect import Detection
from rag.ingestion.parsers.text import read_text
from rag.ingestion.schema import FileType, SourceInfo, make_doc

CODE_CHUNK_CHARS = 1500      # ~400 tokens of code
JSON_CHUNK_CHARS = 1500

_LANGUAGE = {
    ".js": Language.JS, ".mjs": Language.JS, ".cjs": Language.JS, ".jsx": Language.JS,
    ".ts": Language.TS, ".tsx": Language.TS, ".py": Language.PYTHON, ".java": Language.JAVA,
    ".go": Language.GO, ".rb": Language.RUBY, ".rs": Language.RUST, ".c": Language.C,
    ".h": Language.C, ".cpp": Language.CPP, ".hpp": Language.CPP, ".cs": Language.CSHARP,
    ".php": Language.PHP, ".kt": Language.KOTLIN, ".swift": Language.SWIFT,
    ".scala": Language.SCALA, ".sol": Language.SOL, ".lua": Language.LUA,
}


def parse_code_or_json(path: Path, det: Detection, src: SourceInfo, ctx) -> list[Document]:
    if det.file_type is FileType.json:
        return _parse_json(path, src)
    return _parse_code(path, src)


def _parse_code(path: Path, src: SourceInfo) -> list[Document]:
    language = _LANGUAGE.get(path.suffix.lower())
    if language:
        splitter = RecursiveCharacterTextSplitter.from_language(
            language, chunk_size=CODE_CHUNK_CHARS, chunk_overlap=150)
    else:
        splitter = RecursiveCharacterTextSplitter(chunk_size=CODE_CHUNK_CHARS, chunk_overlap=150)
    lang_name = language.value if language else path.suffix.lstrip(".")
    return [
        # The filename header gives each chunk context the code alone lacks.
        make_doc(f"File: {src.filename}\n```{lang_name}\n{part}\n```", src,
                 parser="code_json", prechunked=True, language=lang_name)
        for part in splitter.split_text(read_text(path))
    ]


def _parse_json(path: Path, src: SourceInfo) -> list[Document]:
    raw = read_text(path)
    if path.suffix.lower() == ".jsonl":
        records = [json.loads(line) for line in raw.splitlines() if line.strip()]
        data: dict | list = {"records": records}
    else:
        data = json.loads(raw)
    if isinstance(data, list):
        data = {"items": data}

    splitter = RecursiveJsonSplitter(max_chunk_size=JSON_CHUNK_CHARS)
    parts = splitter.split_json(json_data=data, convert_lists=True)
    return [
        make_doc(f"File: {src.filename}\n{json.dumps(part, ensure_ascii=False, indent=1)}", src,
                 parser="code_json", prechunked=True)
        for part in parts
    ]
