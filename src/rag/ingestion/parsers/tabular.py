"""CSV and Excel: groups of rows, each rendered with the header repeated.

A row alone ("argon2, auth, ...") means nothing without its column names, so
every chunk is a small Markdown table with the header, plus the sheet name and
any title text found above the header row.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
from langchain_core.documents import Document

from rag.ingestion.detect import Detection
from rag.ingestion.schema import FileType, SourceInfo, make_doc

TARGET_CHARS = 1600          # ~400 tokens per chunk
HEADER_SEARCH_ROWS = 15


def parse_tabular(path: Path, det: Detection, src: SourceInfo, ctx) -> list[Document]:
    if det.file_type is FileType.csv:
        sep = "\t" if path.suffix.lower() == ".tsv" else ","
        sheets = {"": pd.read_csv(path, sep=sep, header=None, dtype=str, keep_default_na=False)}
    else:
        sheets = pd.read_excel(path, sheet_name=None, header=None, dtype=str)

    docs: list[Document] = []
    for sheet_name, raw in sheets.items():
        docs.extend(_sheet_to_docs(raw, sheet_name, src))
    return docs


def _sheet_to_docs(raw: pd.DataFrame, sheet: str, src: SourceInfo) -> list[Document]:
    df = raw.replace("", pd.NA).dropna(how="all").dropna(axis=1, how="all")
    if df.empty:
        return []

    # Header = first row with the most filled cells among the top rows. Rows
    # above it (report titles) and single-cell rows below it (notes) are context.
    top = df.head(HEADER_SEARCH_ROWS)
    header_pos = int(top.notna().sum(axis=1).argmax())
    title_lines = [_row_text(r) for _, r in df.iloc[:header_pos].iterrows()]
    header = [str(h) if pd.notna(h) else f"col{i + 1}" for i, h in enumerate(df.iloc[header_pos])]
    body = df.iloc[header_pos + 1:]

    ncols = len(header)
    rows, notes = [], []
    for _, r in body.iterrows():
        if r.notna().sum() == 1 and ncols > 2:
            notes.append(_row_text(r))
        else:
            rows.append(["" if pd.isna(v) else str(v).replace("|", "/") for v in r])

    context = "\n".join(filter(None, [
        f"File: {src.filename}" + (f" | Sheet: {sheet}" if sheet else ""),
        *title_lines,
    ]))
    head_md = "| " + " | ".join(header) + " |\n|" + "---|" * ncols

    docs, batch, size = [], [], 0
    for row in rows:
        line = "| " + " | ".join(row) + " |"
        if batch and size + len(line) > TARGET_CHARS:
            docs.append(_make(context, head_md, batch, notes, sheet, src))
            batch, size = [], 0
        batch.append(line)
        size += len(line)
    if batch or notes:
        docs.append(_make(context, head_md, batch, notes, sheet, src))
    return docs


def _make(context, head_md, lines, notes, sheet, src) -> Document:
    text = f"{context}\n\n{head_md}\n" + "\n".join(lines)
    if notes:
        text += "\n\n" + "\n".join(notes)
    return make_doc(text.strip(), src, parser="tabular", section_path=sheet, prechunked=True,
                    sheet=sheet or None)


def _row_text(r: pd.Series) -> str:
    return " ".join(str(v) for v in r if pd.notna(v))
