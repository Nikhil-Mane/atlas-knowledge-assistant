"""Vision-model OCR for scanned pages, handwriting, whiteboards and diagrams.

This is the expensive path, so only pages with no text layer come here.
Pages are sent in small batches with bounded concurrency so a 300-page scan
neither exhausts memory nor bursts past the OpenAI rate limit.
"""
from __future__ import annotations

import base64
import io
from functools import lru_cache
from pathlib import Path

from langchain_core.documents import Document
from langchain_core.messages import HumanMessage
from PIL import Image

from rag.cost import CACHE_DIR, SqliteCache, cache_key, get_budget, get_rate_limiter
from rag.ingestion.detect import Detection
from rag.ingestion.schema import FileType, SourceInfo, make_doc
from rag.models import model_id, vision_llm

PROMPT_VERSION = "v1"   # bump when PROMPT changes so cached transcriptions are redone

MAX_SIDE_PX = 2000     # enough for handwriting; larger only adds cost
RENDER_SCALE = 2.0     # PDF points -> pixels (~144 dpi)
PAGE_BATCH = 8         # pages rendered in memory at once

PROMPT = (
    "Transcribe this page into Markdown.\n"
    "- Copy all text exactly as written, including code, symbols and numbers. Do not fix or summarize.\n"
    "- Keep the structure: headings, bullet lists, tables, code blocks.\n"
    "- For a diagram or drawing, add one line: [Diagram: what it shows and its labels].\n"
    "- Write [illegible] for anything you cannot read. Never guess.\n"
    "Return only the transcription."
)


def parse_scanned(path: Path, det: Detection, src: SourceInfo, ctx) -> list[Document]:
    if det.file_type is FileType.pdf:
        return ocr_pdf_pages(path, det.scanned_pages or list(range(1, (det.page_count or 0) + 1)), src, ctx)
    with Image.open(path) as img:
        return _ocr([(None, img.copy())], src, ctx)


def ocr_pdf_pages(path: Path, pages: list[int], src: SourceInfo, ctx) -> list[Document]:
    import pypdfium2 as pdfium

    docs: list[Document] = []
    pdf = pdfium.PdfDocument(str(path))
    try:
        for start in range(0, len(pages), PAGE_BATCH):
            batch = [(n, pdf[n - 1].render(scale=RENDER_SCALE).to_pil())
                     for n in pages[start:start + PAGE_BATCH]]
            docs += _ocr(batch, src, ctx)
    finally:
        pdf.close()
    return docs


def _ocr(images: list[tuple[int | None, Image.Image]], src: SourceInfo, ctx) -> list[Document]:
    s = ctx.settings
    ctx.stats["ocr_pages"] += len(images)
    if ctx.dry_run or not s.ocr_enabled:
        return []    # counted for the estimate, never sent

    urls = [_data_url(img) for _, img in images]
    if s.llm_provider == "fake":      # free stand-in: no cache, no budget
        replies = vision_llm(s).batch([[HumanMessage(content=[
            {"type": "text", "text": PROMPT}, {"type": "image_url", "image_url": {"url": u}}])]
            for u in urls])
        return [make_doc(_reply_text(r), src, parser="ocr_vision", page=page, ocr=True,
                         section_path=f"Page {page}" if page else "")
                for (page, _), r in zip(images, replies)]

    keys = [cache_key(model_id(s, "vision"), PROMPT_VERSION, u) for u in urls]
    cache = ocr_cache() if s.ocr_cache_enabled else None
    hits = cache.get_many(keys) if cache else {}
    budget = get_budget(s)
    budget.ocr_cache_hits += len(hits)

    missing = [i for i, k in enumerate(keys) if k not in hits]
    if missing:
        budget.charge_ocr(len(missing))       # raises before any call if over the cap
        limiter = get_rate_limiter(s)
        if limiter:
            limiter.acquire("vision_rpm", len(missing), s.vision_rpm_limit)
        messages = [[HumanMessage(content=[
            {"type": "text", "text": PROMPT},
            {"type": "image_url", "image_url": {"url": urls[i], "detail": "high"}},
        ])] for i in missing]
        try:
            replies = vision_llm(s).batch(messages, config={"max_concurrency": s.ocr_concurrency})
        except Exception:
            budget.refund_ocr(len(missing))
            raise
        new = {keys[i]: _reply_text(r).encode("utf-8") for i, r in zip(missing, replies)}
        if cache:
            cache.put_many(new)
        hits.update(new)

    if cache:
        cache.add_owner(src.sha256, keys)     # lets document deletion purge this text
    docs = []
    for (page, _), key in zip(images, keys):
        text = hits[key].decode("utf-8").strip()
        if text:
            docs.append(make_doc(text, src, parser="ocr_vision", page=page, ocr=True,
                                 section_path=f"Page {page}" if page else ""))
    return docs


def _reply_text(reply) -> str:
    if isinstance(reply.content, str):
        return reply.content
    return "".join(part.get("text", "") for part in reply.content if isinstance(part, dict))


@lru_cache
def ocr_cache() -> SqliteCache:
    return SqliteCache(CACHE_DIR / "ocr.sqlite")


def _data_url(img: Image.Image) -> str:
    img = img.convert("RGB")
    img.thumbnail((MAX_SIDE_PX, MAX_SIDE_PX))
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=85)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()
