"""LangGraph ingestion graph for one document (diagram 2.2 in the plan).

    detect -> dedup_check -> route -> <parser> -> chunk -> index -> finish
                   |
                   +--- duplicate / unsupported ------------------> finish

Each node records its wall-clock time in `stage_ms`; those timings are saved
on the job row and feed the capacity model with real per-page costs.
"""
from __future__ import annotations

import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Annotated, Any, TypedDict

from langchain_core.documents import Document
from langgraph.graph import END, START, StateGraph
from qdrant_client import models
from sqlalchemy import select, update
from sqlalchemy.exc import IntegrityError

from rag.ingestion.chunking import chunk_documents
from rag.ingestion.detect import Detection, detect
from rag.ingestion.indexer import index_chunks
from rag.ingestion.parsers import ParseContext, get_parser, route
from rag.ingestion.schema import SourceInfo
from rag.storage.postgres import Document as DocRow
from rag.storage.postgres import Job, JobStatus

PARSER_NODES = ["pdf_text", "ocr_vision", "office", "tabular", "email", "web_markup",
                "code_json", "fallback"]


def _merge(a: dict | None, b: dict | None) -> dict:
    return {**(a or {}), **(b or {})}


class IngestState(TypedDict, total=False):
    # inputs
    path: str
    filename: str
    tenant: str
    source_uri: str
    sha256: str
    blob_key: str
    size_bytes: int
    force: bool
    job_id: str          # set when the API pre-registered the job
    # produced by nodes
    detection: Detection
    src: SourceInfo
    document_id: str
    job_id: str
    skip_reason: str | None
    note: str | None
    route: str
    docs: list[Document]
    chunks: list[Document]
    ocr_pages: int
    safety: dict[str, Any]
    result: dict[str, Any]
    stage_ms: Annotated[dict[str, int], _merge]


def build_ingest_graph(deps):
    """`deps` provides settings, sessions (sessionmaker), qdrant client and vector store."""

    def timed(name, fn):
        def run(state: IngestState) -> dict:
            t0 = time.perf_counter()
            out = fn(state) or {}
            out["stage_ms"] = {name: round((time.perf_counter() - t0) * 1000)}
            return out
        return run

    # --- nodes -------------------------------------------------------------
    def detect_type(state: IngestState) -> dict:
        det = detect(Path(state["path"]))
        src = SourceInfo(tenant=state["tenant"], source_uri=state["source_uri"],
                         filename=state["filename"], sha256=state["sha256"],
                         mime=det.mime, file_type=det.file_type)
        return {"detection": det, "src": src, "route": route(det)}

    def dedup_check(state: IngestState) -> dict:
        src, det = state["src"], state["detection"]
        with deps.sessions() as db:
            row = get_or_create_doc(db, deps.settings.collection, src, det,
                                    size_bytes=state["size_bytes"], blob_key=state["blob_key"])

            skip, note = None, None
            if row.indexed_at is not None:
                if state.get("force"):
                    note = "re-indexed (--force)"
                elif not _has_chunks(deps, row.tenant, row.source_uri):
                    # Postgres says indexed, but the chunks are gone (e.g. the Qdrant
                    # collection was deleted). Re-index; the embedding cache makes it ~free.
                    note = "was indexed but its chunks are missing in Qdrant: re-indexed"
                else:
                    skip = f"duplicate of already indexed '{row.filename}'"
            if skip:
                pass
            elif state["route"] == "unsupported":
                skip = f"unsupported file type ({det.mime or 'unknown'})"
            elif state["route"] == "ocr_vision" and not deps.settings.ocr_enabled:
                skip = "needs OCR, which is turned off for this run"

            status = JobStatus.skipped if skip else JobStatus.processing
            now = datetime.now(timezone.utc)
            if state.get("job_id"):
                # Registered by the API and picked up by a worker: one row per job,
                # updated on every (re)try so attempts and errors stay together.
                job = db.get(Job, uuid.UUID(state["job_id"]))
                job.document_id, job.status, job.started_at = row.id, status, now
                job.attempts = (job.attempts or 0) + 1
                job.error = None
            else:
                job = Job(document_id=row.id, queue=queue_for(state["detection"]), attempts=1,
                          status=status, started_at=now)
                db.add(job)
            db.commit()
            return {"document_id": str(row.id), "job_id": str(job.id), "skip_reason": skip,
                    "note": note}

    def make_parser_node(name):
        def parse(state: IngestState) -> dict:
            ctx = ParseContext(settings=deps.settings)
            docs = get_parser(name)(Path(state["path"]), state["detection"], state["src"], ctx)
            return {"docs": docs, "ocr_pages": ctx.stats["ocr_pages"]}
        return parse

    def chunk(state: IngestState) -> dict:
        return {"chunks": chunk_documents(state["docs"])}

    def safety(state: IngestState) -> dict:
        from rag.security.ingest_safety import apply_ingest_safety
        chunks, report = apply_ingest_safety(state["chunks"], deps.settings)
        return {"chunks": chunks, "safety": report.as_dict()}

    def index(state: IngestState) -> dict:
        r = index_chunks(state["chunks"], state["src"], client=deps.qdrant,
                         store=deps.store, settings=deps.settings)
        return {"result": {"chunks": r.total, "new": r.new, "unchanged": r.unchanged,
                           "deleted": r.deleted, **state.get("safety", {})}}

    def finish(state: IngestState) -> dict:
        now = datetime.now(timezone.utc)
        with deps.sessions() as db:
            job = db.get(Job, uuid.UUID(state["job_id"]))
            job.finished_at = now
            job.stage_ms = state.get("stage_ms")
            if not state.get("skip_reason"):
                job.status = JobStatus.indexed
                doc = db.get(DocRow, uuid.UUID(state["document_id"]))
                doc.indexed_at = now
                doc.chunk_count = state["result"]["chunks"]
                doc.ocr_pages = state.get("ocr_pages", 0)
                # Older versions of this source are no longer what Qdrant holds, so
                # they must not count as "already indexed" if that content comes back
                # (e.g. an edit is undone). Their chunks come back from the cache.
                db.execute(
                    update(DocRow)
                    .where(DocRow.collection == doc.collection, DocRow.tenant == doc.tenant,
                           DocRow.source_uri == doc.source_uri, DocRow.id != doc.id,
                           DocRow.indexed_at.is_not(None))
                    .values(indexed_at=None))
            db.commit()
        if not state.get("skip_reason"):
            # Cached answers for this tenant may now be out of date.
            from rag.retrieval.answer_cache import bump_index_version
            bump_index_version(deps.settings, state["tenant"])
        return {}

    # --- wiring ------------------------------------------------------------
    g = StateGraph(IngestState)
    g.add_node("detect_type", timed("detect_type", detect_type))
    g.add_node("dedup_check", timed("dedup_check", dedup_check))
    for name in PARSER_NODES:
        g.add_node(name, timed(name, make_parser_node(name)))
    g.add_node("chunk", timed("chunk", chunk))
    g.add_node("safety", timed("safety", safety))
    g.add_node("index", timed("index", index))
    g.add_node("finish", finish)

    g.add_edge(START, "detect_type")
    g.add_edge("detect_type", "dedup_check")
    g.add_conditional_edges(
        "dedup_check",
        lambda s: "finish" if s.get("skip_reason") else s["route"],
        {**{n: n for n in PARSER_NODES}, "finish": "finish"},
    )
    for name in PARSER_NODES:
        g.add_edge(name, "chunk")
    g.add_edge("chunk", "safety")
    g.add_edge("safety", "index")
    g.add_edge("index", "finish")
    g.add_edge("finish", END)
    return g.compile()


def _has_chunks(deps, tenant: str, source_uri: str) -> bool:
    f = models.Filter(must=[
        models.FieldCondition(key="metadata.tenant", match=models.MatchValue(value=tenant)),
        models.FieldCondition(key="metadata.source_uri", match=models.MatchValue(value=source_uri)),
    ])
    # Uses the payload indexes; one cheap count, not a scan.
    return deps.qdrant.count(deps.settings.collection, count_filter=f, exact=False).count > 0


def get_or_create_doc(db, collection: str, src: SourceInfo, det: Detection, *, size_bytes: int,
                      blob_key: str) -> DocRow:
    """The document row for this content in this collection; safe under concurrency."""
    def find():
        return db.scalar(select(DocRow).where(DocRow.collection == collection,
                                              DocRow.sha256 == src.sha256))
    row = find()
    if row is None:
        row = DocRow(sha256=src.sha256, collection=collection, doc_id=uuid.UUID(src.doc_id),
                     tenant=src.tenant, source_uri=src.source_uri, filename=src.filename,
                     mime=det.mime, file_type=det.file_type.value, size_bytes=size_bytes,
                     blob_key=blob_key, page_count=det.page_count)
        db.add(row)
        try:
            db.flush()
        except IntegrityError:          # another worker registered it first
            db.rollback()
            row = find()
    return row


def already_indexed(deps, row: DocRow) -> bool:
    """Indexed per Postgres AND the chunks really are in Qdrant."""
    return row.indexed_at is not None and _has_chunks(deps, row.tenant, row.source_uri)


def queue_for(det: Detection) -> str:
    """Celery queue for a document: anything needing the vision model goes to the
    OCR workers (I/O-bound, rate-limited); everything else to the CPU parsers."""
    return "ocr_vision" if route(det) == "ocr_vision" or det.scanned_pages else "parse_cpu"
