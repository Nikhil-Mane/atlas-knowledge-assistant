"""Entry point for ingesting one file: store the blob, run the graph, record failures.

The same function runs inline (CLI) now and inside Celery workers in Phase 5.
"""
from __future__ import annotations

import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from sqlalchemy import select
from sqlalchemy.orm import sessionmaker

from rag.config import Settings, get_settings
from rag.ingestion.graph import build_ingest_graph
from rag.ingestion.indexer import get_vector_store
from rag.models import dense_embeddings
from rag.storage.blob import BlobStore, get_blob_store
from rag.storage.postgres import Document as DocRow
from rag.storage.postgres import Job, JobStatus, session_factory
from rag.storage.qdrant import ensure_collection, get_client

MAX_ATTEMPTS = 3


@dataclass
class Deps:
    settings: Settings
    blob: BlobStore
    sessions: sessionmaker
    qdrant: QdrantClient
    store: QdrantVectorStore


def build_deps(*, fake_embeddings: bool = False, settings: Settings | None = None) -> Deps:
    s = settings or get_settings()
    if fake_embeddings:
        # Fake mode for every model; `Settings.collection` then adds the `_fake`
        # suffix, so fake vectors never mix with real ones.
        s = s.model_copy(update={"llm_provider": "fake"})
    dense = dense_embeddings(s)
    client = get_client(s)
    ensure_collection(client, s)
    return Deps(settings=s, blob=get_blob_store(s), sessions=session_factory(s),
                qdrant=client, store=get_vector_store(client, dense, s))


def estimate_file(path: Path, settings: Settings | None = None) -> dict:
    """Dry run: parse and chunk locally, count what a real run would pay for.

    Makes no API calls and writes nothing to Postgres, Qdrant or the blob store.
    Files already indexed (same SHA-256) and texts already in the embedding
    cache count as free, exactly as they would in a real run.
    """
    from rag.cost import GuardedEmbeddings, count_tokens
    from rag.ingestion.chunking import chunk_documents
    from rag.ingestion.detect import detect
    from rag.ingestion.parsers import ParseContext, get_parser, route
    from rag.ingestion.schema import SourceInfo
    from rag.models import model_id
    from rag.storage.blob import sha256_file

    s = settings or get_settings()
    path = path.resolve()
    sha = sha256_file(path)
    det = detect(path)
    r = route(det)
    base = {"file": path.name, "route": r, "pages": det.page_count}

    with session_factory(s)() as db:
        row = db.scalar(select(DocRow).where(DocRow.collection == s.collection,
                                             DocRow.sha256 == sha))
    if row is not None and row.indexed_at is not None:
        from rag.ingestion.graph import _has_chunks

        probe = Deps(settings=s, blob=None, sessions=None, qdrant=get_client(s), store=None)
        if _has_chunks(probe, row.tenant, row.source_uri):
            return {**base, "status": "skip", "reason": "already indexed", "tokens": 0,
                    "ocr_pages": 0}
    if r == "unsupported":
        return {**base, "status": "skip", "reason": "unsupported", "tokens": 0, "ocr_pages": 0}

    src = SourceInfo(tenant="default", source_uri=path.as_uri(), filename=path.name, sha256=sha,
                     mime=det.mime, file_type=det.file_type)
    ctx = ParseContext(settings=s, dry_run=True)
    chunks = chunk_documents(get_parser(r)(path, det, src, ctx))
    texts = [c.page_content for c in chunks]

    guard = GuardedEmbeddings(inner=None, model_id=model_id(s, "embedding"),
                              dim=s.embedding_dim, settings=s)
    ocr_pages = ctx.stats["ocr_pages"]
    return {**base, "status": "estimate", "chunks": len(chunks),
            "tokens": guard.uncached_tokens(texts), "tokens_total": count_tokens(texts),
            "ocr_pages": ocr_pages,
            # OCR text doesn't exist yet in a dry run: assume ~600 tokens per page.
            "ocr_tokens_est": ocr_pages * 600}


def register_upload(deps: Deps, path: Path, *, filename: str, tenant: str, source_uri: str,
                    force: bool = False) -> dict:
    """Store an uploaded file and create its `queued` job (or a `skipped` one).

    Cheap (hash + type sniffing), so the API can call it inline and return a
    job id immediately; parsing and embedding happen later in a worker.
    """
    from rag.ingestion.detect import detect
    from rag.ingestion.graph import already_indexed, get_or_create_doc, queue_for
    from rag.ingestion.parsers import route
    from rag.ingestion.schema import SourceInfo

    sha, key = deps.blob.put(path)
    det = detect(path)
    src = SourceInfo(tenant=tenant, source_uri=source_uri, filename=filename, sha256=sha,
                     mime=det.mime, file_type=det.file_type)
    with deps.sessions() as db:
        row = get_or_create_doc(db, deps.settings.collection, src, det,
                                size_bytes=path.stat().st_size, blob_key=key)
        skip = None
        if not force and already_indexed(deps, row):
            skip = f"duplicate of already indexed '{row.filename}'"
        elif route(det) == "unsupported":
            skip = f"unsupported file type ({det.mime or 'unknown'})"
        elif det.page_count and det.page_count > deps.settings.max_pdf_pages:
            skip = f"too many pages ({det.page_count} > MAX_PDF_PAGES={deps.settings.max_pdf_pages})"
        now = datetime.now(timezone.utc)
        job = Job(document_id=row.id, queue=queue_for(det), attempts=0,
                  status=JobStatus.skipped if skip else JobStatus.queued,
                  error=skip, finished_at=now if skip else None)
        db.add(job)
        db.commit()
        return {"job_id": str(job.id), "document_id": str(row.id), "doc_id": str(row.doc_id),
                "filename": filename, "status": job.status.value, "reason": skip,
                "queue": job.queue, "blob_key": key, "file_type": det.file_type.value}


class Ingestor:
    def __init__(self, deps: Deps):
        self.deps = deps
        self.graph = build_ingest_graph(deps)

    def ingest(self, path: Path, *, tenant: str = "default", source_uri: str | None = None,
               force: bool = False, job_id: str | None = None, filename: str | None = None,
               final_attempt: bool = True) -> dict:
        """Ingest one file. `job_id`: a job pre-registered by `register_upload`.
        `final_attempt=False` (a worker that will retry) records failures as
        `failed`; the last attempt records `dead`."""
        path = path.resolve()
        name = filename or path.name
        sha, key = self.deps.blob.put(path)
        state = {
            "path": str(path), "filename": name, "tenant": tenant,
            "source_uri": source_uri or path.as_uri(), "sha256": sha, "blob_key": key,
            "size_bytes": path.stat().st_size, "force": force,
        }
        if job_id:
            state["job_id"] = job_id
        try:
            out = self.graph.invoke(state)
        except Exception as e:
            status = self._record_failure(sha, e, job_id=job_id, final_attempt=final_attempt)
            return {"file": name, "status": status.value, "error": f"{type(e).__name__}: {e}",
                    "error_type": type(e).__name__, "job_id": job_id}

        det = out["detection"]
        return {
            "file": name,
            "job_id": out.get("job_id"),
            "document_id": out.get("document_id"),
            "status": "skipped" if out.get("skip_reason") else "indexed",
            "reason": out.get("skip_reason") or out.get("note"),
            "file_type": det.file_type.value,
            "route": out["route"],
            "pages": det.page_count,
            "ocr_pages": out.get("ocr_pages", 0),
            **out.get("result", {}),
            "stage_ms": out.get("stage_ms", {}),
        }

    def _record_failure(self, sha: str, err: Exception, *, job_id: str | None = None,
                        final_attempt: bool = True) -> JobStatus:
        """Mark the job failed, or dead once retries are exhausted."""
        error = f"{type(err).__name__}: {err}"[:4000]
        if job_id:
            with self.deps.sessions() as db:
                job = db.get(Job, uuid.UUID(job_id))
                if job is None:
                    return JobStatus.failed
                job.status = JobStatus.failed if not final_attempt else JobStatus.dead
                job.error = error
                job.finished_at = datetime.now(timezone.utc)
                db.commit()
                return job.status
        with self.deps.sessions() as db:
            doc = db.scalar(select(DocRow).where(
                DocRow.collection == self.deps.settings.collection, DocRow.sha256 == sha))
            if doc is None:            # failed before dedup_check registered it
                return JobStatus.failed
            attempts = db.query(Job).filter(Job.document_id == doc.id,
                                            Job.status != JobStatus.skipped).count()
            job = db.scalar(select(Job).where(Job.document_id == doc.id)
                            .order_by(Job.queued_at.desc()).limit(1))
            status = JobStatus.dead if attempts >= MAX_ATTEMPTS else JobStatus.failed
            if job is not None and job.status is JobStatus.processing:
                job.status = status
                job.attempts = attempts
                job.error = f"{type(err).__name__}: {err}"[:4000]
                job.finished_at = datetime.now(timezone.utc)
                db.commit()
            return status
