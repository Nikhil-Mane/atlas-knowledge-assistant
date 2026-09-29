"""FastAPI service: upload documents, track jobs, ask questions (JSON or streaming).

Run:  uvicorn rag.api.app:app --host 0.0.0.0 --port 8000
Docs: http://127.0.0.1:8000/docs      UI: http://127.0.0.1:8000/      Metrics: /metrics

Security: with AUTH_MODE=keys every request needs an X-API-Key. The key
decides the tenant; `user` keys can't see other tenants' documents or
conversations. Per-key request rate and daily question quotas apply.

Concurrency: chat endpoints are async end to end (LLM calls, retrieval,
Postgres memory), so one process serves many questions at once while they
wait on the model. Uploads and admin endpoints run in the threadpool.
"""
from __future__ import annotations

import asyncio
import json
import logging
import re
import tempfile
import threading
import time
import uuid
from contextlib import asynccontextmanager
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from fastapi import BackgroundTasks, Depends, FastAPI, File, Form, Header, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, Response, StreamingResponse
from langchain_core.messages import AIMessage, AIMessageChunk, HumanMessage
from pydantic import BaseModel, Field
from qdrant_client import models
from sqlalchemy import delete, func, select
from sqlalchemy.dialects.postgresql import insert as pg_insert

from rag.auth import AuthError, Authenticator, Principal
from rag.config import Settings, get_settings
from rag.cost import BudgetExceeded, get_budget
from rag.health import run_checks
from rag.ingestion.pipeline import Deps, Ingestor, build_deps, register_upload
from rag.observability import (CHAT_SECONDS, CHAT_TOTAL, REGISTRY, UPLOADS, StateCollector,
                               configure_logging, install_http_middleware, metrics_payload)
from rag.retrieval.answer_cache import AnswerCache, bump_index_version
from rag.retrieval.graph import FILTERED_REPLY, ChatUnavailable, ContentBlocked, turn_input
from rag.retrieval.service import build_service_graph_async, use_selector_loop_on_windows
from rag.storage.postgres import Document as DocRow
from rag.storage.postgres import AuditEvent, Job, JobStatus, Thread
from rag.security.audit import record as audit
from rag.security.guardrails import OutputGuard, StreamRedactor

STATIC = Path(__file__).parent / "static"
_CHUNK = 1024 * 1024
log = logging.getLogger("rag.api")
use_selector_loop_on_windows()


class ChatRequest(BaseModel):
    question: str = Field(min_length=1, max_length=4000)
    thread_id: str | None = Field(default=None, max_length=100)
    tenant: str | None = None          # admin keys only; users are pinned to their key's tenant
    file_types: list[str] | None = None


class ChatResponse(BaseModel):
    thread_id: str
    answer: str
    status: str
    citations: list[dict[str, Any]]
    grounded: bool | None
    unsupported: list[str]
    search_query: str | None
    cached: bool = False
    guard: list[str] = []              # guardrail notes: blocked/flagged input, redactions


def safe_name(name: str | None) -> str:
    base = Path(name or "upload.bin").name
    return re.sub(r"[^\w.\- ]", "_", base)[:200] or "upload.bin"


def sse(event: str, data: dict) -> str:
    return f"event: {event}\ndata: {json.dumps(data, default=str)}\n\n"


def create_app(settings: Settings | None = None, *, memory: str = "postgres") -> FastAPI:
    # Long-running and shared: budgets are per day across all processes.
    s = (settings or get_settings()).model_copy(update={"budget_scope": "daily"})
    configure_logging(s.log_json)
    state: dict[str, Any] = {}
    inline_lock = threading.Lock()      # one inline ingest at a time (Docling is heavy)

    @asynccontextmanager
    async def lifespan(_app: FastAPI):
        # Blocking client calls (Qdrant, Redis) run in the default executor; its
        # default size (~12 threads) caps concurrency far below what async allows.
        from concurrent.futures import ThreadPoolExecutor
        asyncio.get_running_loop().set_default_executor(ThreadPoolExecutor(max_workers=s.io_threads))
        state["deps"] = build_deps(settings=s)
        state["auth"] = Authenticator(s, state["deps"].sessions)
        state["graph"], state["pool"] = await build_service_graph_async(s, memory=memory)
        state["cache"] = AnswerCache(s)
        collector = StateCollector(s, state["deps"].sessions)
        REGISTRY.register(collector)
        log.info("api ready", extra={"extra_fields": {"provider": s.llm_provider,
                                                      "auth_mode": s.auth_mode,
                                                      "collection": s.collection}})
        yield
        REGISTRY.unregister(collector)
        if state.get("pool") is not None:
            await state["pool"].close()

    app = FastAPI(title="Atlas: Enterprise Knowledge Assistant API", version="2.1", lifespan=lifespan)
    install_http_middleware(app)
    origins = [o.strip() for o in s.cors_origins.split(",") if o.strip()]
    if origins:
        app.add_middleware(CORSMiddleware, allow_origins=origins, allow_methods=["GET", "POST", "DELETE"],
                           allow_headers=["X-API-Key", "Content-Type"], allow_credentials=False)

    csp = ("default-src 'self'; script-src 'self' 'unsafe-inline'; style-src 'self' 'unsafe-inline' "
           "https://fonts.googleapis.com; font-src https://fonts.gstatic.com; img-src 'self' data:; "
           "connect-src 'self'; frame-ancestors 'none'; base-uri 'none'; form-action 'self'")

    @app.middleware("http")
    async def security_headers(request: Request, call_next):
        response = await call_next(request)
        h = response.headers
        h.setdefault("X-Content-Type-Options", "nosniff")
        h.setdefault("X-Frame-Options", "DENY")
        h.setdefault("Referrer-Policy", "no-referrer")
        h.setdefault("Permissions-Policy", "camera=(), microphone=(), geolocation=()")
        h.setdefault("Content-Security-Policy", csp)
        if request.url.path not in ("/", "/docs", "/openapi.json"):
            h.setdefault("Cache-Control", "no-store")     # answers and document lists aren't cached
        if s.hsts:
            h.setdefault("Strict-Transport-Security", "max-age=31536000; includeSubDomains")
        return response

    @app.exception_handler(AuthError)
    async def auth_error(_request: Request, exc: AuthError):
        return JSONResponse({"detail": str(exc)}, status_code=exc.status)

    def client_ip(request: Request) -> str:
        return request.client.host if request.client else "unknown"

    def principal(request: Request, x_api_key: str | None = Header(default=None)) -> Principal:
        auth: Authenticator = state["auth"]
        p = auth.authenticate(x_api_key, ip=client_ip(request))
        auth.check_rate(p)
        request.state.tenant = p.tenant
        request.state.principal = p
        return p

    def tenant_for(p: Principal, requested: str | None, request: Request) -> str:
        tenant = p.resolve_tenant(requested)
        if tenant != p.tenant:          # only admins get here: record cross-tenant access
            audit(deps().sessions, "admin_cross_tenant", tenant=tenant, key_id=p.key_id,
                  target=request.url.path, ip=client_ip(request))
        return tenant

    def admin(p: Principal = Depends(principal)) -> Principal:
        if not p.is_admin:
            raise AuthError(403, "Admin key required")
        return p

    def deps() -> Deps:
        return state["deps"]

    # --- UI, health, metrics -----------------------------------------------
    @app.get("/", include_in_schema=False)
    def ui():
        return FileResponse(STATIC / "index.html")

    @app.get("/live", include_in_schema=False)
    def live():
        """Liveness: the process is up. (Readiness uses /health, which checks dependencies.)"""
        return {"ok": True}

    @app.get("/health")
    def health():
        results = run_checks(s)
        ok = all(r["ok"] for r in results.values())
        if not ok:
            raise HTTPException(503, {"ok": False, "services": results})
        return {"ok": True, "provider": s.llm_provider, "auth_mode": s.auth_mode,
                "collection": s.collection, "ingest_mode": s.ingest_mode, "services": results}

    @app.get("/metrics", include_in_schema=False)
    def metrics():
        return Response(metrics_payload(), media_type="text/plain; version=0.0.4")

    @app.get("/me")
    def me(p: Principal = Depends(principal)):
        return {"name": p.name, "tenant": p.tenant, "role": p.role,
                "requests_per_minute": p.requests_per_minute, "questions_per_day": p.questions_per_day}

    # --- ingestion ---------------------------------------------------------
    def run_inline(reg: dict, tenant: str, source_uri: str, force: bool) -> None:
        with inline_lock:
            if "ingestor" not in state:
                state["ingestor"] = Ingestor(deps())
            ing: Ingestor = state["ingestor"]
            with tempfile.TemporaryDirectory() as tmp:
                local = ing.deps.blob.fetch(reg["blob_key"], Path(tmp) / reg["filename"])
                ing.ingest(local, tenant=tenant, source_uri=source_uri, force=force,
                           job_id=reg["job_id"], filename=reg["filename"])

    @app.post("/ingest")
    def ingest(request: Request, background: BackgroundTasks, files: list[UploadFile] = File(...),
               tenant: str | None = Form(None), force: bool = Form(False),
               source_uri: str | None = Form(None), p: Principal = Depends(principal)):
        tenant = tenant_for(p, tenant, request)
        if len(files) > s.max_files_per_upload:
            raise HTTPException(413, f"At most {s.max_files_per_upload} files per request")
        if source_uri and len(files) > 1:
            raise HTTPException(400, "source_uri can only be given with a single file")
        limit = s.max_upload_mb * _CHUNK
        jobs = []
        for upload in files:
            name = safe_name(upload.filename)
            uri = source_uri or f"upload://{tenant}/{name}"
            with tempfile.TemporaryDirectory() as tmp:
                path = Path(tmp) / name
                size = 0
                with open(path, "wb") as out:
                    while block := upload.file.read(_CHUNK):
                        size += len(block)
                        if size > limit:
                            raise HTTPException(413, f"{name} is larger than {s.max_upload_mb} MB")
                        out.write(block)
                reg = register_upload(deps(), path, filename=name, tenant=tenant, source_uri=uri,
                                      force=force)
            UPLOADS.labels(reg["status"]).inc()
            if reg["status"] == JobStatus.queued.value:
                if s.ingest_mode == "queue":
                    from rag.worker import enqueue
                    enqueue(reg, tenant=tenant, source_uri=uri, force=force)
                else:
                    background.add_task(run_inline, reg, tenant, uri, force)
            jobs.append({k: reg[k] for k in ("job_id", "doc_id", "filename", "status", "reason",
                                             "queue", "file_type")})
        audit(deps().sessions, "ingest", tenant=tenant, key_id=p.key_id, ip=client_ip(request),
              detail={"files": [{"name": j["filename"], "status": j["status"]} for j in jobs]})
        return {"tenant": tenant, "jobs": jobs}

    @app.get("/jobs/{job_id}")
    def job_status(job_id: uuid.UUID, p: Principal = Depends(principal)):
        with deps().sessions() as db:
            job = db.get(Job, job_id)
            doc = db.get(DocRow, job.document_id) if job else None
            if job is None or doc is None or (doc.tenant != p.tenant and not p.is_admin):
                raise HTTPException(404, "job not found")      # don't reveal other tenants' jobs
            return {"job_id": str(job.id), "status": job.status.value, "queue": job.queue,
                    "attempts": job.attempts, "error": job.error, "stage_ms": job.stage_ms,
                    "queued_at": job.queued_at, "started_at": job.started_at,
                    "finished_at": job.finished_at, "filename": doc.filename,
                    "doc_id": str(doc.doc_id), "chunk_count": doc.chunk_count}

    @app.get("/documents")
    def list_documents(request: Request, tenant: str | None = None, limit: int = 50, offset: int = 0,
                       p: Principal = Depends(principal)):
        tenant = tenant_for(p, tenant, request)
        with deps().sessions() as db:
            rows = db.scalars(
                select(DocRow).where(DocRow.collection == s.collection, DocRow.tenant == tenant,
                                     DocRow.indexed_at.is_not(None))
                .order_by(DocRow.indexed_at.desc()).limit(min(limit, 500)).offset(offset)).all()
            return {"tenant": tenant, "documents": [
                {"doc_id": str(r.doc_id), "filename": r.filename, "file_type": r.file_type,
                 "source_uri": r.source_uri, "chunks": r.chunk_count, "pages": r.page_count,
                 "ocr_pages": r.ocr_pages, "indexed_at": r.indexed_at} for r in rows]}

    @app.delete("/documents/{doc_id}")
    def delete_document(request: Request, doc_id: uuid.UUID, tenant: str | None = None,
                        p: Principal = Depends(principal)):
        """Removes every copy: chunks, records, the raw file, and cached OCR text."""
        tenant = tenant_for(p, tenant, request)
        d = deps()
        flt = models.Filter(must=[
            models.FieldCondition(key="metadata.doc_id", match=models.MatchValue(value=str(doc_id))),
            models.FieldCondition(key="metadata.tenant", match=models.MatchValue(value=tenant)),
        ])
        points = d.qdrant.count(s.collection, count_filter=flt, exact=True).count
        with d.sessions() as db:
            rows = db.scalars(select(DocRow).where(
                DocRow.collection == s.collection, DocRow.doc_id == doc_id,
                DocRow.tenant == tenant)).all()
            if not points and not rows:
                raise HTTPException(404, "document not found")
            versions = [(r.sha256, r.blob_key) for r in rows]
            d.qdrant.delete(s.collection, points_selector=models.FilterSelector(filter=flt))
            for r in rows:
                db.delete(r)
            db.commit()
            blobs_deleted = 0
            for _sha, key in versions:       # raw files no other document still references
                if not db.scalar(select(func.count()).select_from(DocRow).where(DocRow.blob_key == key)):
                    d.blob.delete(key)
                    blobs_deleted += 1
        ocr_deleted = 0
        from rag.ingestion.parsers.ocr_vision import ocr_cache
        for sha, _key in versions:
            ocr_deleted += ocr_cache().delete_owned(sha)
        bump_index_version(s, tenant)
        result = {"doc_id": str(doc_id), "chunks_deleted": points, "versions_deleted": len(versions),
                  "raw_files_deleted": blobs_deleted, "ocr_pages_purged": ocr_deleted}
        audit(d.sessions, "document_deleted", tenant=tenant, key_id=p.key_id, target=str(doc_id),
              detail=result, ip=client_ip(request))
        return result

    @app.get("/stats")
    def stats(_: Principal = Depends(admin)):
        since = datetime.now(timezone.utc) - timedelta(hours=24)
        with deps().sessions() as db:
            by_status = dict(db.execute(
                select(Job.status, func.count()).where(Job.queued_at >= since)
                .group_by(Job.status)).all())
        out: dict[str, Any] = {"jobs_last_24h": {k.value: v for k, v in by_status.items()}}
        try:
            import redis
            r = redis.Redis.from_url(s.redis_url, socket_timeout=2)
            out["queue_depth"] = {q: r.llen(q) for q in ("parse_cpu", "ocr_vision")}
            b = get_budget(s)
            out["usage_today"] = {m: {"used": b.used(m), "cap": b.caps[m]}
                                  for m in ("embed_tokens", "ocr_pages", "chat_calls")}
        except Exception as e:
            out["redis_error"] = str(e)
        return out

    # --- chat ----------------------------------------------------------------
    def config(p: Principal, thread_id: str) -> dict:
        return {"configurable": {"thread_id": p.thread_key(thread_id)}}

    def touch_thread(p: Principal, tid: str) -> None:
        with deps().sessions() as db:
            db.execute(pg_insert(Thread).values(thread_key=p.thread_key(tid), tenant=p.tenant,
                                                key_id=p.key_id)
                       .on_conflict_do_update(index_elements=[Thread.thread_key],
                                              set_={"last_active_at": func.now()}))
            db.commit()

    def after_turn(p: Principal, tid: str, result: dict, request: Request) -> None:
        touch_thread(p, tid)
        if result.get("status") == "blocked" and result.get("guard"):
            # Reasons only: the question text itself is never written to the audit log.
            audit(deps().sessions, "question_blocked", tenant=p.tenant, key_id=p.key_id,
                  detail={"reasons": result["guard"][:5]}, ip=client_ip(request))

    async def start_turn(req: ChatRequest, p: Principal, request: Request) -> tuple[str, str, dict | None]:
        """Quota, tenant and cache lookup shared by both chat endpoints."""
        tenant = tenant_for(p, req.tenant, request)
        await asyncio.to_thread(state["auth"].charge_question, p)
        tid = req.thread_id or str(uuid.uuid4())
        cached = None
        if req.thread_id is None:           # first turn: no history, so answers are reusable
            cached = await state["cache"].get(tenant, req.question, req.file_types)
            if cached:
                # Record the turn so follow-ups in this new thread have context.
                await state["graph"].aupdate_state(config(p, tid), {
                    **turn_input(req.question, tenant=tenant, file_types=req.file_types),
                    "messages": [HumanMessage(req.question), AIMessage(cached["answer"])],
                    **{k: cached.get(k) for k in ("answer", "citations", "grounded", "status")},
                }, as_node="check_grounded")
        return tenant, tid, cached

    def result_of(values: dict) -> dict:
        return {"answer": values.get("answer", ""), "status": values.get("status", ""),
                "citations": values.get("citations", []), "grounded": values.get("grounded"),
                "unsupported": values.get("unsupported", []),
                "search_query": values.get("search_query"), "guard": values.get("guard", [])}

    @app.post("/chat", response_model=ChatResponse)
    async def chat(req: ChatRequest, request: Request, p: Principal = Depends(principal)):
        t0 = time.perf_counter()
        tenant, tid, cached = await start_turn(req, p, request)
        if cached:
            await asyncio.to_thread(touch_thread, p, tid)
            CHAT_TOTAL.labels(cached["status"], "yes").inc()
            CHAT_SECONDS.observe(time.perf_counter() - t0)
            return ChatResponse(thread_id=tid, cached=True, **{**cached, "guard": cached.get("guard", [])})
        try:
            out = await state["graph"].ainvoke(
                turn_input(req.question, tenant=tenant, file_types=req.file_types), config(p, tid),
                durability=s.checkpoint_durability)
        except BudgetExceeded as e:
            raise HTTPException(429, str(e)) from e
        except ChatUnavailable as e:
            raise HTTPException(503, str(e), headers={"Retry-After": "10"}) from e
        except ContentBlocked:
            audit(deps().sessions, "content_filtered", tenant=p.tenant, key_id=p.key_id,
                  ip=client_ip(request))
            CHAT_TOTAL.labels("blocked", "no").inc()
            return ChatResponse(thread_id=tid, answer=FILTERED_REPLY, status="blocked", citations=[],
                                grounded=None, unsupported=[], search_query=None,
                                guard=["provider content filter"])
        result = result_of(out)
        await asyncio.to_thread(after_turn, p, tid, result, request)
        if req.thread_id is None:
            await state["cache"].put(tenant, req.question, req.file_types, result)
        CHAT_TOTAL.labels(result["status"], "no").inc()
        CHAT_SECONDS.observe(time.perf_counter() - t0)
        return ChatResponse(thread_id=tid, **result)

    @app.post("/chat/stream")
    async def chat_stream(req: ChatRequest, request: Request, p: Principal = Depends(principal)):
        """Server-sent events: `start`, `step` (per graph node), `token`, `final`, `error`."""
        t0 = time.perf_counter()
        tenant, tid, cached = await start_turn(req, p, request)
        graph = state["graph"]
        redactor = StreamRedactor(OutputGuard(s))     # secrets/leaks never reach the client

        async def events():
            yield sse("start", {"thread_id": tid})
            if cached:
                await asyncio.to_thread(touch_thread, p, tid)
                yield sse("step", {"node": "cache"})
                yield sse("final", {"thread_id": tid, "cached": True, **cached})
                CHAT_TOTAL.labels(cached["status"], "yes").inc()
                CHAT_SECONDS.observe(time.perf_counter() - t0)
                return
            try:
                async for mode, chunk in graph.astream(
                        turn_input(req.question, tenant=tenant, file_types=req.file_types),
                        config(p, tid), stream_mode=["updates", "messages"],
                        durability=s.checkpoint_durability):
                    if mode == "messages":
                        msg, meta = chunk
                        # Only streamed pieces: the complete message saved to the
                        # conversation afterwards is emitted too and would repeat the answer.
                        if (meta.get("langgraph_node") == "generate"
                                and isinstance(msg, AIMessageChunk) and msg.content):
                            text = redactor.feed(msg.content)
                            if text:
                                yield sse("token", {"text": text})
                    else:
                        for node, update in chunk.items():
                            yield sse("step", {"node": node, **_step_summary(node, update or {})})
                tail = redactor.flush()
                if tail:
                    yield sse("token", {"text": tail})
                result = result_of((await graph.aget_state(config(p, tid))).values)
                await asyncio.to_thread(after_turn, p, tid, result, request)
                if req.thread_id is None:
                    await state["cache"].put(tenant, req.question, req.file_types, result)
                CHAT_TOTAL.labels(result["status"], "no").inc()
                CHAT_SECONDS.observe(time.perf_counter() - t0)
                yield sse("final", {"thread_id": tid, "cached": False, **result})
            except BudgetExceeded as e:
                yield sse("error", {"message": str(e), "code": 429})
            except ChatUnavailable as e:
                yield sse("error", {"message": str(e), "code": 503})
            except ContentBlocked:
                audit(deps().sessions, "content_filtered", tenant=p.tenant, key_id=p.key_id,
                      ip=client_ip(request))
                yield sse("final", {"thread_id": tid, "cached": False, "answer": FILTERED_REPLY,
                                    "status": "blocked", "citations": [], "grounded": None,
                                    "unsupported": [], "guard": ["provider content filter"]})
            except Exception as e:
                log.exception("chat stream failed")
                yield sse("error", {"message": f"{type(e).__name__}: {e}", "code": 500})

        return StreamingResponse(events(), media_type="text/event-stream",
                                 headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})

    @app.delete("/threads/{thread_id}")
    async def delete_thread(thread_id: str, request: Request, p: Principal = Depends(principal)):
        """Erase a conversation: every saved turn and its tracking row."""
        key = p.thread_key(thread_id)
        await state["graph"].checkpointer.adelete_thread(key)

        def drop_row() -> int:
            with deps().sessions() as db:
                n = db.execute(delete(Thread).where(Thread.thread_key == key)).rowcount
                db.commit()
                return n
        existed = await asyncio.to_thread(drop_row)
        audit(deps().sessions, "thread_deleted", tenant=p.tenant, key_id=p.key_id, target=thread_id,
              ip=client_ip(request))
        return {"thread_id": thread_id, "deleted": True, "tracked": bool(existed)}

    @app.get("/audit")
    def audit_log(tenant: str | None = None, action: str | None = None, limit: int = 100,
                  _: Principal = Depends(admin)):
        with deps().sessions() as db:
            q = select(AuditEvent).order_by(AuditEvent.ts.desc()).limit(min(limit, 1000))
            if tenant:
                q = q.where(AuditEvent.tenant == tenant)
            if action:
                q = q.where(AuditEvent.action == action)
            return {"events": [{"ts": e.ts, "action": e.action, "tenant": e.tenant, "key_id": e.key_id,
                                "target": e.target, "detail": e.detail, "ip": e.ip}
                               for e in db.scalars(q).all()]}

    @app.get("/threads/{thread_id}")
    async def thread(thread_id: str, p: Principal = Depends(principal)):
        values = (await state["graph"].aget_state(config(p, thread_id))).values
        return {"thread_id": thread_id, "messages": [
            {"role": "user" if m.type == "human" else "assistant", "content": m.content}
            for m in values.get("messages", [])]}

    return app


def _step_summary(node: str, update: dict) -> dict:
    if node == "rewrite_query":
        return {"search_query": update.get("search_query")}
    if node == "hybrid_retrieve":
        return {"candidates": len(update.get("candidates", []))}
    if node == "rerank":
        return {"kept": len(update.get("context", []))}
    if node == "grade":
        return {"relevant": len(update.get("context", [])), "attempts": update.get("attempts")}
    if node == "check_grounded":
        return {"grounded": update.get("grounded")}
    return {}


def __getattr__(name: str):
    # `uvicorn rag.api.app:app` builds the app lazily, so importing this module
    # (e.g. in tests) doesn't connect to anything.
    if name == "app":
        return create_app()
    raise AttributeError(name)
