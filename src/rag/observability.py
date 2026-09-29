"""Metrics (Prometheus), structured logs and request IDs.

GET /metrics exposes:
  rag_http_requests_total{method,route,status}   rag_http_request_seconds{route}
  rag_chat_total{status,cached}                  rag_chat_seconds
  rag_llm_calls_total                            rag_ingest_uploads_total{status}
  rag_queue_depth{queue}   rag_jobs{status}   rag_budget_used{meter}   (read at scrape time)

Run one API process per container and scale with replicas, so each scrape
sees one process's counters (no multiprocess registry needed).
"""
from __future__ import annotations

import json
import logging
import sys
import time
import uuid
from contextvars import ContextVar

from prometheus_client import CollectorRegistry, Counter, Histogram, generate_latest
from prometheus_client.core import GaugeMetricFamily

REGISTRY = CollectorRegistry()
HTTP_REQUESTS = Counter("rag_http_requests_total", "HTTP requests", ["method", "route", "status"],
                        registry=REGISTRY)
HTTP_SECONDS = Histogram("rag_http_request_seconds", "HTTP request duration", ["route"],
                         buckets=(0.05, 0.1, 0.25, 0.5, 1, 2, 5, 10, 20, 40), registry=REGISTRY)
CHAT_TOTAL = Counter("rag_chat_total", "Questions answered", ["status", "cached"], registry=REGISTRY)
CHAT_SECONDS = Histogram("rag_chat_seconds", "Question latency",
                         buckets=(0.1, 0.5, 1, 2, 4, 8, 15, 30, 60), registry=REGISTRY)
LLM_CALLS = Counter("rag_llm_calls_total", "Chat-model calls", registry=REGISTRY)
UPLOADS = Counter("rag_ingest_uploads_total", "Uploaded files", ["status"], registry=REGISTRY)

request_id: ContextVar[str] = ContextVar("request_id", default="-")


class StateCollector:
    """Gauges read from Redis/Postgres when Prometheus scrapes (every ~15 s)."""

    def __init__(self, settings, sessions):
        self.s, self.sessions = settings, sessions

    def collect(self):
        from datetime import datetime, timedelta, timezone

        import redis
        from sqlalchemy import func, select

        from rag.cost import get_budget
        from rag.storage.postgres import Job

        depth = GaugeMetricFamily("rag_queue_depth", "Jobs waiting in the queue", labels=["queue"])
        budget = GaugeMetricFamily("rag_budget_used", "Usage today against the daily cap", labels=["meter"])
        jobs = GaugeMetricFamily("rag_jobs", "Jobs in the last 24 h by status", labels=["status"])
        try:
            r = redis.Redis.from_url(self.s.redis_url, socket_timeout=2)
            for q in ("parse_cpu", "ocr_vision"):
                depth.add_metric([q], r.llen(q))
            b = get_budget(self.s)
            for m in ("embed_tokens", "ocr_pages", "chat_calls"):
                budget.add_metric([m], b.used(m))
        except Exception:
            pass
        try:
            since = datetime.now(timezone.utc) - timedelta(hours=24)
            with self.sessions() as db:
                for status, n in db.execute(select(Job.status, func.count()).where(
                        Job.queued_at >= since).group_by(Job.status)):
                    jobs.add_metric([status.value], n)
        except Exception:
            pass
        yield from (depth, budget, jobs)


def metrics_payload() -> bytes:
    return generate_latest(REGISTRY)


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        entry = {"ts": time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime(record.created)),
                 "level": record.levelname, "logger": record.name, "msg": record.getMessage(),
                 "request_id": request_id.get()}
        entry.update(getattr(record, "extra_fields", {}))
        if record.exc_info:
            entry["exc"] = self.formatException(record.exc_info)
        return json.dumps(entry, default=str)


def configure_logging(json_logs: bool) -> None:
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(JsonFormatter() if json_logs else
                         logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
    root = logging.getLogger()
    root.handlers = [handler]
    root.setLevel(logging.INFO)


def install_http_middleware(app) -> None:
    access = logging.getLogger("rag.access")

    @app.middleware("http")
    async def observe(request, call_next):
        rid = request.headers.get("x-request-id") or uuid.uuid4().hex[:16]
        token = request_id.set(rid)
        start = time.perf_counter()
        status = 500
        try:
            response = await call_next(request)
            status = response.status_code
            response.headers["X-Request-ID"] = rid
            return response
        finally:
            route = request.scope.get("route")
            path = getattr(route, "path", "unmatched")
            elapsed = time.perf_counter() - start
            if path != "/metrics":
                HTTP_REQUESTS.labels(request.method, path, str(status)).inc()
                HTTP_SECONDS.labels(path).observe(elapsed)
                access.info("request", extra={"extra_fields": {
                    "method": request.method, "path": path, "status": status,
                    "ms": round(elapsed * 1000), "tenant": getattr(request.state, "tenant", None)}})
            request_id.reset(token)
