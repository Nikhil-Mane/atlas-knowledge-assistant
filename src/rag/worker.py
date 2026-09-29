"""Celery workers: the parallel ingestion that makes 10K docs/day possible.

Two queues, scaled independently:
  parse_cpu   Docling parsing + chunking + embedding. CPU-bound: one process per core.
  ocr_vision  scanned pages and images. Waits on the vision API, so a few
              processes each running `OCR_CONCURRENCY` calls in parallel.

Run (Linux/Docker):
  celery -A rag.worker worker -Q parse_cpu  --concurrency 8
  celery -A rag.worker worker -Q ocr_vision --concurrency 2
On Windows use `--pool solo` and start several worker processes instead.

Reliability: tasks are acknowledged only after they finish (a crashed worker's
task is redelivered), transient errors are retried with exponential backoff,
and the last attempt marks the job `dead` for a human to look at.
"""
from __future__ import annotations

import logging
import tempfile
from pathlib import Path

from celery import Celery
from celery.signals import worker_process_init

from rag.config import get_settings

log = logging.getLogger(__name__)
_s = get_settings()

app = Celery("rag", broker=_s.redis_url)
app.conf.update(
    task_acks_late=True,                 # ack after success: crash => redelivery
    task_reject_on_worker_lost=True,
    worker_prefetch_multiplier=1,        # long tasks: don't hoard queued work
    task_ignore_result=True,             # status lives in Postgres, not a result backend
    task_default_queue="parse_cpu",
    worker_max_tasks_per_child=200,      # recycle processes: bounds Docling/torch memory growth
    broker_connection_retry_on_startup=True,
    # Must exceed the longest task (a big scanned PDF), or Redis redelivers it mid-run.
    broker_transport_options={"visibility_timeout": 4 * 3600},
)

# Errors worth retrying: the next attempt may well succeed.
TRANSIENT = {"RateLimitError", "APIConnectionError", "APITimeoutError", "InternalServerError",
             "TimeoutError", "ConnectionError", "OperationalError", "ServiceUnavailableError",
             "ResponseHandlingException", "UnexpectedResponse", "_InactiveRpcError"}
_ingestor = None


def worker_settings():
    # Workers share the day's budget through Redis; a per-process "run" budget
    # would reset every time a process is recycled.
    return get_settings().model_copy(update={"budget_scope": "daily"})


def get_ingestor():
    global _ingestor
    if _ingestor is None:
        from rag.ingestion.pipeline import Ingestor, build_deps
        _ingestor = Ingestor(build_deps(settings=worker_settings()))
    return _ingestor


@worker_process_init.connect
def _reset_after_fork(**_):
    # Connections (gRPC, Postgres pools) must not be shared across forked processes.
    global _ingestor
    _ingestor = None


def backoff_seconds(retries: int, error_type: str) -> int:
    if error_type == "BudgetExceeded":
        return 3600                      # daily cap: try again in an hour
    return min(600, 30 * 2 ** retries)   # 30s, 60s, 120s ... capped at 10 min


@app.task(bind=True, name="rag.ingest", max_retries=_s.ingest_max_retries)
def ingest_task(self, *, job_id: str, blob_key: str, filename: str, tenant: str,
                source_uri: str, force: bool = False) -> dict:
    ingestor = get_ingestor()
    with tempfile.TemporaryDirectory() as tmp:
        # Keep the original name: parsers use the extension as a hint.
        local = ingestor.deps.blob.fetch(blob_key, Path(tmp) / Path(filename).name)
        result = ingestor.ingest(local, tenant=tenant, source_uri=source_uri, force=force,
                                 job_id=job_id, filename=filename, final_attempt=False)

    err = result.get("error_type")
    if err:
        retryable = err in TRANSIENT or err == "BudgetExceeded"
        if retryable and self.request.retries < self.max_retries:
            delay = backoff_seconds(self.request.retries, err)
            log.warning("job %s %s: retry %d in %ss", job_id, err, self.request.retries + 1, delay)
            raise self.retry(countdown=delay)
        # Not retryable (bad file, bad config) or out of retries: needs a human.
        mark_dead(ingestor, job_id)
        result["status"] = "dead"
    return {k: v for k, v in result.items() if k != "stage_ms"}


def mark_dead(ingestor, job_id: str) -> None:
    import uuid

    from rag.storage.postgres import Job, JobStatus
    with ingestor.deps.sessions() as db:
        job = db.get(Job, uuid.UUID(job_id))
        if job is not None:
            job.status = JobStatus.dead
            db.commit()


def enqueue(registration: dict, *, tenant: str, source_uri: str, force: bool = False) -> None:
    """Send a registered upload to the queue that matches its work."""
    ingest_task.apply_async(
        kwargs={"job_id": registration["job_id"], "blob_key": registration["blob_key"],
                "filename": registration["filename"], "tenant": tenant,
                "source_uri": source_uri, "force": force},
        queue=registration["queue"],
    )
