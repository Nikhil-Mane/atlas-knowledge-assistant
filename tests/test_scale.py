"""Production scale pieces: shared daily budget, shared rate limiter, worker retries."""
import threading
import time
import uuid
from pathlib import Path

import pytest
import redis

from rag.config import get_settings

pytestmark = pytest.mark.integration
SAMPLES = Path(__file__).resolve().parents[1] / "data" / "samples"


@pytest.fixture
def r():
    client = redis.Redis.from_url(get_settings().redis_url)
    try:
        client.ping()
    except Exception:
        pytest.skip("Redis not running")
    return client


def test_daily_budget_is_shared_and_atomic(r):
    from rag.cost import BudgetExceeded, RedisDailyBudget

    s = get_settings().model_copy(update={"max_ocr_pages_per_day": 100})
    prefix = f"test:budget:{uuid.uuid4().hex}"
    budgets = [RedisDailyBudget(s, prefix=prefix) for _ in range(4)]   # 4 "workers"
    denied = []

    def worker(b):
        for _ in range(40):
            try:
                b.charge_ocr(1)
            except BudgetExceeded:
                denied.append(1)

    threads = [threading.Thread(target=worker, args=(b,)) for b in budgets]
    [t.start() for t in threads]
    [t.join() for t in threads]
    assert budgets[0].ocr_pages == 100            # never over the cap, despite 160 attempts
    assert len(denied) == 60
    budgets[1].refund_ocr(10)
    assert budgets[2].ocr_pages == 90
    r.delete(*r.keys(f"{prefix}:*"))


def test_rate_limiter_window(r):
    from rag.cost import RedisRateLimiter

    s = get_settings()
    lim = RedisRateLimiter(s, prefix=f"test:rl:{uuid.uuid4().hex}")
    # Wait if we're about to cross a minute boundary, so both calls hit one window.
    if 60 - time.time() % 60 < 3:
        time.sleep(3)
    assert lim.acquire("tpm", 600, 1000) == 0
    with pytest.raises(TimeoutError):
        lim.acquire("tpm", 600, 1000, max_wait_s=0.5)        # would exceed; won't wait a minute
    assert lim.acquire("big", 5000, 1000) == 0                # oversized request, empty window: allowed


def test_worker_task_indexes_and_retries(monkeypatch, tmp_path):
    """Run the Celery task in-process: success path, transient retry, and dead-lettering."""
    from celery.exceptions import Retry
    from sqlalchemy import delete

    import rag.worker as worker
    from rag.ingestion.pipeline import build_deps, register_upload
    from rag.storage.postgres import Document as DocRow
    from rag.storage.postgres import Job

    s = get_settings().model_copy(update={"llm_provider": "fake",
                                          "qdrant_collection": f"test_{uuid.uuid4().hex[:8]}"})
    try:
        deps = build_deps(settings=s)
    except Exception as e:
        pytest.skip(f"services not running: {e}")
    from rag.ingestion.pipeline import Ingestor
    monkeypatch.setattr(worker, "_ingestor", Ingestor(deps))

    def register(name):
        return register_upload(deps, SAMPLES / name, filename=name, tenant="default",
                               source_uri=f"upload://default/{name}")

    def run(reg, retries=0):
        task = worker.ingest_task
        task.push_request(retries=retries)
        try:
            return task.run(job_id=reg["job_id"], blob_key=reg["blob_key"], filename=reg["filename"],
                            tenant="default", source_uri=f"upload://default/{reg['filename']}")
        finally:
            task.pop_request()

    def job(reg):
        with deps.sessions() as db:
            return db.get(Job, uuid.UUID(reg["job_id"]))

    try:
        # 1. Success.
        reg = register("06-security-checklist.txt")
        assert reg["queue"] == "parse_cpu"
        assert run(reg)["status"] == "indexed"
        assert job(reg).status.value == "indexed" and job(reg).attempts == 1

        # 2. Transient error: task asks Celery to retry; job stays 'failed' (not dead).
        reg = register("03-npm-packages.csv")
        class RateLimitError(Exception):
            pass
        real = deps.store.add_documents
        monkeypatch.setattr(deps.store, "add_documents",
                            lambda *a, **k: (_ for _ in ()).throw(RateLimitError("429")))
        with pytest.raises(Retry):
            run(reg)
        assert job(reg).status.value == "failed" and "RateLimitError" in job(reg).error

        # ...and the retry succeeds once the API recovers. Same job row, attempts = 2.
        monkeypatch.setattr(deps.store, "add_documents", real)
        assert run(reg, retries=1)["status"] == "indexed"
        assert job(reg).status.value == "indexed" and job(reg).attempts == 2

        # 3. Non-transient error: no retry, straight to dead.
        reg = register("05-express-server.js")
        monkeypatch.setattr(deps.store, "add_documents",
                            lambda *a, **k: (_ for _ in ()).throw(ValueError("bad data")))
        assert run(reg)["status"] == "dead"
        assert job(reg).status.value == "dead"

        # Scanned files go to the OCR queue.
        assert register("09-buffers-handwritten-scan.pdf")["queue"] == "ocr_vision"
    finally:
        deps.qdrant.delete_collection(s.collection)
        with deps.sessions() as db:
            db.execute(delete(DocRow).where(DocRow.collection == s.collection))
            db.commit()
