"""Service checks shared by `scripts/health.py` and the API's /health endpoint."""
from __future__ import annotations

import redis
from sqlalchemy import inspect, text

from rag.config import Settings
from rag.storage.blob import get_blob_store
from rag.storage.postgres import get_engine
from rag.storage.qdrant import DENSE, SPARSE, get_client


def check_qdrant(s: Settings) -> str:
    info = get_client(s).get_collection(s.collection)
    p = info.config.params
    dense = p.vectors[DENSE]
    quant = info.config.quantization_config
    return (f"collection '{s.collection}': {info.points_count} points, status {info.status.value}, "
            f"dense {dense.size}-d {dense.distance.value} on_disk={dense.on_disk}, "
            f"sparse={'yes' if SPARSE in (p.sparse_vectors or {}) else 'NO'}, "
            f"int8={'yes' if quant and quant.scalar else 'NO'}, shards={p.shard_number}, "
            f"indexes={len(info.payload_schema)}")


def check_postgres(s: Settings) -> str:
    engine = get_engine(s.postgres_dsn)
    with engine.connect() as conn:
        conn.execute(text("select 1"))
        max_conn = conn.execute(text("show max_connections")).scalar()
    missing = {"documents", "jobs"} - set(inspect(engine).get_table_names())
    if missing:
        raise RuntimeError(f"missing tables {sorted(missing)}; run scripts/bootstrap.py")
    return f"tables documents, jobs; max_connections={max_conn}"


def check_redis(s: Settings) -> str:
    r = redis.Redis.from_url(s.redis_url, socket_timeout=3)
    r.ping()
    policy = r.config_get("maxmemory-policy")["maxmemory-policy"]
    aof = r.config_get("appendonly")["appendonly"]
    if policy != "noeviction":
        raise RuntimeError(f"maxmemory-policy is '{policy}'; queued jobs could be evicted, set 'noeviction'")
    queued = {q: r.llen(q) for q in ("parse_cpu", "ocr_vision")}
    return f"maxmemory-policy={policy}, appendonly={aof}, queue depth {queued}"


def check_blobs(s: Settings) -> str:
    get_blob_store(s).check()
    return f"{s.blob_backend} store writable"


CHECKS = {"qdrant": check_qdrant, "postgres": check_postgres, "redis": check_redis,
          "blobs": check_blobs}


def run_checks(s: Settings) -> dict[str, dict]:
    results = {}
    for name, check in CHECKS.items():
        try:
            results[name] = {"ok": True, "detail": check(s)}
        except Exception as e:  # report every failure, not just the first
            results[name] = {"ok": False, "detail": f"{type(e).__name__}: {e}"}
    return results
