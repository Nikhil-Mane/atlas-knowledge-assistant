"""Capacity model: turns a docs/day target into concrete resource needs.

Two different questions are answered separately, because they have different
answers on the same hardware:

* sustained: can the system process a full day's volume within the day?
* peak: can it keep up during a burst, and if not, how long until the
  backlog drains?
"""
from __future__ import annotations

import math
from dataclasses import dataclass

from rag.config import Settings

SECONDS_PER_DAY = 86_400
BYTES_PER_GB = 1024**3


@dataclass(frozen=True)
class Resources:
    """What the machine running the workers and Qdrant provides."""
    cpu_cores: int
    qdrant_ram_gb: float
    parse_workers: int | None = None      # default: cores - 2 (leave room for Qdrant/Postgres)
    peak_hours: float = 1.0               # how long a burst lasts


@dataclass(frozen=True)
class CapacityReport:
    # volume
    docs_per_day: int
    pages_per_day: float
    chunks_per_day: float
    avg_docs_per_min: float
    peak_docs_per_min: float
    # parsing (CPU)
    parse_cpu_hours_per_day: float
    parse_cores_sustained: float
    parse_workers_for_peak: int
    parse_workers_available: int
    sustained_ok: bool
    peak_backlog_pages: float
    backlog_drain_minutes: float
    # OCR (I/O bound, vision API)
    ocr_pages_per_day: float
    ocr_concurrency_peak: int
    # embeddings
    embed_tokens_per_day: float
    embed_tpm_peak: float
    embed_tpm_limit: int
    embed_requests_per_min_peak: float
    embed_ok: bool
    # Qdrant
    points_retained: float
    qdrant_ram_gb_per_day: float
    qdrant_ram_gb_retained: float
    qdrant_disk_gb_retained: float
    qdrant_days_in_ram: float
    qdrant_ok: bool


def estimate(s: Settings, r: Resources) -> CapacityReport:
    pages_day = s.docs_per_day * s.avg_pages_per_doc
    chunks_day = pages_day * s.chunks_per_page
    text_pages_day = pages_day * (1 - s.scanned_page_ratio)
    ocr_pages_day = pages_day * s.scanned_page_ratio

    # --- Parsing: CPU-bound, one page per worker process at a time.
    cpu_seconds_day = text_pages_day * s.parse_seconds_per_page
    cores_sustained = cpu_seconds_day / SECONDS_PER_DAY / s.worker_utilization
    peak_pages_per_s = text_pages_day / SECONDS_PER_DAY * s.peak_factor
    workers_for_peak = math.ceil(peak_pages_per_s * s.parse_seconds_per_page / s.worker_utilization)

    available = r.parse_workers if r.parse_workers is not None else max(1, r.cpu_cores - 2)
    capacity_pages_per_s = available / s.parse_seconds_per_page
    avg_pages_per_s = text_pages_day / SECONDS_PER_DAY
    growth = max(0.0, peak_pages_per_s - capacity_pages_per_s)
    backlog = growth * r.peak_hours * 3600
    drain_rate = capacity_pages_per_s - avg_pages_per_s
    drain_min = (backlog / drain_rate / 60) if backlog and drain_rate > 0 else (0.0 if not backlog else math.inf)

    # --- OCR: latency-bound API calls; concurrency = arrival rate x latency (Little's law).
    ocr_peak_per_s = ocr_pages_day / SECONDS_PER_DAY * s.peak_factor
    ocr_concurrency = max(1, math.ceil(ocr_peak_per_s * s.ocr_seconds_per_page))

    # --- Embeddings.
    tokens_day = chunks_day * s.tokens_per_chunk
    tpm_peak = tokens_day / 1440 * s.peak_factor
    rpm_peak = chunks_day / 1440 * s.peak_factor / s.embed_batch_size

    # --- Qdrant: int8 vector in RAM (1 B/dim) + HNSW links (~m*2 ids x 4 B, on disk
    # but hot in page cache). Originals float32 + payload (~1.5 KB) on disk.
    points = chunks_day * s.retention_days
    ram_per_point = s.embedding_dim * 1 + s.hnsw_m * 2 * 4
    disk_per_point = s.embedding_dim * 4 + 1536 + 200   # float32 + payload + sparse vector
    ram_day_gb = chunks_day * ram_per_point / BYTES_PER_GB
    ram_ret_gb = points * ram_per_point / BYTES_PER_GB
    disk_ret_gb = points * disk_per_point / BYTES_PER_GB

    return CapacityReport(
        docs_per_day=s.docs_per_day,
        pages_per_day=pages_day,
        chunks_per_day=chunks_day,
        avg_docs_per_min=s.docs_per_day / 1440,
        peak_docs_per_min=s.docs_per_day / 1440 * s.peak_factor,
        parse_cpu_hours_per_day=cpu_seconds_day / 3600,
        parse_cores_sustained=cores_sustained,
        parse_workers_for_peak=workers_for_peak,
        parse_workers_available=available,
        sustained_ok=cores_sustained <= available,
        peak_backlog_pages=backlog,
        backlog_drain_minutes=drain_min,
        ocr_pages_per_day=ocr_pages_day,
        ocr_concurrency_peak=ocr_concurrency,
        embed_tokens_per_day=tokens_day,
        embed_tpm_peak=tpm_peak,
        embed_tpm_limit=s.openai_embed_tpm_limit,
        embed_requests_per_min_peak=rpm_peak,
        embed_ok=tpm_peak <= 0.8 * s.openai_embed_tpm_limit,
        points_retained=points,
        qdrant_ram_gb_per_day=ram_day_gb,
        qdrant_ram_gb_retained=ram_ret_gb,
        qdrant_disk_gb_retained=disk_ret_gb,
        qdrant_days_in_ram=(r.qdrant_ram_gb / ram_day_gb) if ram_day_gb else math.inf,
        qdrant_ok=ram_ret_gb <= r.qdrant_ram_gb,
    )
