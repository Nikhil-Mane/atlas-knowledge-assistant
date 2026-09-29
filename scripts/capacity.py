"""Show whether a machine can handle the configured docs/day target.

Usage:
  .venv/Scripts/python scripts/capacity.py                       # this machine
  .venv/Scripts/python scripts/capacity.py --cores 16 --qdrant-ram-gb 64
  .venv/Scripts/python scripts/capacity.py --docs-per-day 20000

Per-page timings are estimates until Phase 2 measures them on real files
(set PARSE_SECONDS_PER_PAGE / OCR_SECONDS_PER_PAGE in .env).
"""
import argparse
import math
import os

from rag.capacity import Resources, estimate
from rag.config import get_settings


def fmt(n: float) -> str:
    if n >= 1e6:
        return f"{n / 1e6:,.1f}M"
    if n >= 1e3:
        return f"{n / 1e3:,.1f}K"
    return f"{n:,.1f}"


def mark(ok: bool) -> str:
    return "OK  " if ok else "SHORT"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cores", type=int, default=os.cpu_count() or 4)
    ap.add_argument("--qdrant-ram-gb", type=float, default=1.0,
                    help="RAM available to Qdrant for vectors (default: docker-compose limit minus overhead)")
    ap.add_argument("--parse-workers", type=int, default=None)
    ap.add_argument("--peak-hours", type=float, default=1.0)
    ap.add_argument("--docs-per-day", type=int, default=None)
    args = ap.parse_args()

    s = get_settings()
    if args.docs_per_day:
        s = s.model_copy(update={"docs_per_day": args.docs_per_day})
    r = Resources(cpu_cores=args.cores, qdrant_ram_gb=args.qdrant_ram_gb,
                  parse_workers=args.parse_workers, peak_hours=args.peak_hours)
    c = estimate(s, r)

    print(f"\nTarget: {c.docs_per_day:,} docs/day  "
          f"({c.avg_docs_per_min:.1f}/min average, {c.peak_docs_per_min:.1f}/min at {s.peak_factor:g}x peak)")
    print(f"Volume: {fmt(c.pages_per_day)} pages/day -> {fmt(c.chunks_per_day)} chunks/day\n")

    print("PARSING (CPU)")
    print(f"  {mark(c.sustained_ok)} sustained: {c.parse_cpu_hours_per_day:.0f} CPU-hours/day "
          f"= {c.parse_cores_sustained:.1f} worker processes; {c.parse_workers_available} available")
    peak_ok = c.parse_workers_available >= c.parse_workers_for_peak
    drain = "never (add workers)" if math.isinf(c.backlog_drain_minutes) else f"{c.backlog_drain_minutes:.0f} min"
    print(f"  {mark(peak_ok)} peak: {c.parse_workers_for_peak} workers keep up with no backlog"
          + ("" if peak_ok else f"; with {c.parse_workers_available}, a {r.peak_hours:g} h burst queues "
             f"{fmt(c.peak_backlog_pages)} pages that drain in {drain}"))

    print("\nOCR (vision API, I/O-bound)")
    print(f"  OK   {fmt(c.ocr_pages_per_day)} scanned pages/day; "
          f"{c.ocr_concurrency_peak} concurrent vision calls at peak")

    print("\nEMBEDDINGS (OpenAI)")
    print(f"  {mark(c.embed_ok)} {fmt(c.embed_tokens_per_day)} tokens/day; peak {fmt(c.embed_tpm_peak)} TPM "
          f"vs limit {fmt(c.embed_tpm_limit)} (80% budget); {c.embed_requests_per_min_peak:.0f} requests/min")

    print(f"\nQDRANT ({s.retention_days} days retention)")
    print(f"  {fmt(c.points_retained)} points; RAM {c.qdrant_ram_gb_per_day:.2f} GB/day "
          f"-> {c.qdrant_ram_gb_retained:,.0f} GB retained; disk {c.qdrant_disk_gb_retained:,.0f} GB")
    print(f"  {mark(c.qdrant_ok)} {r.qdrant_ram_gb:g} GB available holds "
          f"{c.qdrant_days_in_ram:.1f} days of full-volume data in RAM")
    print()


if __name__ == "__main__":
    main()
