"""Load test: push synthetic documents through the API and real workers, measure throughput.

Every generated document is unique (no duplicate skipping), so each one is
fully parsed, chunked, embedded and indexed. Run it in fake mode for free:
the API and workers must be started with LLM_PROVIDER=fake.

  # terminal 1: API       (INGEST_MODE=queue)
  # terminal 2..n: workers celery -A rag.worker worker -Q parse_cpu --pool solo
  .venv/Scripts/python scripts/load_test.py --count 30 --rate 35 --workers 2

The report ends with a projection: docs/day at the measured rate and the
number of workers needed for DOCS_PER_DAY at AVG_PAGES_PER_DOC.
"""
import argparse
import itertools
import math
import random
import statistics
import sys
import tempfile
import time
import uuid
from datetime import datetime
from pathlib import Path

import httpx

from rag.config import get_settings

WORDS = ("event loop stream buffer promise callback module require import export cluster worker "
         "thread pool libuv microtask timer socket request response middleware router express "
         "error handler async await pipeline backpressure memory heap garbage collector npm package "
         "dependency version semver test mock deploy container process signal environment").split()


def paragraph(rng: random.Random, n: int = 60) -> str:
    return " ".join(rng.choice(WORDS) for _ in range(n)).capitalize() + "."


def make_pdf(path: Path, pages: int, tag: str, rng: random.Random) -> None:
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import getSampleStyleSheet
    from reportlab.platypus import PageBreak, Paragraph, SimpleDocTemplate

    st = getSampleStyleSheet()
    story = []
    for p in range(pages):
        story.append(Paragraph(f"Section {p + 1}: load test document {tag}", st["Heading2"]))
        story += [Paragraph(paragraph(rng), st["BodyText"]) for _ in range(6)]
        if p < pages - 1:
            story.append(PageBreak())
    SimpleDocTemplate(str(path), pagesize=A4).build(story)


def make_doc(kind: str, folder: Path, pages: int, rng: random.Random) -> tuple[Path, int]:
    tag = uuid.uuid4().hex[:10]           # unique content => no duplicate skipping
    if kind == "pdf":
        path = folder / f"load-{tag}.pdf"
        make_pdf(path, pages, tag, rng)
        return path, pages
    if kind == "md":
        path = folder / f"load-{tag}.md"
        path.write_text("\n\n".join(f"## Topic {i} ({tag})\n\n{paragraph(rng)}" for i in range(8)))
        return path, 1
    path = folder / f"load-{tag}.csv"
    path.write_text("id,name,description\n" + "\n".join(
        f"{i},{tag}-{rng.choice(WORDS)},{paragraph(rng, 12)}" for i in range(60)))
    return path, 1


def percentile(values: list[float], p: float) -> float:
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, max(0, math.ceil(p / 100 * len(ordered)) - 1))]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--api-url", default="http://127.0.0.1:8000")
    ap.add_argument("--api-key")
    ap.add_argument("--count", type=int, default=30)
    ap.add_argument("--rate", type=float, default=35, help="uploads per minute (5x peak = 35)")
    ap.add_argument("--pages", type=int, default=5, help="pages per generated PDF")
    ap.add_argument("--mix", default="pdf,pdf,md,csv", help="document kinds, cycled")
    ap.add_argument("--workers", type=int, default=1, help="parse worker processes running (for the projection)")
    ap.add_argument("--timeout-min", type=float, default=30)
    args = ap.parse_args()

    s = get_settings()
    headers = {"X-API-Key": args.api_key} if args.api_key else {}
    api = httpx.Client(base_url=args.api_url, headers=headers, timeout=60)
    health = api.get("/health").json()
    if health.get("provider") != "fake":
        print(f"The API is running with provider '{health.get('provider')}'. A load test would "
              "spend real money. Restart the API and workers with LLM_PROVIDER=fake.")
        return 1

    rng = random.Random(42)
    kinds = itertools.cycle(args.mix.split(","))
    jobs: dict[str, dict] = {}
    interval = 60 / args.rate
    print(f"Uploading {args.count} documents at {args.rate:g}/min ({args.mix}, {args.pages}-page PDFs)...")
    t_start = time.time()
    max_depth = 0
    with tempfile.TemporaryDirectory() as tmp:
        for i in range(args.count):
            path, pages = make_doc(next(kinds), Path(tmp), args.pages, rng)
            with open(path, "rb") as f:
                r = api.post("/ingest", files={"files": (path.name, f)})
            r.raise_for_status()
            job = r.json()["jobs"][0]
            jobs[job["job_id"]] = {"pages": pages, "kind": path.suffix[1:], "status": job["status"]}
            depth = sum(api.get("/stats").json().get("queue_depth", {}).values())
            max_depth = max(max_depth, depth)
            sleep = t_start + (i + 1) * interval - time.time()
            if sleep > 0:
                time.sleep(sleep)

    print(f"All uploaded in {time.time() - t_start:.0f}s. Waiting for workers...")
    deadline = time.time() + args.timeout_min * 60
    pending = {j for j, v in jobs.items() if v["status"] in ("queued", "processing")}
    while pending and time.time() < deadline:
        for jid in list(pending):
            info = api.get(f"/jobs/{jid}").json()
            jobs[jid].update(info)
            if info["status"] in ("indexed", "skipped", "dead"):
                pending.discard(jid)
        depth = sum(api.get("/stats").json().get("queue_depth", {}).values())
        max_depth = max(max_depth, depth)
        print(f"\r  done {len(jobs) - len(pending)}/{len(jobs)}   queue depth {depth}   ", end="", flush=True)
        if pending:
            time.sleep(2)
    print()

    done = [j for j in jobs.values() if j.get("status") == "indexed"]
    failed = [j for j in jobs.values() if j.get("status") in ("dead", "failed")]
    if not done:
        print("No document finished. Are workers running (celery -A rag.worker worker ...)?")
        return 1
    ts = lambda v: datetime.fromisoformat(str(v)).timestamp()  # noqa: E731
    latencies = [ts(j["finished_at"]) - ts(j["queued_at"]) for j in done]
    first = min(ts(j["queued_at"]) for j in done)
    last = min(max(ts(j["finished_at"]) for j in done), time.time())
    per_min = len(done) / max(1e-9, (last - first) / 60)
    pdf_spp = [j["stage_ms"]["pdf_text"] / 1000 / j["pages"] for j in done
               if j["kind"] == "pdf" and j.get("stage_ms") and "pdf_text" in j["stage_ms"]]
    stages: dict[str, list[int]] = {}
    for j in done:
        for k, v in (j.get("stage_ms") or {}).items():
            stages.setdefault(k, []).append(v)

    print(f"\nindexed {len(done)}/{len(jobs)}   failed {len(failed)}   max queue depth {max_depth}")
    print(f"throughput: {per_min:.1f} docs/min with {args.workers} parse worker(s)")
    print(f"job latency (queued -> indexed): p50 {percentile(latencies, 50):.1f}s   "
          f"p95 {percentile(latencies, 95):.1f}s   max {max(latencies):.1f}s")
    print("median stage time: " + "   ".join(f"{k} {statistics.median(v) / 1000:.2f}s"
                                            for k, v in sorted(stages.items())))
    if pdf_spp:
        spp = statistics.median(pdf_spp)
        print(f"measured Docling cost: {spp:.2f} s/page (median of {len(pdf_spp)} PDFs)")
        pages_day = s.docs_per_day * s.avg_pages_per_doc * (1 - s.scanned_page_ratio)
        need = math.ceil(pages_day * spp / 86_400 / s.worker_utilization)
        need_peak = math.ceil(pages_day * spp / 86_400 * s.peak_factor / s.worker_utilization)
        print(f"\nprojection for {s.docs_per_day:,} docs/day x {s.avg_pages_per_doc:g} pages "
              f"(PARSE_SECONDS_PER_PAGE={spp:.2f}):")
        print(f"  {need} parse workers sustain the daily volume; {need_peak} keep up with a "
              f"{s.peak_factor:g}x peak without backlog.")
        print("  Update .env with PARSE_SECONDS_PER_PAGE and run scripts/capacity.py for the full report.")
    for j in failed[:5]:
        print(f"  failed: {j.get('filename')}: {j.get('error')}")
    return 0 if not failed else 1


if __name__ == "__main__":
    sys.exit(main())
