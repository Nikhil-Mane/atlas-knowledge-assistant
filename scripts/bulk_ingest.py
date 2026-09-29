"""Ingest files or folders and print what happened to each.

Cost-safe workflow (see data/samples/README.md):
  1. --dry-run                  parse + chunk locally, print tokens / OCR pages / est. cost. No API calls.
  2. --no-ocr                   real embeddings for text files only (cheapest real run)
  3. --only "09-*" --max-ocr-pages 1   one real OCR page

Usage:
  .venv/Scripts/python scripts/bulk_ingest.py data/samples --dry-run
  .venv/Scripts/python scripts/bulk_ingest.py data/samples --no-ocr --max-embed-tokens 20000
  .venv/Scripts/python scripts/bulk_ingest.py data/samples --fake-embeddings   # no API at all

Caps default to MAX_EMBED_TOKENS_PER_RUN / MAX_OCR_PAGES_PER_RUN in .env. The
run stops at the first file that would exceed a cap; nothing past it is sent.
"""
import argparse
import fnmatch
import statistics
import sys
import time
from pathlib import Path

from rag.config import get_settings
from rag.cost import reset_budget

SKIP_NAMES = {"README.md", ".gitkeep"}
# Errors that mean the configuration is wrong, not the file: stop instead of
# repeating the same failing call for every remaining file.
FATAL_ERRORS = ("NotFoundError", "AuthenticationError", "PermissionDeniedError", "MissingConfig",
                "APIConnectionError")


def preflight(s) -> bool:
    """One ~1-token embedding call so a bad endpoint/key/deployment fails before any file."""
    from rag.models import model_id, raw_dense_embeddings

    try:
        dims = len(raw_dense_embeddings(s).embed_query("ping"))
    except Exception as e:
        print(f"Preflight failed for {model_id(s, 'embedding')}: {type(e).__name__}: {str(e)[:200]}")
        print("Nothing was ingested. Check .env, then run scripts/check_llm.py.")
        return False
    if dims != s.embedding_dim:
        print(f"Preflight: deployment returned {dims} dims but EMBEDDING_DIM={s.embedding_dim}. "
              "Fix EMBEDDING_DIM (and use a new QDRANT_COLLECTION).")
        return False
    return True


def iter_files(paths: list[str], only: str | None):
    for p in map(Path, paths):
        files = sorted(f for f in p.rglob("*") if f.is_file()) if p.is_dir() else [p]
        for f in files:
            if f.name in SKIP_NAMES or (only and not fnmatch.fnmatch(f.name, only)):
                continue
            yield f


def dry_run(files, s) -> int:
    from rag.ingestion.pipeline import estimate_file

    print(f"DRY RUN: no API calls, nothing written. Provider: {s.llm_provider}\n")
    print(f"{'file':38} {'route':11} {'pages':>5} {'chunks':>6} {'tokens':>8} {'ocr pg':>6}  note")
    tokens = ocr_pages = ocr_tokens = 0
    for f in files:
        r = estimate_file(f, s)
        tokens += r["tokens"]
        ocr_pages += r["ocr_pages"]
        ocr_tokens += r.get("ocr_tokens_est", 0)
        note = r.get("reason") or (f"{r['tokens_total'] - r['tokens']:,} tokens cached"
                                   if r.get("tokens_total", 0) > r["tokens"] else "")
        print(f"{r['file'][:38]:38} {r['route']:11} {r.get('pages') or '-':>5} "
              f"{r.get('chunks', '-'):>6} {r['tokens']:>8,} {r['ocr_pages'] or '-':>6}  {note}")

    embed_total = tokens + ocr_tokens
    cost = embed_total / 1e6 * s.embed_price_per_1m_tokens + ocr_pages * s.ocr_price_per_page
    print(f"\nA real run would send: {embed_total:,} embedding tokens "
          f"({ocr_tokens:,} of them estimated OCR text) and {ocr_pages} pages to the vision model.")
    print(f"Estimated cost: ${cost:.4f}  (prices from .env: ${s.embed_price_per_1m_tokens}/1M tokens, "
          f"${s.ocr_price_per_page}/page; check them against your Azure pricing)")
    print(f"Current caps: {s.max_embed_tokens_per_run:,} tokens, {s.max_ocr_pages_per_run} OCR pages")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("paths", nargs="+")
    ap.add_argument("--tenant", default="default")
    ap.add_argument("--only", help="filename glob, e.g. '0[1-8]-*'")
    ap.add_argument("--dry-run", action="store_true", help="estimate cost; no API calls, no writes")
    ap.add_argument("--fake-embeddings", action="store_true", help="LLM_PROVIDER=fake for this run: no API at all, separate _fake collection")
    ap.add_argument("--no-ocr", action="store_true", help="skip files and pages that need OCR")
    ap.add_argument("--max-embed-tokens", type=int)
    ap.add_argument("--max-ocr-pages", type=int)
    ap.add_argument("--force", action="store_true",
                    help="re-index even if already indexed (cached content stays free)")
    args = ap.parse_args()

    s = get_settings()
    overrides = {}
    if args.fake_embeddings:
        overrides["llm_provider"] = "fake"
    if args.no_ocr:
        overrides["ocr_enabled"] = False
    if args.max_embed_tokens is not None:
        overrides["max_embed_tokens_per_run"] = args.max_embed_tokens
    if args.max_ocr_pages is not None:
        overrides["max_ocr_pages_per_run"] = args.max_ocr_pages
    s = s.model_copy(update=overrides)
    budget = reset_budget(s)

    files = list(iter_files(args.paths, args.only))
    if args.dry_run:
        return dry_run(files, s)

    from rag.ingestion.pipeline import Ingestor, build_deps

    if s.llm_provider != "fake" and not preflight(s):
        return 1
    ingestor = Ingestor(build_deps(fake_embeddings=args.fake_embeddings, settings=s))
    print(f"provider: {s.llm_provider}   collection: {ingestor.deps.settings.collection}   "
          f"caps: {s.max_embed_tokens_per_run:,} tokens / {s.max_ocr_pages_per_run} OCR pages"
          f"{'   OCR off' if not s.ocr_enabled else ''}\n")
    print(f"{'file':38} {'status':8} {'route':11} {'pages':>5} {'ocr':>4} {'chunks':>6} "
          f"{'new':>4} {'del':>4} {'ms':>7}")

    results, t0 = [], time.perf_counter()
    for f in files:
        r = ingestor.ingest(f, tenant=args.tenant, force=args.force)
        results.append(r)
        ms = sum(r.get("stage_ms", {}).values())
        print(f"{r['file'][:38]:38} {r['status']:8} {r.get('route', '-'):11} "
              f"{r.get('pages') or '-':>5} {r.get('ocr_pages') or '-':>4} {r.get('chunks', '-'):>6} "
              f"{r.get('new', '-'):>4} {r.get('deleted', '-'):>4} {ms:>7}")
        if r.get("error"):
            print(f"{'':38} ! {r['error'][:160]}")
            if r["error"].startswith("BudgetExceeded"):
                print("\nStopped: cost cap reached. Nothing further was sent.")
                break
            if r["error"].startswith(FATAL_ERRORS):
                print("\nStopped: this is a configuration error, so every remaining file would "
                      "fail the same way. Fix .env and run scripts/check_llm.py.")
                break
        elif r.get("reason"):
            print(f"{'':38} - {r['reason']}")

    elapsed = time.perf_counter() - t0
    ok = [r for r in results if r["status"] == "indexed"]
    print(f"\n{len(results)} files in {elapsed:.1f}s: {len(ok)} indexed, "
          f"{sum(r['status'] == 'skipped' for r in results)} skipped, "
          f"{sum(r['status'] in ('failed', 'dead') for r in results)} failed")
    if s.llm_provider != "fake":
        print(budget.summary(s))

    # Measured parse cost per text page: feeds PARSE_SECONDS_PER_PAGE in the capacity model.
    per_page = [r["stage_ms"]["pdf_text"] / 1000 / max(1, r["pages"] - r.get("ocr_pages", 0))
                for r in ok if r.get("route") == "pdf_text" and r.get("pages")]
    if per_page:
        print(f"measured Docling cost: {statistics.median(per_page):.2f} s/page "
              f"(median of {len(per_page)} PDFs; the first includes model loading)")
    return 0 if all(r["status"] in ("indexed", "skipped") for r in results) else 1


if __name__ == "__main__":
    sys.exit(main())
