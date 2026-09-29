"""Reset one collection to empty so files are ingested as if for the first time.

Deletes the Qdrant collection's points (the collection is recreated empty) and
that collection's rows in Postgres (documents + jobs). With --clear-cache it
also deletes the local embedding/OCR caches, so the next ingest really calls
the API again (and costs money again).

Without --yes it only prints what it would delete.

Usage:
  .venv/Scripts/python scripts/reset.py                       # preview
  .venv/Scripts/python scripts/reset.py --yes                 # empty 'docs' (cache kept: re-ingest is free)
  .venv/Scripts/python scripts/reset.py --yes --clear-cache   # true first-time run, pays again
  .venv/Scripts/python scripts/reset.py --collection docs_fake --yes
"""
import argparse

from sqlalchemy import delete, func, select

from rag.config import get_settings
from rag.cost import CACHE_DIR, SqliteCache
from rag.storage.postgres import Document as DocRow
from rag.storage.postgres import session_factory
from rag.storage.qdrant import ensure_collection, get_client

CACHE_FILES = [CACHE_DIR / "embeddings.sqlite", CACHE_DIR / "ocr.sqlite"]


def cache_entries(path) -> int:
    if not path.exists():
        return 0
    return SqliteCache(path)._conn.execute("select count(*) from kv").fetchone()[0]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--collection")
    ap.add_argument("--clear-cache", action="store_true")
    ap.add_argument("--yes", action="store_true", help="actually delete")
    args = ap.parse_args()

    s = get_settings()
    coll = args.collection or s.collection
    s = s.model_copy(update={"qdrant_collection": coll})
    client = get_client(s)
    sessions = session_factory(s)

    points = client.count(coll).count if client.collection_exists(coll) else 0
    with sessions() as db:
        rows = db.scalar(select(func.count()).select_from(DocRow).where(DocRow.collection == coll))

    print(f"collection '{coll}': {points} Qdrant points, {rows} Postgres documents (+ their jobs)")
    if args.clear_cache:
        for p in CACHE_FILES:
            print(f"cache {p.name}: {cache_entries(p)} entries "
                  "(shared by all collections; next ingest pays the API again)")
    if not args.yes:
        print("\nPreview only. Add --yes to delete.")
        return

    if client.collection_exists(coll):
        client.delete_collection(coll)
    ensure_collection(client, s)           # recreate empty, same layout and indexes
    with sessions() as db:
        db.execute(delete(DocRow).where(DocRow.collection == coll))   # jobs cascade
        db.commit()
    if args.clear_cache:
        for p in CACHE_FILES:
            p.unlink(missing_ok=True)
    print("Done. The next ingest treats every file as new.")


if __name__ == "__main__":
    main()
