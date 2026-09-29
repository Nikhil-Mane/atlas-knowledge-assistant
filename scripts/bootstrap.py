"""Create the Qdrant collection, payload indexes, Postgres tables and blob folder.

Safe to run any number of times.
Usage:  .venv/Scripts/python scripts/bootstrap.py
"""
from rag.config import get_settings
from rag.storage.blob import get_blob_store
from rag.storage.postgres import init_schema
from rag.storage.qdrant import ensure_collection, get_client


def main() -> None:
    s = get_settings()

    result = ensure_collection(get_client(s), s)
    state = "created" if result.created else "already exists"
    print(f"qdrant    collection '{s.collection}' {state}")
    for name in result.indexes_added:
        print(f"          + payload index {name}")

    tables = init_schema()
    print(f"postgres  tables ready: {', '.join(tables)}")

    get_blob_store(s).check()
    print(f"blobs     {s.blob_backend} store ready ({s.blob_root if s.blob_backend == 'local' else s.s3_bucket})")


if __name__ == "__main__":
    main()
