"""Back up (or restore) the Qdrant collection and the Postgres database.

For self-hosted deployments. Managed services (Qdrant Cloud, Azure Database
for PostgreSQL) have their own scheduled backups; use those in production and
this script for local/VM setups and before risky changes.

  .venv/Scripts/python scripts/backup.py backup                  # -> backups/<timestamp>/
  .venv/Scripts/python scripts/backup.py restore backups/20260929-190000
  .venv/Scripts/python scripts/backup.py restore <dir> --collection docs_restored --database rag_restored

Postgres dumps run `pg_dump`/`pg_restore` inside the compose `postgres`
container (no local client install needed). Raw files: back up the blob
store separately (the data/blobs folder, or storage-account replication).
"""
import argparse
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import httpx

from rag.config import PROJECT_ROOT, get_settings
from rag.storage.qdrant import get_client


def qdrant_headers(s) -> dict:
    return {"api-key": s.qdrant_api_key.get_secret_value()} if s.qdrant_api_key else {}


def backup(out: Path, collection: str) -> None:
    s = get_settings()
    out.mkdir(parents=True, exist_ok=True)
    client = get_client(s)

    snap = client.create_snapshot(collection_name=collection, wait=True)
    target = out / f"{collection}.snapshot"
    url = f"{s.qdrant_url.rstrip('/')}/collections/{collection}/snapshots/{snap.name}"
    with httpx.stream("GET", url, headers=qdrant_headers(s), timeout=None) as r:
        r.raise_for_status()
        with open(target, "wb") as f:
            for block in r.iter_bytes(1 << 20):
                f.write(block)
    client.delete_snapshot(collection_name=collection, snapshot_name=snap.name)   # free server disk
    points = client.count(collection).count
    print(f"qdrant    {collection}: {points} points -> {target.name} ({target.stat().st_size / 1e6:.1f} MB)")

    dump = out / "postgres.dump"
    with open(dump, "wb") as f:
        subprocess.run(["docker", "compose", "exec", "-T", "postgres", "pg_dump", "-U", "rag",
                        "-Fc", "rag"], stdout=f, check=True, cwd=PROJECT_ROOT)
    print(f"postgres  rag -> {dump.name} ({dump.stat().st_size / 1e6:.2f} MB)")

    (out / "manifest.json").write_text(json.dumps({
        "created": datetime.now().isoformat(timespec="seconds"), "collection": collection,
        "points": points, "qdrant_snapshot": target.name, "postgres_dump": dump.name}, indent=2))
    print(f"backup    {out}")


def restore(src: Path, collection: str | None, database: str | None) -> None:
    s = get_settings()
    manifest = json.loads((src / "manifest.json").read_text())
    collection = collection or manifest["collection"]
    snapshot = src / manifest["qdrant_snapshot"]
    url = f"{s.qdrant_url.rstrip('/')}/collections/{collection}/snapshots/upload?priority=snapshot&wait=true"
    with open(snapshot, "rb") as f:
        r = httpx.post(url, files={"snapshot": (snapshot.name, f)}, headers=qdrant_headers(s),
                       timeout=None)
    r.raise_for_status()
    print(f"qdrant    restored {get_client(s).count(collection).count} points into '{collection}'")

    database = database or "rag"
    if database != "rag":
        subprocess.run(["docker", "compose", "exec", "-T", "postgres", "createdb", "-U", "rag", database],
                       check=True, cwd=PROJECT_ROOT)
    with open(src / manifest["postgres_dump"], "rb") as f:
        subprocess.run(["docker", "compose", "exec", "-T", "postgres", "pg_restore", "-U", "rag",
                        "-d", database, "--clean", "--if-exists", "--no-owner"],
                       stdin=f, check=True, cwd=PROJECT_ROOT)
    print(f"postgres  restored into database '{database}'")


def main() -> int:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("backup")
    b.add_argument("--out")
    b.add_argument("--collection")
    r = sub.add_parser("restore")
    r.add_argument("dir")
    r.add_argument("--collection", help="restore into this collection (default: the original)")
    r.add_argument("--database", help="restore into this database (default: rag, overwriting it)")
    args = ap.parse_args()

    s = get_settings()
    if args.cmd == "backup":
        out = Path(args.out) if args.out else PROJECT_ROOT / "backups" / datetime.now().strftime("%Y%m%d-%H%M%S")
        backup(out, args.collection or s.collection)
    else:
        restore(Path(args.dir), args.collection, args.database)
    return 0


if __name__ == "__main__":
    sys.exit(main())
