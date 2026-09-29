"""Delete conversations inactive for longer than CHAT_RETENTION_DAYS.

  .venv/Scripts/python scripts/purge.py              # preview
  .venv/Scripts/python scripts/purge.py --yes        # delete
  .venv/Scripts/python scripts/purge.py --days 7 --yes

Runs nightly in Kubernetes (deploy/k8s/purge-cronjob.yaml). Deletes every
saved turn of each expired thread (LangGraph checkpoint tables) and its
tracking row, and writes one audit event with the count.
"""
import argparse
from datetime import datetime, timedelta, timezone

from sqlalchemy import delete, select

from rag.config import get_settings
from rag.security.audit import record as audit
from rag.storage.postgres import Thread, session_factory


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", type=int)
    ap.add_argument("--yes", action="store_true", help="actually delete")
    args = ap.parse_args()

    s = get_settings()
    days = args.days if args.days is not None else s.chat_retention_days
    cutoff = datetime.now(timezone.utc) - timedelta(days=days)
    sessions = session_factory(s)
    with sessions() as db:
        keys = list(db.scalars(select(Thread.thread_key).where(Thread.last_active_at < cutoff)))
    print(f"{len(keys)} conversation(s) inactive for more than {days} days")
    if not args.yes or not keys:
        if keys:
            print("Preview only. Add --yes to delete.")
        return

    from langgraph.checkpoint.postgres import PostgresSaver
    conninfo = s.postgres_dsn.replace("postgresql+psycopg://", "postgresql://")
    with PostgresSaver.from_conn_string(conninfo) as saver:
        saver.setup()
        for key in keys:
            saver.delete_thread(key)
    with sessions() as db:
        db.execute(delete(Thread).where(Thread.thread_key.in_(keys)))
        db.commit()
    audit(sessions, "conversations_purged", detail={"count": len(keys), "older_than_days": days})
    print(f"Deleted {len(keys)} conversation(s).")


if __name__ == "__main__":
    main()
