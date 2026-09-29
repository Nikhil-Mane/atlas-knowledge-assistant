"""Create, list and revoke API keys.

  .venv/Scripts/python scripts/api_keys.py create --name "Support team" --tenant acme
  .venv/Scripts/python scripts/api_keys.py create --name ops --tenant default --role admin
  .venv/Scripts/python scripts/api_keys.py list
  .venv/Scripts/python scripts/api_keys.py revoke rag_AbCdEfGh

The key is printed once at creation and never stored in plain text.
"""
import argparse

from sqlalchemy import select

from rag.auth import create_key
from rag.config import get_settings
from rag.security.audit import record as audit
from rag.storage.postgres import ApiKey, session_factory


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("create")
    c.add_argument("--name", required=True)
    c.add_argument("--tenant", required=True)
    c.add_argument("--role", choices=["user", "admin"], default="user")
    c.add_argument("--rpm", type=int, default=60, help="requests per minute")
    c.add_argument("--qpd", type=int, default=500, help="questions per day")
    sub.add_parser("list")
    r = sub.add_parser("revoke")
    r.add_argument("prefix", help="the key's first 12 characters, as shown by 'list'")
    args = ap.parse_args()

    sessions = session_factory(get_settings())
    if args.cmd == "create":
        plain, row = create_key(sessions, name=args.name, tenant=args.tenant, role=args.role,
                                requests_per_minute=args.rpm, questions_per_day=args.qpd)
        audit(sessions, "api_key_created", tenant=row.tenant, key_id=str(row.id),
              detail={"name": row.name, "role": row.role, "prefix": row.key_prefix})
        print(f"Created {row.role} key '{row.name}' for tenant '{row.tenant}'.")
        print(f"\n  {plain}\n\nStore it now: it cannot be shown again.")
    elif args.cmd == "list":
        with sessions() as db:
            rows = db.scalars(select(ApiKey).order_by(ApiKey.created_at)).all()
        print(f"{'prefix':13} {'name':24} {'tenant':12} {'role':6} {'rpm':>5} {'q/day':>6} {'active':6} last used")
        for k in rows:
            print(f"{k.key_prefix:13} {k.name[:24]:24} {k.tenant[:12]:12} {k.role:6} "
                  f"{k.requests_per_minute:>5} {k.questions_per_day:>6} {str(k.active):6} "
                  f"{k.last_used_at or '-'}")
    else:
        with sessions() as db:
            rows = db.scalars(select(ApiKey).where(ApiKey.key_prefix == args.prefix)).all()
            for k in rows:
                k.active = False
            db.commit()
        for k in rows:
            audit(sessions, "api_key_revoked", tenant=k.tenant, key_id=str(k.id),
                  detail={"prefix": k.key_prefix})
        print(f"Revoked {len(rows)} key(s). Takes effect within a minute (API cache).")


if __name__ == "__main__":
    main()
