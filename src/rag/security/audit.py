"""Audit trail for security-relevant actions.

Recorded: uploads, document and conversation deletions, blocked questions,
admin access to another tenant, failed authentication bursts, and API-key
creation and revocation. Question text and document content are never
stored here: only who, what, which tenant and when.

Writing an audit event never fails the request that triggered it.
"""
from __future__ import annotations

import logging

from rag.storage.postgres import AuditEvent

log = logging.getLogger("rag.audit")


def record(sessions, action: str, *, tenant: str | None = None, key_id: str | None = None,
           target: str | None = None, detail: dict | None = None, ip: str | None = None) -> None:
    try:
        with sessions() as db:
            db.add(AuditEvent(action=action, tenant=tenant, key_id=key_id, target=target,
                              detail=detail, ip=ip))
            db.commit()
    except Exception as e:
        log.warning("audit write failed for %s: %s", action, e)
