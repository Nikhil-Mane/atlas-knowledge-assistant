"""Safety pass over chunks before they are embedded and indexed.

1. Secrets are always replaced (SECRET_REDACTION): nothing a search returns
   should contain a live credential.
2. PII follows PII_INGEST_POLICY: tag chunks (metadata `pii`), redact the text,
   or block the whole document.
3. Hidden instructions follow GUARD_DOC_INJECTION: tag chunks (metadata
   `suspicious`) or drop them from the index.

Runs before chunk IDs are computed, so the stored text is what gets hashed:
re-ingesting under the same policy is still free.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from langchain_core.documents import Document

from rag.security.guardrails import scan_document_chunk
from rag.security.pii import detector_for, redact_secrets


class PiiBlocked(ValueError):
    """Raised for PII_INGEST_POLICY=block; not retried (the file won't change)."""


@dataclass
class SafetyReport:
    secrets_redacted: int = 0
    pii_chunks: int = 0
    pii_entities: dict[str, int] = field(default_factory=dict)
    suspicious_chunks: int = 0
    dropped_chunks: int = 0

    def as_dict(self) -> dict:
        return {k: v for k, v in self.__dict__.items() if v}


def apply_ingest_safety(chunks: list[Document], settings) -> tuple[list[Document], SafetyReport]:
    report = SafetyReport()
    detector = detector_for(settings) if settings.pii_ingest_policy != "off" else None
    out: list[Document] = []
    for doc in chunks:
        text = doc.page_content
        if settings.secret_redaction:
            text, found = redact_secrets(text)
            report.secrets_redacted += len(found)
        if detector:
            findings = detector.find(text)
            if findings:
                kinds = sorted({f.entity for f in findings})
                if settings.pii_ingest_policy == "block":
                    raise PiiBlocked(f"document contains PII ({', '.join(kinds)}) and "
                                     "PII_INGEST_POLICY=block")
                report.pii_chunks += 1
                for f in findings:
                    report.pii_entities[f.entity] = report.pii_entities.get(f.entity, 0) + 1
                doc.metadata["pii"] = kinds
                if settings.pii_ingest_policy == "redact":
                    text, _ = detector.redact(text)
        if settings.guard_doc_injection != "off":
            reasons = scan_document_chunk(text)
            if reasons:
                report.suspicious_chunks += 1
                if settings.guard_doc_injection == "drop":
                    report.dropped_chunks += 1
                    continue
                doc.metadata["suspicious"] = reasons[:3]
        doc.page_content = text
        out.append(doc)
    return out, report
