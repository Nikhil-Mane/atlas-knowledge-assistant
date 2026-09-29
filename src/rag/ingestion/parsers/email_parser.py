"""EML and MSG: headers become metadata, the body becomes text, and each
attachment is parsed as its own file (recursively, up to `ctx.max_depth`).
"""
from __future__ import annotations

import email
import email.policy
import re
import tempfile
from dataclasses import replace
from pathlib import Path

from langchain_core.documents import Document

from rag.ingestion.detect import Detection, detect
from rag.ingestion.schema import SourceInfo, make_doc


def parse_email(path: Path, det: Detection, src: SourceInfo, ctx) -> list[Document]:
    if path.suffix.lower() == ".msg":
        headers, body, attachments = _read_msg(path)
    else:
        headers, body, attachments = _read_eml(path)

    head_text = "\n".join(f"{k}: {v}" for k, v in headers.items() if v)
    meta = {"email_subject": headers.get("Subject"), "email_from": headers.get("From"),
            "email_date": headers.get("Date")}
    docs = [make_doc(f"{head_text}\n\n{body}".strip(), src, parser="email",
                     section_path="Email body", **meta)]
    docs += _parse_attachments(attachments, headers.get("Subject"), src, ctx)
    return docs


def _read_eml(path: Path):
    msg = email.message_from_bytes(path.read_bytes(), policy=email.policy.default)
    headers = {k: str(msg.get(k, "")) for k in ("From", "To", "Cc", "Date", "Subject")}
    body_part = msg.get_body(preferencelist=("plain", "html"))
    body = body_part.get_content() if body_part else ""
    if body_part is not None and body_part.get_content_type() == "text/html":
        body = _strip_html(body)
    attachments = [(part.get_filename() or "attachment.bin", part.get_payload(decode=True) or b"")
                   for part in msg.iter_attachments()]
    return headers, body, attachments


def _read_msg(path: Path):
    import extract_msg

    with extract_msg.openMsg(str(path)) as m:
        headers = {"From": m.sender, "To": m.to, "Cc": m.cc, "Date": str(m.date or ""),
                   "Subject": m.subject}
        body = m.body or _strip_html(m.htmlBody.decode("utf-8", "ignore") if m.htmlBody else "")
        attachments = [(a.longFilename or a.shortFilename or "attachment.bin", a.data or b"")
                       for a in m.attachments if isinstance(a.data, bytes)]
    return headers, body, attachments


def _parse_attachments(attachments, subject, src: SourceInfo, ctx) -> list[Document]:
    from rag.ingestion.parsers import UnsupportedFile, parse_file

    if ctx.depth >= ctx.max_depth:
        return []
    docs = []
    with tempfile.TemporaryDirectory() as tmp:
        for name, data in attachments:
            if not data:
                continue
            safe = re.sub(r"[^\w.\-]", "_", Path(name).name) or "attachment.bin"
            p = Path(tmp) / safe
            p.write_bytes(data)
            det = detect(p)
            child_src = replace(src, filename=f"{src.filename} > {safe}", mime=det.mime,
                                file_type=det.file_type)
            ctx.depth += 1
            try:
                children = parse_file(p, det, child_src, ctx)
            except UnsupportedFile:
                children = []
            finally:
                ctx.depth -= 1
            for d in children:
                # Keep the parent's identity so deletes and dedup work per email.
                d.metadata.update(doc_id=src.doc_id, source_uri=src.source_uri,
                                  sha256=src.sha256, attachment_name=safe,
                                  email_subject=subject,
                                  section_path=f"Attachment: {safe}" + (
                                      f" > {d.metadata['section_path']}"
                                      if d.metadata.get("section_path") else ""))
            docs.extend(children)
    return docs


def _strip_html(html: str) -> str:
    text = re.sub(r"(?is)<(script|style).*?</\1>", " ", html)
    text = re.sub(r"(?s)<[^>]+>", " ", text)
    return re.sub(r"[ \t]+", " ", text).strip()
