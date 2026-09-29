"""End-to-end ingestion against Qdrant + Postgres with fake embeddings.

Checks the three properties the 10K docs/day design depends on:
idempotency, duplicate skipping, and incremental updates.
"""
import uuid

import pytest
from qdrant_client import models
from sqlalchemy import delete

from rag.config import get_settings
from rag.ingestion.pipeline import Ingestor, build_deps
from rag.storage.postgres import Document as DocRow

pytestmark = pytest.mark.integration


@pytest.fixture
def ingestor():
    s = get_settings().model_copy(update={"qdrant_collection": f"test_{uuid.uuid4().hex[:8]}"})
    try:
        deps = build_deps(fake_embeddings=True, settings=s)
    except Exception as e:
        pytest.skip(f"services not running: {e}")
    yield Ingestor(deps)
    deps.qdrant.delete_collection(deps.settings.collection)
    with deps.sessions() as db:     # jobs rows go with them (ON DELETE CASCADE)
        db.execute(delete(DocRow).where(DocRow.tenant.like("test-%")))
        db.commit()


def count(ing, source_uri):
    f = models.Filter(must=[models.FieldCondition(
        key="metadata.source_uri", match=models.MatchValue(value=source_uri))])
    return ing.deps.qdrant.count(ing.deps.settings.collection, count_filter=f).count


def test_reingest_and_update(ingestor, tmp_path):
    tenant = f"test-{uuid.uuid4().hex[:6]}"
    doc = tmp_path / "notes.md"
    doc.write_text("# Streams\n\nUse pipeline() so errors propagate.\n\n"
                   "# Buffers\n\nBuffer.alloc is zero-filled and safe.\n")
    uri = f"test://{tenant}/notes.md"

    first = ingestor.ingest(doc, tenant=tenant, source_uri=uri)
    assert first["status"] == "indexed" and first["new"] == first["chunks"] > 0
    assert count(ingestor, uri) == first["chunks"]

    # Same bytes again: skipped before parsing, nothing embedded.
    again = ingestor.ingest(doc, tenant=tenant, source_uri=uri)
    assert again["status"] == "skipped" and "duplicate" in again["reason"]

    # Edit one section: only the changed chunk is embedded, the old one deleted.
    doc.write_text("# Streams\n\nUse pipeline() so errors propagate.\n\n"
                   "# Buffers\n\nBuffer.allocUnsafe is fast but may contain old data.\n")
    updated = ingestor.ingest(doc, tenant=tenant, source_uri=uri)
    assert updated["status"] == "indexed"
    assert updated["unchanged"] >= 1 and updated["new"] >= 1 and updated["deleted"] >= 1
    assert count(ingestor, uri) == updated["chunks"]


def test_empty_file_fails_without_deleting(ingestor, tmp_path):
    empty = tmp_path / "empty.txt"
    empty.write_text("   ")
    r = ingestor.ingest(empty, tenant="test-empty", source_uri=f"test://{uuid.uuid4()}/empty.txt")
    assert r["status"] == "failed" and "no text extracted" in r["error"]


def test_missing_chunks_revert_and_force(ingestor, tmp_path):
    tenant = f"test-{uuid.uuid4().hex[:6]}"
    uri = f"test://{tenant}/guide.md"
    doc = tmp_path / "guide.md"
    v1 = "# Cluster\n\nfork one worker per CPU with cluster.fork().\n"
    v2 = "# Cluster\n\nPM2 runs one worker per CPU: pm2 start app.js -i max.\n"
    coll = ingestor.deps.settings.collection

    doc.write_text(v1)
    assert ingestor.ingest(doc, tenant=tenant, source_uri=uri)["status"] == "indexed"

    # 1. Chunks deleted from Qdrant behind Postgres' back -> re-indexed, not skipped.
    ingestor.deps.qdrant.delete(coll, points_selector=models.FilterSelector(filter=models.Filter(
        must=[models.FieldCondition(key="metadata.source_uri", match=models.MatchValue(value=uri))])))
    r = ingestor.ingest(doc, tenant=tenant, source_uri=uri)
    assert r["status"] == "indexed" and "missing in Qdrant" in r["reason"]
    assert count(ingestor, uri) == r["chunks"]

    # 2. Edit, then undo the edit -> v1 content must be indexed again, not "duplicate".
    doc.write_text(v2)
    assert ingestor.ingest(doc, tenant=tenant, source_uri=uri)["status"] == "indexed"
    doc.write_text(v1)
    r = ingestor.ingest(doc, tenant=tenant, source_uri=uri)
    assert r["status"] == "indexed" and r["deleted"] >= 1
    hits = ingestor.deps.qdrant.scroll(coll, scroll_filter=models.Filter(must=[models.FieldCondition(
        key="metadata.source_uri", match=models.MatchValue(value=uri))]), with_payload=True)[0]
    assert all("cluster.fork()" in p.payload["page_content"] for p in hits)

    # 3. Unchanged file: skipped normally, re-indexed with force (nothing new to embed).
    assert ingestor.ingest(doc, tenant=tenant, source_uri=uri)["status"] == "skipped"
    r = ingestor.ingest(doc, tenant=tenant, source_uri=uri, force=True)
    assert r["status"] == "indexed" and r["new"] == 0 and "--force" in r["reason"]
