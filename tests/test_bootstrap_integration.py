"""Runs against the docker compose services. Skipped when they're not up."""
import uuid

import pytest
from qdrant_client import models
from sqlalchemy import inspect

from rag.config import get_settings
from rag.storage.postgres import get_engine, init_schema
from rag.storage.qdrant import DENSE, PAYLOAD_INDEXES, SPARSE, ensure_collection, get_client

pytestmark = pytest.mark.integration


@pytest.fixture
def qdrant():
    client = get_client()
    try:
        client.get_collections()
    except Exception:
        pytest.skip("Qdrant is not running (docker compose up -d)")
    return client


@pytest.fixture
def scratch_settings(qdrant):
    """A throwaway collection so the test never touches real data."""
    s = get_settings().model_copy(update={"qdrant_collection": f"test_{uuid.uuid4().hex[:8]}"})
    yield s
    qdrant.delete_collection(s.collection)


def test_collection_layout_and_idempotency(qdrant, scratch_settings):
    first = ensure_collection(qdrant, scratch_settings)
    second = ensure_collection(qdrant, scratch_settings)
    assert first.created and not second.created
    assert set(first.indexes_added) == set(PAYLOAD_INDEXES) and second.indexes_added == []

    info = qdrant.get_collection(scratch_settings.collection)
    dense = info.config.params.vectors[DENSE]
    assert dense.size == scratch_settings.embedding_dim and dense.on_disk
    assert info.config.params.sparse_vectors[SPARSE].modifier == models.Modifier.IDF
    assert info.config.quantization_config.scalar.type == models.ScalarType.INT8
    assert info.config.hnsw_config.on_disk


def test_dimension_mismatch_is_rejected(qdrant, scratch_settings):
    ensure_collection(qdrant, scratch_settings)
    wrong = scratch_settings.model_copy(update={"embedding_dim": 768})
    with pytest.raises(RuntimeError, match="dense size 1024"):
        ensure_collection(qdrant, wrong)


def test_postgres_schema():
    try:
        tables = init_schema()
    except Exception:
        pytest.skip("Postgres is not running (docker compose up -d)")
    assert tables == ["api_keys", "audit_events", "documents", "jobs", "threads"]
    indexes = {i["name"] for i in inspect(get_engine()).get_indexes("jobs")}
    assert {"ix_jobs_status_queued_at", "ix_jobs_dead"} <= indexes
    from sqlalchemy import text
    with get_engine().connect() as conn:       # schema is at the latest migration
        assert conn.execute(text("select version_num from alembic_version")).scalar() == "0004"
