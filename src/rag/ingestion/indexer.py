"""Incremental, idempotent indexing into Qdrant.

1. Chunk IDs are derived from content, so an unchanged chunk has the same ID.
2. IDs already in Qdrant are skipped: only new chunks are embedded (the
   OpenAI cost of re-ingesting an unchanged file is zero).
3. After upserting, chunks from earlier versions of the same source that are
   no longer present are deleted with one filtered delete.
"""
from __future__ import annotations

from dataclasses import dataclass

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_qdrant import QdrantVectorStore, RetrievalMode
from qdrant_client import QdrantClient, models

from rag.config import Settings
from rag.ingestion.schema import SourceInfo, assign_chunk_ids
from rag.models import sparse_embeddings
from rag.storage.qdrant import DENSE, SPARSE

RETRIEVE_BATCH = 1000


@dataclass
class IndexResult:
    total: int
    new: int
    unchanged: int
    deleted: int


def get_vector_store(client: QdrantClient, dense: Embeddings, settings: Settings) -> QdrantVectorStore:
    return QdrantVectorStore(
        client=client,
        collection_name=settings.collection,
        embedding=dense,
        sparse_embedding=sparse_embeddings(),
        retrieval_mode=RetrievalMode.HYBRID,
        vector_name=DENSE,
        sparse_vector_name=SPARSE,
        # Layout is guaranteed by scripts/bootstrap.py; validation would cost an embedding call.
        validate_collection_config=False,
    )


def index_chunks(chunks: list[Document], src: SourceInfo, *, client: QdrantClient,
                 store: QdrantVectorStore, settings: Settings) -> IndexResult:
    if not chunks:
        # Never treat "nothing extracted" as "document is now empty": that would
        # delete every chunk of the previous version.
        raise ValueError("no text extracted from the document")

    ids = assign_chunk_ids(chunks)
    existing = _existing_ids(client, settings.collection, ids)
    new = [(i, d) for i, d in zip(ids, chunks) if i not in existing]

    # All-or-nothing budget check, so a document is never left half-embedded.
    precheck = getattr(store.embeddings, "precheck", None)
    if new and precheck:
        precheck([d.page_content for _, d in new])

    for start in range(0, len(new), settings.upsert_batch_size):
        batch = new[start:start + settings.upsert_batch_size]
        store.add_documents([d for _, d in batch], ids=[i for i, _ in batch],
                            batch_size=settings.upsert_batch_size)

    stale = models.Filter(
        must=[
            models.FieldCondition(key="metadata.tenant", match=models.MatchValue(value=src.tenant)),
            models.FieldCondition(key="metadata.source_uri", match=models.MatchValue(value=src.source_uri)),
        ],
        must_not=[models.HasIdCondition(has_id=ids)],
    )
    deleted = client.count(settings.collection, count_filter=stale, exact=True).count
    if deleted:
        client.delete(settings.collection, points_selector=models.FilterSelector(filter=stale))

    return IndexResult(total=len(ids), new=len(new), unchanged=len(ids) - len(new), deleted=deleted)


def _existing_ids(client: QdrantClient, collection: str, ids: list[str]) -> set[str]:
    found: set[str] = set()
    for start in range(0, len(ids), RETRIEVE_BATCH):
        points = client.retrieve(collection, ids=ids[start:start + RETRIEVE_BATCH],
                                 with_payload=False, with_vectors=False)
        found.update(str(p.id) for p in points)
    return found
