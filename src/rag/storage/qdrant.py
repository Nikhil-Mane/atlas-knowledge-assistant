"""Qdrant client and collection bootstrap.

One collection holds every chunk. Layout chosen for ~450K new points/day:

* `dense`  named vector: OpenAI embedding, cosine, stored on disk, int8 scalar
  quantization kept in RAM (4x smaller than float32) and rescored from disk.
* `sparse` named vector: BM25 term weights; Qdrant applies IDF at query time.
* HNSW graph on disk; payload on disk.
* Payload indexes for the filters the pipeline and the query graph use.

LangChain's QdrantVectorStore stores metadata under the `metadata` key, so
indexed fields are `metadata.<name>`.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from qdrant_client import QdrantClient, models

from rag.config import Settings, get_settings

DENSE = "dense"
SPARSE = "sparse"

PAYLOAD_INDEXES: dict[str, models.PayloadSchemaType | models.KeywordIndexParams] = {
    # is_tenant groups each tenant's points together on disk, so a filtered
    # search on one tenant stays fast as the collection grows.
    "metadata.tenant": models.KeywordIndexParams(type=models.KeywordIndexType.KEYWORD, is_tenant=True),
    "metadata.doc_id": models.PayloadSchemaType.KEYWORD,
    "metadata.source_uri": models.PayloadSchemaType.KEYWORD,
    "metadata.sha256": models.PayloadSchemaType.KEYWORD,
    "metadata.file_type": models.PayloadSchemaType.KEYWORD,
    "metadata.ocr": models.PayloadSchemaType.BOOL,
    "metadata.ingested_at": models.PayloadSchemaType.DATETIME,
}


def get_client(settings: Settings | None = None) -> QdrantClient:
    s = settings or get_settings()
    return QdrantClient(
        url=s.qdrant_url,
        api_key=s.qdrant_api_key.get_secret_value() if s.qdrant_api_key else None,
        prefer_grpc=True,   # binary protocol: noticeably faster for bulk upserts
        timeout=30,
    )


@dataclass
class BootstrapResult:
    created: bool
    indexes_added: list[str] = field(default_factory=list)


def ensure_collection(client: QdrantClient, settings: Settings | None = None) -> BootstrapResult:
    """Create the collection and payload indexes if missing. Safe to run repeatedly."""
    s = settings or get_settings()
    name = s.collection
    created = False

    if not client.collection_exists(name):
        client.create_collection(
            collection_name=name,
            vectors_config={
                DENSE: models.VectorParams(
                    size=s.embedding_dim,
                    distance=models.Distance.COSINE,
                    on_disk=True,
                ),
            },
            sparse_vectors_config={
                SPARSE: models.SparseVectorParams(
                    index=models.SparseIndexParams(on_disk=True),
                    modifier=models.Modifier.IDF,
                ),
            },
            quantization_config=models.ScalarQuantization(
                scalar=models.ScalarQuantizationConfig(
                    type=models.ScalarType.INT8,
                    quantile=0.99,
                    always_ram=True,
                ),
            ),
            hnsw_config=models.HnswConfigDiff(
                m=s.hnsw_m,
                ef_construct=s.hnsw_ef_construct,
                on_disk=True,
            ),
            on_disk_payload=True,
            shard_number=s.qdrant_shards,
            replication_factor=s.qdrant_replicas,
        )
        created = True
    else:
        _check_compatible(client, s)

    existing = set((client.get_collection(name).payload_schema or {}).keys())
    added = []
    for field_name, schema in PAYLOAD_INDEXES.items():
        if field_name not in existing:
            client.create_payload_index(name, field_name=field_name, field_schema=schema, wait=True)
            added.append(field_name)
    return BootstrapResult(created=created, indexes_added=added)


def _check_compatible(client: QdrantClient, s: Settings) -> None:
    """Fail loudly if the existing collection doesn't match the configured embedding size."""
    params = client.get_collection(s.collection).config.params
    vectors = params.vectors if isinstance(params.vectors, dict) else {}
    dense = vectors.get(DENSE)
    if dense is None or dense.size != s.embedding_dim:
        found = dense.size if dense else "no 'dense' vector"
        raise RuntimeError(
            f"Collection '{s.collection}' has dense size {found}, but EMBEDDING_DIM="
            f"{s.embedding_dim}. Use a new QDRANT_COLLECTION name or drop the collection."
        )
    if SPARSE not in (params.sparse_vectors or {}):
        raise RuntimeError(f"Collection '{s.collection}' has no '{SPARSE}' sparse vector.")
