"""Postgres schema: documents, ingest jobs and API keys.

At 10K docs/day the `jobs` table grows by ~3.7M rows/year, so every query the
workers and dashboards run is backed by an index. The schema is managed by
Alembic migrations in `rag/migrations` (`init_schema` applies them).
"""
from __future__ import annotations

import enum
import uuid
from datetime import datetime
from functools import lru_cache

from sqlalchemy import (
    BigInteger, Boolean, DateTime, Enum, ForeignKey, Index, Integer, String, Text, UniqueConstraint,
    create_engine, func, text,
)
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.engine import Engine
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, sessionmaker

from rag.config import Settings, get_settings


class Base(DeclarativeBase):
    pass


class JobStatus(str, enum.Enum):
    queued = "queued"
    processing = "processing"
    indexed = "indexed"
    skipped = "skipped"      # duplicate content, nothing to do
    failed = "failed"        # will be retried
    dead = "dead"            # retries exhausted; needs a human


class Document(Base):
    """A unique piece of content per Qdrant collection. (collection, sha256) is the dedup key."""
    __tablename__ = "documents"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    sha256: Mapped[str] = mapped_column(String(64), nullable=False)
    # Which Qdrant collection holds its chunks: a fake-embedding test run must
    # never make the real collection treat a file as already indexed.
    collection: Mapped[str] = mapped_column(String(128), nullable=False)
    # Stable per (tenant, source_uri) across versions; matches `metadata.doc_id` in Qdrant.
    doc_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), nullable=False)
    tenant: Mapped[str] = mapped_column(String(64), nullable=False, default="default")
    source_uri: Mapped[str] = mapped_column(Text, nullable=False)
    filename: Mapped[str] = mapped_column(Text, nullable=False)
    mime: Mapped[str | None] = mapped_column(String(128))
    file_type: Mapped[str | None] = mapped_column(String(32))
    size_bytes: Mapped[int] = mapped_column(BigInteger, nullable=False)
    blob_key: Mapped[str] = mapped_column(Text, nullable=False)
    page_count: Mapped[int | None] = mapped_column(Integer)
    ocr_pages: Mapped[int | None] = mapped_column(Integer)
    chunk_count: Mapped[int | None] = mapped_column(Integer)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    indexed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))

    __table_args__ = (
        UniqueConstraint("collection", "sha256", name="uq_documents_collection_sha256"),
        Index("ix_documents_source_uri", "source_uri"),
        Index("ix_documents_doc_id", "doc_id"),
        Index("ix_documents_tenant_created", "tenant", "created_at"),
    )


class Job(Base):
    """One ingest attempt for a document, including the retry history."""
    __tablename__ = "jobs"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    document_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("documents.id", ondelete="CASCADE"), nullable=False)
    status: Mapped[JobStatus] = mapped_column(
        Enum(JobStatus, name="job_status"), nullable=False, default=JobStatus.queued)
    queue: Mapped[str | None] = mapped_column(String(32))          # parse_cpu / ocr_vision / embed_io
    attempts: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    error: Mapped[str | None] = mapped_column(Text)
    stage_ms: Mapped[dict | None] = mapped_column(JSONB)           # per-node timings, feeds capacity tuning
    queued_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    started_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    finished_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))

    __table_args__ = (
        Index("ix_jobs_document", "document_id"),
        # Throughput and backlog queries: "how many queued/failed since X".
        Index("ix_jobs_status_queued_at", "status", "queued_at"),
        # Small partial index so the dead-letter view stays instant.
        Index("ix_jobs_dead", "finished_at", postgresql_where=text("status = 'dead'")),
    )


class ApiKey(Base):
    """A client credential. Only a SHA-256 hash of the key is stored.

    The key decides the tenant (which documents the caller can see and add)
    and the role: `user` can ingest and chat in its tenant; `admin` can also
    act on any tenant and manage keys. Limits are enforced per key.
    """
    __tablename__ = "api_keys"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    name: Mapped[str] = mapped_column(String(128), nullable=False)
    key_prefix: Mapped[str] = mapped_column(String(16), nullable=False)      # shown in listings
    key_hash: Mapped[str] = mapped_column(String(64), nullable=False, unique=True)
    tenant: Mapped[str] = mapped_column(String(64), nullable=False)
    role: Mapped[str] = mapped_column(String(16), nullable=False, default="user")
    requests_per_minute: Mapped[int] = mapped_column(Integer, nullable=False, default=60)
    questions_per_day: Mapped[int] = mapped_column(Integer, nullable=False, default=500)
    active: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    last_used_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))

    __table_args__ = (Index("ix_api_keys_tenant", "tenant"),)


class Thread(Base):
    """Ownership and activity of a conversation, for retention and deletion.

    The conversation itself lives in LangGraph's checkpoint tables under
    `thread_key` (tenant/key/thread_id).
    """
    __tablename__ = "threads"

    thread_key: Mapped[str] = mapped_column(Text, primary_key=True)
    tenant: Mapped[str] = mapped_column(String(64), nullable=False)
    key_id: Mapped[str] = mapped_column(String(64), nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    last_active_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())

    __table_args__ = (Index("ix_threads_last_active", "last_active_at"),)


class AuditEvent(Base):
    """Security-relevant actions: who did what, to which tenant, from where."""
    __tablename__ = "audit_events"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    ts: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    action: Mapped[str] = mapped_column(String(64), nullable=False)
    tenant: Mapped[str | None] = mapped_column(String(64))
    key_id: Mapped[str | None] = mapped_column(String(64))
    target: Mapped[str | None] = mapped_column(Text)
    detail: Mapped[dict | None] = mapped_column(JSONB)
    ip: Mapped[str | None] = mapped_column(String(64))

    __table_args__ = (Index("ix_audit_ts", "ts"), Index("ix_audit_tenant_ts", "tenant", "ts"))


@lru_cache
def get_engine(dsn: str | None = None) -> Engine:
    s = get_settings()
    return create_engine(
        dsn or s.postgres_dsn,
        pool_size=s.db_pool_size,
        max_overflow=s.db_max_overflow,
        pool_pre_ping=True,     # survive Postgres restarts without stale-connection errors
        connect_args={"connect_timeout": 5},   # fail fast instead of hanging when Postgres is down
    )


def session_factory(settings: Settings | None = None) -> sessionmaker:
    s = settings or get_settings()
    return sessionmaker(bind=get_engine(s.postgres_dsn), expire_on_commit=False)


def alembic_config(dsn: str):
    from alembic.config import Config

    cfg = Config()
    cfg.set_main_option("script_location", "rag:migrations")
    cfg.set_main_option("sqlalchemy.url", dsn.replace("%", "%%"))
    return cfg


def init_schema(engine: Engine | None = None, dsn: str | None = None) -> list[str]:
    """Apply all migrations. Returns the application tables that exist afterwards.

    A database created before migrations existed (tables present, no
    `alembic_version`) is stamped at the first revision, then upgraded.
    """
    from alembic import command
    from sqlalchemy import inspect

    dsn = dsn or get_settings().postgres_dsn
    engine = engine or get_engine(dsn)
    cfg = alembic_config(dsn)
    tables = set(inspect(engine).get_table_names())
    if "documents" in tables and "alembic_version" not in tables:
        command.stamp(cfg, "0001")
    command.upgrade(cfg, "head")
    return sorted(set(inspect(engine).get_table_names()) & set(Base.metadata.tables))
