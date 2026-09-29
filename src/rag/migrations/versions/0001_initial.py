"""documents and jobs

Revision ID: 0001
Revises:
"""
import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision = "0001"
down_revision = None
branch_labels = None
depends_on = None

job_status = postgresql.ENUM("queued", "processing", "indexed", "skipped", "failed", "dead",
                             name="job_status", create_type=False)


def upgrade() -> None:
    job_status.create(op.get_bind(), checkfirst=True)
    op.create_table(
        "documents",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column("sha256", sa.String(64), nullable=False),
        sa.Column("collection", sa.String(128), nullable=False),
        sa.Column("doc_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("tenant", sa.String(64), nullable=False),
        sa.Column("source_uri", sa.Text, nullable=False),
        sa.Column("filename", sa.Text, nullable=False),
        sa.Column("mime", sa.String(128)),
        sa.Column("file_type", sa.String(32)),
        sa.Column("size_bytes", sa.BigInteger, nullable=False),
        sa.Column("blob_key", sa.Text, nullable=False),
        sa.Column("page_count", sa.Integer),
        sa.Column("ocr_pages", sa.Integer),
        sa.Column("chunk_count", sa.Integer),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column("indexed_at", sa.DateTime(timezone=True)),
        sa.UniqueConstraint("collection", "sha256", name="uq_documents_collection_sha256"),
    )
    op.create_index("ix_documents_source_uri", "documents", ["source_uri"])
    op.create_index("ix_documents_doc_id", "documents", ["doc_id"])
    op.create_index("ix_documents_tenant_created", "documents", ["tenant", "created_at"])

    op.create_table(
        "jobs",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column("document_id", postgresql.UUID(as_uuid=True),
                  sa.ForeignKey("documents.id", ondelete="CASCADE"), nullable=False),
        sa.Column("status", job_status, nullable=False),
        sa.Column("queue", sa.String(32)),
        sa.Column("attempts", sa.Integer, nullable=False),
        sa.Column("error", sa.Text),
        sa.Column("stage_ms", postgresql.JSONB),
        sa.Column("queued_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column("started_at", sa.DateTime(timezone=True)),
        sa.Column("finished_at", sa.DateTime(timezone=True)),
    )
    op.create_index("ix_jobs_document", "jobs", ["document_id"])
    op.create_index("ix_jobs_status_queued_at", "jobs", ["status", "queued_at"])
    op.create_index("ix_jobs_dead", "jobs", ["finished_at"], postgresql_where=sa.text("status = 'dead'"))


def downgrade() -> None:
    op.drop_table("jobs")
    op.drop_table("documents")
    job_status.drop(op.get_bind(), checkfirst=True)
