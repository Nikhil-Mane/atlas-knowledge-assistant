"""threads (retention, deletion) and audit_events

Revision ID: 0003
Revises: 0002
"""
import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision = "0003"
down_revision = "0002"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "threads",
        sa.Column("thread_key", sa.Text, primary_key=True),
        sa.Column("tenant", sa.String(64), nullable=False),
        sa.Column("key_id", sa.String(64), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column("last_active_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
    )
    op.create_index("ix_threads_last_active", "threads", ["last_active_at"])
    op.create_table(
        "audit_events",
        sa.Column("id", sa.BigInteger, primary_key=True, autoincrement=True),
        sa.Column("ts", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column("action", sa.String(64), nullable=False),
        sa.Column("tenant", sa.String(64)),
        sa.Column("key_id", sa.String(64)),
        sa.Column("target", sa.Text),
        sa.Column("detail", postgresql.JSONB),
        sa.Column("ip", sa.String(64)),
    )
    op.create_index("ix_audit_ts", "audit_events", ["ts"])
    op.create_index("ix_audit_tenant_ts", "audit_events", ["tenant", "ts"])


def downgrade() -> None:
    op.drop_table("audit_events")
    op.drop_table("threads")
