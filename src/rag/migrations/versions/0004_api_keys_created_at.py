"""api_keys.created_at NOT NULL (databases created by an early 0002 had it nullable)

Revision ID: 0004
Revises: 0003
"""
import sqlalchemy as sa
from alembic import op

revision = "0004"
down_revision = "0003"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute("update api_keys set created_at = now() where created_at is null")
    op.alter_column("api_keys", "created_at", existing_type=sa.DateTime(timezone=True), nullable=False)


def downgrade() -> None:
    op.alter_column("api_keys", "created_at", existing_type=sa.DateTime(timezone=True), nullable=True)
