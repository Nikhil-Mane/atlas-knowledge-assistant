"""Alembic environment: runs migrations against the configured Postgres."""
from alembic import context
from sqlalchemy import create_engine, pool

from rag.storage.postgres import Base

target_metadata = Base.metadata


def include_object(obj, name, type_, reflected, compare_to):
    # Only manage our own tables. LangGraph owns the checkpoint* tables
    # (conversation memory); autogenerate must never propose dropping them.
    if type_ == "table" and reflected and compare_to is None:
        return False
    return True


def run_migrations_online() -> None:
    url = context.config.get_main_option("sqlalchemy.url")
    engine = create_engine(url, poolclass=pool.NullPool, connect_args={"connect_timeout": 5})
    with engine.connect() as connection:
        context.configure(connection=connection, target_metadata=target_metadata,
                          compare_type=True, include_object=include_object)
        with context.begin_transaction():
            context.run_migrations()


run_migrations_online()
