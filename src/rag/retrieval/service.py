"""Assemble the query graph with real (or fake) models and Postgres-backed memory."""
from __future__ import annotations

import asyncio
import sys

from rag.config import Settings, get_settings
from rag.ingestion.indexer import get_vector_store
from rag.models import chat_llm, dense_embeddings
from rag.retrieval.graph import QueryDeps, build_query_graph
from rag.storage.qdrant import ensure_collection, get_client


def use_selector_loop_on_windows() -> None:
    """psycopg's async driver can't run on Windows' default Proactor event loop."""
    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())


def query_deps(settings: Settings | None = None) -> QueryDeps:
    s = settings or get_settings()
    client = get_client(s)
    ensure_collection(client, s)
    return QueryDeps(settings=s, store=get_vector_store(client, dense_embeddings(s), s),
                     chat=chat_llm(s), fast=chat_llm(s, fast=True))


async def build_service_graph_async(settings: Settings | None = None, *, memory: str = "postgres"):
    """For the API. `memory`: "postgres" (shared by all replicas) or "memory" (tests).
    Returns (graph, pool); close the pool on shutdown."""
    s = settings or get_settings()
    if memory != "postgres":
        from langgraph.checkpoint.memory import InMemorySaver
        return build_query_graph(query_deps(s), checkpointer=InMemorySaver()), None

    from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
    from psycopg.rows import dict_row
    from psycopg_pool import AsyncConnectionPool

    conninfo = s.postgres_dsn.replace("postgresql+psycopg://", "postgresql://")
    pool = AsyncConnectionPool(conninfo, max_size=s.checkpoint_pool_size, open=False,
                               kwargs={"autocommit": True, "prepare_threshold": 0,
                                       "row_factory": dict_row})
    await pool.open()
    saver = AsyncPostgresSaver(pool)
    await saver.setup()            # creates checkpoint tables once; safe to repeat
    return build_query_graph(query_deps(s), checkpointer=saver), pool


def build_service_graph(settings: Settings | None = None, *, memory: str = "memory"):
    """For scripts: in-memory conversation state. Returns (graph, None)."""
    from langgraph.checkpoint.memory import InMemorySaver

    s = settings or get_settings()
    return build_query_graph(query_deps(s), checkpointer=InMemorySaver()), None
