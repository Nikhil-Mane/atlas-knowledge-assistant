"""Query graph behaviour in fake mode: answer path, retry path, memory, filters."""
import uuid
from pathlib import Path

import pytest
from langgraph.checkpoint.memory import InMemorySaver
from sqlalchemy import delete

from rag.config import get_settings
from rag.cost import reset_budget
from rag.fakes import FakeChatModel
from rag.ingestion.pipeline import Ingestor, build_deps
from rag.retrieval.graph import QueryDeps, build_query_graph, extract_citations, run_turn
from rag.storage.postgres import Document as DocRow

pytestmark = pytest.mark.integration
SAMPLES = Path(__file__).resolve().parents[1] / "data" / "samples"


@pytest.fixture(scope="module")
def store_and_settings():
    s = get_settings().model_copy(update={
        "llm_provider": "fake", "qdrant_collection": f"test_{uuid.uuid4().hex[:8]}",
        "rerank_enabled": False})
    try:
        deps = build_deps(settings=s)
    except Exception as e:
        pytest.skip(f"services not running: {e}")
    ing = Ingestor(deps)
    for name in ("04-error-codes.json", "01-event-loop.md", "07-incident-email.eml"):
        assert ing.ingest(SAMPLES / name, tenant="default")["status"] == "indexed"
    yield deps.store, s
    deps.qdrant.delete_collection(s.collection)
    with deps.sessions() as db:
        db.execute(delete(DocRow).where(DocRow.collection == s.collection))
        db.commit()


def graph(store, s, **fake):
    reset_budget(s)
    model = FakeChatModel(**fake)
    return build_query_graph(QueryDeps(settings=s, store=store, chat=model, fast=model),
                             checkpointer=InMemorySaver())


def cfg():
    return str(uuid.uuid4())


def test_answer_with_citation(store_and_settings):
    store, s = store_and_settings
    out = run_turn(graph(store, s), "What does EMFILE mean?", cfg())
    assert out["status"] == "answered" and out["grounded"] is True
    assert out["citations"][0]["filename"] == "04-error-codes.json"
    assert out["search_query"] == "What does EMFILE mean?"     # first turn: no rewrite call


def test_nothing_relevant_retries_then_no_answer(store_and_settings):
    store, s = store_and_settings
    out = run_turn(graph(store, s, relevant=False), "What does EMFILE mean?", cfg())
    assert out["status"] == "no_answer" and out["citations"] == []
    assert len(out["tried_queries"]) == s.max_query_retries + 1       # 1 search + 2 rewrites
    assert "couldn't find" in out["answer"]


def test_follow_up_keeps_history(store_and_settings):
    store, s = store_and_settings
    g, c = graph(store, s), cfg()
    run_turn(g, "What does EMFILE mean?", c)
    out = run_turn(g, "And how do I fix it?", c)
    assert len(out["messages"]) == 4 and out["status"] == "answered"


def test_file_type_filter(store_and_settings):
    store, s = store_and_settings
    out = run_turn(graph(store, s), "memory leak listeners", cfg(), file_types=["email"])
    assert out["citations"] and {c["filename"] for c in out["citations"]} == {"07-incident-email.eml"}


def test_chat_budget_blocks_calls(store_and_settings):
    from rag.cost import BudgetExceeded

    store, s = store_and_settings
    tight = s.model_copy(update={"max_chat_calls_per_run": 1})     # grade uses the only call
    g = graph(store, tight)
    with pytest.raises(BudgetExceeded, match="chat call cap"):
        run_turn(g, "What does EMFILE mean?", cfg())


def test_extract_citations_ignores_out_of_range():
    from langchain_core.documents import Document

    ctx = [Document(page_content="a", metadata={"filename": "a.md"}),
           Document(page_content="b", metadata={"filename": "b.md"})]
    cites = extract_citations("x [2] y [1][2] z [7]", ctx)
    assert [c["n"] for c in cites] == [2, 1] and [c["filename"] for c in cites] == ["b.md", "a.md"]


def test_null_lists_from_real_models_are_accepted():
    """Azure returned file_types=null in a live test and the turn crashed."""
    from rag.retrieval.graph import GradeResult, GroundedResult, RewriteResult

    assert RewriteResult(search_query="x", file_types=None).file_types == []
    assert GradeResult(relevant_ids=None).relevant_ids == []
    assert GroundedResult(grounded=True, unsupported=None).unsupported == []


def test_follow_up_with_null_lists_completes(store_and_settings):
    store, s = store_and_settings
    g, c = graph(store, s, null_lists=True), cfg()
    run_turn(g, "What does EMFILE mean?", c)
    out = run_turn(g, "And how do I fix it?", c)          # rewrite runs, returns file_types=None
    assert out["status"] == "answered"


def test_statement_mid_conversation_is_not_searched(store_and_settings):
    """'my name is nikhil' used to be merged with the previous question and searched."""
    from rag.retrieval.graph import SMALL_TALK_REPLY

    store, s = store_and_settings
    g, c = graph(store, s, chit_chat=True), cfg()
    run_turn(g, "What is our Kubernetes budget?", c)      # first turn: normal path
    out = run_turn(g, "my name is nikhil", c)
    assert out["status"] == "chat" and out["answer"] == SMALL_TALK_REPLY
    assert out["citations"] == [] and out["search_query"] == ""
    assert "nikhil" not in out["answer"].lower()          # not echoed back
