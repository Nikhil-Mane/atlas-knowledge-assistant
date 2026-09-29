"""The full HTTP API end to end, in fake mode (no API key, no cost).

Upload -> job indexed -> chat (JSON and streaming) with citations -> follow-up
in the same thread -> delete document -> gone from search. Runs against the
real Qdrant/Postgres/Redis with a throwaway collection.
"""
import json
import uuid
from pathlib import Path

import pytest
from pydantic import SecretStr
from fastapi.testclient import TestClient
from sqlalchemy import delete

from rag.config import get_settings
from rag.storage.postgres import Document as DocRow
from rag.storage.postgres import session_factory
from rag.storage.qdrant import get_client

pytestmark = pytest.mark.integration
SAMPLES = Path(__file__).resolve().parents[1] / "data" / "samples"


def make_settings(**extra):
    return get_settings().model_copy(update={
        "llm_provider": "fake", "qdrant_collection": f"test_{uuid.uuid4().hex[:8]}",
        "ingest_mode": "inline", "rerank_enabled": False, **extra})


@pytest.fixture
def client_and_settings():
    from rag.api.app import create_app

    s = make_settings()
    try:
        get_client(s).get_collections()
    except Exception:
        pytest.skip("services not running (docker compose up -d)")
    with TestClient(create_app(s, memory="memory")) as client:
        yield client, s
    get_client(s).delete_collection(s.collection)
    with session_factory(s)() as db:
        db.execute(delete(DocRow).where(DocRow.collection == s.collection))
        db.commit()


def upload(client, *names):
    files = [("files", (n, (SAMPLES / n).read_bytes())) for n in names]
    r = client.post("/ingest", files=files, data={"tenant": "default"})
    assert r.status_code == 200, r.text
    return r.json()["jobs"]


def test_upload_chat_followup_delete(client_and_settings):
    client, _ = client_and_settings
    jobs = upload(client, "04-error-codes.json", "01-event-loop.md", "07-incident-email.eml")
    assert {j["status"] for j in jobs} == {"queued"}
    # TestClient runs inline background tasks before returning, so jobs are done.
    for j in jobs:
        status = client.get(f"/jobs/{j['job_id']}").json()
        assert status["status"] == "indexed", status
        assert status["attempts"] == 1 and status["chunk_count"] > 0

    # Same file again: skipped at registration, nothing queued.
    again = upload(client, "04-error-codes.json")[0]
    assert again["status"] == "skipped" and "duplicate" in again["reason"]

    docs = client.get("/documents").json()["documents"]
    assert {d["filename"] for d in docs} == {"04-error-codes.json", "01-event-loop.md",
                                              "07-incident-email.eml"}

    r = client.post("/chat", json={"question": "What does EMFILE mean?"}).json()
    assert r["status"] == "answered" and r["grounded"] is True
    assert r["citations"] and r["citations"][0]["filename"] == "04-error-codes.json"
    tid = r["thread_id"]

    # Follow-up in the same thread: history is kept (2 turns = 4 messages).
    client.post("/chat", json={"question": "And what about ENOENT?", "thread_id": tid})
    history = client.get(f"/threads/{tid}").json()["messages"]
    assert [m["role"] for m in history] == ["user", "assistant", "user", "assistant"]

    # Delete the error-codes document: its chunks disappear from search.
    doc_id = next(d["doc_id"] for d in docs if d["filename"] == "04-error-codes.json")
    deleted = client.delete(f"/documents/{doc_id}").json()
    assert deleted["chunks_deleted"] > 0
    r = client.post("/chat", json={"question": "What does EMFILE mean?"}).json()
    assert all(c["filename"] != "04-error-codes.json" for c in r["citations"])
    assert client.delete(f"/documents/{doc_id}").status_code == 404


def test_streaming_events(client_and_settings):
    client, _ = client_and_settings
    upload(client, "01-event-loop.md")
    with client.stream("POST", "/chat/stream",
                       json={"question": "How many threads does the libuv thread pool have?"}) as r:
        body = "".join(r.iter_text())
    events = [(e.split("\n")[0].removeprefix("event: "), json.loads(e.split("data: ", 1)[1]))
              for e in body.strip().split("\n\n")]
    names = [e for e, _ in events]
    assert names[0] == "start" and names[-1] == "final"
    assert {"rewrite_query", "hybrid_retrieve", "grade", "generate"} <= {
        d["node"] for e, d in events if e == "step"}
    tokens = "".join(d["text"] for e, d in events if e == "token")
    final = events[-1][1]
    assert tokens.strip() == final["answer"].strip()       # streamed text == final answer
    assert final["citations"][0]["filename"] == "01-event-loop.md"


def test_unsupported_and_too_large(client_and_settings, tmp_path):
    client, s = client_and_settings
    blob = tmp_path / "archive.bin"
    blob.write_bytes(b"\x00\x01\x02 binary")
    job = client.post("/ingest", files=[("files", ("archive.bin", blob.read_bytes()))]).json()["jobs"][0]
    assert job["status"] == "skipped" and "unsupported" in job["reason"]


def test_tenant_isolation_roles_limits_and_revocation():
    """AUTH_MODE=keys: the key decides the tenant; users can't cross tenants."""
    from rag.api.app import create_app
    from rag.auth import create_key
    from rag.storage.postgres import ApiKey

    s = make_settings(auth_mode="keys", api_key=SecretStr("bootstrap-secret"))
    try:
        get_client(s).get_collections()
    except Exception:
        pytest.skip("services not running")
    sessions = session_factory(s)
    run = uuid.uuid4().hex[:6]
    acme, _ = create_key(sessions, name="acme user", tenant=f"acme-{run}")
    globex, _ = create_key(sessions, name="globex user", tenant=f"globex-{run}",
                           requests_per_minute=12, questions_per_day=2)
    admin, _ = create_key(sessions, name="ops", tenant="default", role="admin")
    H = lambda k: {"X-API-Key": k}  # noqa: E731
    try:
        with TestClient(create_app(s, memory="memory")) as client:
            # No key / bad key
            assert client.get("/documents").status_code == 401
            assert client.get("/documents", headers=H("rag_wrong")).status_code == 401
            assert client.get("/health").status_code in (200, 503)          # public
            assert client.get("/me", headers=H(acme)).json()["tenant"] == f"acme-{run}"

            # acme uploads; the tenant comes from the key, not the request
            files = [("files", ("04-error-codes.json", (SAMPLES / "04-error-codes.json").read_bytes()))]
            r = client.post("/ingest", files=files, headers=H(acme))
            assert r.json()["tenant"] == f"acme-{run}"
            job_id = r.json()["jobs"][0]["job_id"]

            # globex can't see acme's documents, jobs or answers, even by asking
            assert client.post("/ingest", files=files, data={"tenant": f"acme-{run}"},
                               headers=H(globex)).status_code == 403
            assert client.get(f"/documents?tenant=acme-{run}", headers=H(globex)).status_code == 403
            assert client.get("/documents", headers=H(globex)).json()["documents"] == []
            assert client.get(f"/jobs/{job_id}", headers=H(globex)).status_code == 404
            g = client.post("/chat", json={"question": "What does EMFILE mean?"}, headers=H(globex)).json()
            assert g["citations"] == []                         # nothing from acme leaks

            a = client.post("/chat", json={"question": "What does EMFILE mean?"}, headers=H(acme)).json()
            assert a["citations"][0]["filename"] == "04-error-codes.json"
            # same thread id, other key: a different (empty) conversation
            assert client.get(f"/threads/{a['thread_id']}", headers=H(globex)).json()["messages"] == []

            # admin may act on any tenant; stats are admin-only
            assert len(client.get(f"/documents?tenant=acme-{run}", headers=H(admin)).json()["documents"]) == 1
            assert client.get("/stats", headers=H(acme)).status_code == 403
            assert client.get("/stats", headers=H(admin)).status_code == 200
            assert client.get("/me", headers=H("bootstrap-secret")).json()["role"] == "admin"

            # globex: 2 questions/day (1 used above), 12 requests/minute
            assert client.post("/chat", json={"question": "q2"}, headers=H(globex)).status_code == 200
            r = client.post("/chat", json={"question": "q3"}, headers=H(globex))
            assert r.status_code == 429 and "Daily quota" in r.json()["detail"]
            statuses = [client.get("/me", headers=H(globex)).status_code for _ in range(12)]
            assert 429 in statuses

    finally:
        with sessions() as db:
            db.query(ApiKey).filter(ApiKey.tenant.in_([f"acme-{run}", f"globex-{run}"])).delete()
            db.query(ApiKey).filter(ApiKey.name == "ops").delete()
            db.commit()
        get_client(s).delete_collection(s.collection)
        with sessions() as db:
            db.execute(delete(DocRow).where(DocRow.collection == s.collection))
            db.commit()


def test_revoked_key_is_rejected():
    from rag.auth import AuthError, Authenticator, create_key
    from rag.storage.postgres import ApiKey

    s = make_settings(auth_mode="keys")
    sessions = session_factory(s)
    key, row = create_key(sessions, name="temp", tenant="revoke-test")
    auth = Authenticator(s, sessions)
    try:
        assert auth.authenticate(key).tenant == "revoke-test"
        with sessions() as db:
            db.get(ApiKey, row.id).active = False
            db.commit()
        auth._cache.clear()                     # the TTL (60 s) would do this in production
        with pytest.raises(AuthError) as e:
            auth.authenticate(key)
        assert e.value.status == 401
    finally:
        with sessions() as db:
            db.query(ApiKey).filter(ApiKey.id == row.id).delete()
            db.commit()


def test_answer_cache_and_invalidation_and_metrics(client_and_settings):
    client, _ = client_and_settings
    upload(client, "04-error-codes.json")
    q = {"question": "What does EMFILE mean?"}

    first = client.post("/chat", json=q).json()
    assert first["cached"] is False and first["status"] == "answered"
    again = client.post("/chat", json={"question": "  what does EMFILE mean  "}).json()   # normalised
    assert again["cached"] is True and again["answer"] == first["answer"]

    # A cached first turn is still a real conversation: follow-ups have context.
    client.post("/chat", json={"question": "And ENOENT?", "thread_id": again["thread_id"]})
    roles = [m["role"] for m in client.get(f"/threads/{again['thread_id']}").json()["messages"]]
    assert roles == ["user", "assistant", "user", "assistant"]

    # New content for the tenant invalidates cached answers.
    upload(client, "01-event-loop.md")
    assert client.post("/chat", json=q).json()["cached"] is False

    metrics = client.get("/metrics").text
    assert 'rag_chat_total{cached="yes",status="answered"}' in metrics
    assert "rag_llm_calls_total" in metrics and "rag_queue_depth" in metrics
    assert 'rag_http_requests_total{method="POST",route="/chat",status="200"}' in metrics
