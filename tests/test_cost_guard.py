"""The cost guard must stop paid calls before they happen and never pay twice."""
import pytest
from langchain_core.embeddings import Embeddings

from rag.config import Settings
from rag.cost import BudgetExceeded, GuardedEmbeddings, SqliteCache, count_tokens, reset_budget


class CountingEmbeddings(Embeddings):
    """Stands in for the paid API and records every text it receives."""

    def __init__(self):
        self.sent: list[str] = []

    def embed_documents(self, texts):
        self.sent.extend(texts)
        return [[float(len(t)), 1.0, 0.0, 0.0] for t in texts]

    def embed_query(self, text):
        return self.embed_documents([text])[0]


@pytest.fixture
def guarded(tmp_path):
    def make(max_tokens=10_000):
        s = Settings(_env_file=None, max_embed_tokens_per_run=max_tokens)
        reset_budget(s)
        api = CountingEmbeddings()
        g = GuardedEmbeddings(api, model_id="azure:test", dim=4, settings=s,
                              cache=SqliteCache(tmp_path / "emb.sqlite"))
        return g, api
    return make


def test_second_embedding_of_same_text_is_free(guarded):
    g, api = guarded()
    first = g.embed_documents(["event loop", "streams"])
    second = g.embed_documents(["streams", "event loop", "buffers"])
    assert api.sent == ["event loop", "streams", "buffers"]      # each text sent once
    assert second[0] == first[1] and second[1] == first[0]


def test_cache_survives_a_new_process(guarded):
    g1, _ = guarded()
    g1.embed_documents(["cached text"])
    g2, api2 = guarded()                                          # fresh budget, same cache file
    g2.embed_documents(["cached text"])
    assert api2.sent == []


def test_cap_blocks_the_call_before_it_is_made(guarded):
    g, api = guarded(max_tokens=5)
    with pytest.raises(BudgetExceeded):
        g.embed_documents(["this sentence is definitely longer than five tokens"])
    assert api.sent == []


def test_precheck_rejects_whole_document_up_front(guarded):
    texts = ["chunk one about streams", "chunk two about buffers"]
    g, api = guarded(max_tokens=count_tokens(texts) - 1)
    with pytest.raises(BudgetExceeded, match="document needs"):
        g.precheck(texts)
    assert api.sent == []


class FailingEmbeddings(Embeddings):
    def embed_documents(self, texts):
        raise RuntimeError("404 Resource not found")

    def embed_query(self, text):
        return self.embed_documents([text])[0]


def test_failed_call_is_not_counted(tmp_path):
    s = Settings(_env_file=None)
    budget = reset_budget(s)
    g = GuardedEmbeddings(FailingEmbeddings(), model_id="azure:x", dim=4, settings=s,
                          cache=SqliteCache(tmp_path / "emb.sqlite"))
    with pytest.raises(RuntimeError):
        g.embed_documents(["some text that fails"])
    assert budget.embed_tokens == 0          # rejected requests aren't billed


@pytest.mark.parametrize("given", [
    "https://nikhil372.openai.azure.com/openai/v1",
    "https://nikhil372.openai.azure.com/openai/v1/",
    '"https://nikhil372.openai.azure.com/"',
    "https://nikhil372.openai.azure.com",
    "https://nikhil372.openai.azure.com/openai/deployments/emb/embeddings?api-version=2024-10-21",
])
def test_azure_endpoint_is_reduced_to_resource_root(given):
    from rag.models import azure_base_endpoint
    assert azure_base_endpoint(given) == "https://nikhil372.openai.azure.com/"
