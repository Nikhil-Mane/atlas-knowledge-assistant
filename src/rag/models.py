"""Factories for every model the pipeline calls, so providers stay swappable.

LLM_PROVIDER=azure uses Azure OpenAI deployments; LLM_PROVIDER=openai uses
api.openai.com. Every paid embedding call goes through `GuardedEmbeddings`
(disk cache + per-run cap).
"""
from __future__ import annotations

from functools import lru_cache
from urllib.parse import urlparse

from langchain_core.embeddings import Embeddings

from rag.config import CACHE_DIR, Settings, get_settings
from rag.cost import GuardedEmbeddings


class MissingConfig(RuntimeError):
    pass


def _need(value, env_name: str):
    raw = value.get_secret_value() if hasattr(value, "get_secret_value") else value
    if not raw:
        raise MissingConfig(f"{env_name} is not set in .env")
    return raw


def azure_base_endpoint(endpoint: str) -> str:
    """Reduce any Azure OpenAI URL to `https://<resource>.openai.azure.com/`.

    The portal also shows the v1 base URL (`.../openai/v1`) and full request
    URLs. The Azure client appends `/openai/deployments/<name>/...` itself, so
    any extra path in the endpoint produces a 404 "Resource not found".
    """
    parsed = urlparse(endpoint.strip().strip('"').strip("'"))
    if not parsed.scheme or not parsed.netloc:
        raise MissingConfig(f"AZURE_OPENAI_ENDPOINT is not a URL: {endpoint!r}")
    return f"{parsed.scheme}://{parsed.netloc}/"


def _azure_common(s: Settings) -> dict:
    return {
        "azure_endpoint": azure_base_endpoint(_need(s.azure_openai_endpoint, "AZURE_OPENAI_ENDPOINT")),
        "api_key": _need(s.azure_openai_api_key, "AZURE_OPENAI_API_KEY"),
        "api_version": s.azure_openai_api_version,
    }


def model_id(s: Settings, kind: str) -> str:
    """Identifier used in cache keys and logs: provider + deployment/model name."""
    if s.llm_provider == "azure":
        name = {"embedding": s.azure_embedding_deployment, "vision": s.azure_vision_deployment,
                "chat": s.azure_chat_deployment, "fast": s.azure_fast_deployment}[kind]
    else:
        name = {"embedding": s.embedding_model, "vision": s.vision_model,
                "chat": s.chat_model, "fast": s.fast_model}[kind]
    return f"{s.llm_provider}:{name}"


def raw_dense_embeddings(settings: Settings | None = None) -> Embeddings:
    """The provider's embedding client with no cache or budget (used by the smoke check)."""
    s = settings or get_settings()
    if s.llm_provider == "fake":
        from rag.fakes import HashingEmbeddings
        return HashingEmbeddings(s.embedding_dim)
    if s.llm_provider == "azure":
        from langchain_openai import AzureOpenAIEmbeddings

        return AzureOpenAIEmbeddings(
            azure_deployment=_need(s.azure_embedding_deployment, "AZURE_EMBEDDING_DEPLOYMENT"),
            model=s.embedding_model,          # used only to pick the tokenizer
            dimensions=s.embedding_dim,
            chunk_size=s.embed_batch_size,
            max_retries=8,
            **_azure_common(s),
        )
    from langchain_openai import OpenAIEmbeddings

    return OpenAIEmbeddings(
        model=s.embedding_model,
        dimensions=s.embedding_dim,
        api_key=_need(s.openai_api_key, "OPENAI_API_KEY"),
        chunk_size=s.embed_batch_size,   # inputs per API request
        max_retries=8,                   # exponential backoff on 429s
    )


def dense_embeddings(settings: Settings | None = None) -> Embeddings:
    s = settings or get_settings()
    if s.llm_provider == "fake":
        from rag.fakes import HashingEmbeddings
        return HashingEmbeddings(s.embedding_dim)
    return GuardedEmbeddings(raw_dense_embeddings(s), model_id=model_id(s, "embedding"),
                             dim=s.embedding_dim, settings=s)


@lru_cache
def sparse_embeddings():
    """BM25 term weights, computed locally (no API call)."""
    from langchain_qdrant import FastEmbedSparse

    # Project-local cache: the default (system temp) gets wiped, forcing re-downloads.
    return FastEmbedSparse(model_name="Qdrant/bm25",
                           cache_dir=str(CACHE_DIR / "fastembed"))


def _chat(s: Settings, kind: str, timeout: int):
    if s.llm_provider == "fake":
        from rag.fakes import FakeChatModel
        return FakeChatModel()
    if s.llm_provider == "azure":
        from langchain_openai import AzureChatOpenAI

        deployment = {"vision": (s.azure_vision_deployment, "AZURE_VISION_DEPLOYMENT"),
                      "chat": (s.azure_chat_deployment, "AZURE_CHAT_DEPLOYMENT"),
                      "fast": (s.azure_fast_deployment, "AZURE_FAST_DEPLOYMENT")}[kind]
        return AzureChatOpenAI(azure_deployment=_need(*deployment), max_retries=6,
                               timeout=timeout, **_azure_common(s))
    from langchain_openai import ChatOpenAI

    model = {"vision": s.vision_model, "chat": s.chat_model, "fast": s.fast_model}[kind]
    return ChatOpenAI(model=model, api_key=_need(s.openai_api_key, "OPENAI_API_KEY"),
                      max_retries=6, timeout=timeout)


def vision_llm(settings: Settings | None = None):
    return _chat(settings or get_settings(), "vision", timeout=180)


def chat_llm(settings: Settings | None = None, *, fast: bool = False):
    return _chat(settings or get_settings(), "fast" if fast else "chat", timeout=120)
