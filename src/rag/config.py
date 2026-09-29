"""Settings loaded from environment variables and `.env`."""
from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parents[2]
# Model downloads and API-result caches. In containers point this at a volume.
CACHE_DIR = Path(os.environ.get("RAG_CACHE_DIR", PROJECT_ROOT / ".cache"))


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=PROJECT_ROOT / ".env", extra="ignore")

    # --- Capacity targets. Defaults are the plan's baseline; replace the
    # per-page timings with measured values once Phase 2 runs on real files.
    docs_per_day: int = 10_000
    peak_factor: float = 5.0
    avg_pages_per_doc: float = 15.0
    chunks_per_page: float = 3.0
    tokens_per_chunk: int = 400
    scanned_page_ratio: float = 0.10
    # Measured by scripts/load_test.py on a laptop CPU: Docling ~2.5 s/page,
    # PDF_PARSER=fast ~0.4 s/page. Server CPUs are faster; a GPU much faster.
    parse_seconds_per_page: float = 2.5
    ocr_seconds_per_page: float = 6.0     # wall-clock latency of one vision call
    worker_utilization: float = 0.75      # headroom so queues don't grow unbounded
    retention_days: int = 365

    # --- LLM provider: "openai" (api.openai.com), "azure" (Azure OpenAI), or
    # "fake" (local stand-ins: no key, no cost; for tests, demos and load tests).
    llm_provider: Literal["openai", "azure", "fake"] = "openai"

    # --- Azure OpenAI (used when LLM_PROVIDER=azure). Deployment names are the
    # names you gave the deployments in Azure AI Foundry, not the model names.
    azure_openai_endpoint: str | None = None
    azure_openai_api_key: SecretStr | None = None
    azure_openai_api_version: str = "2024-10-21"
    azure_embedding_deployment: str | None = None
    azure_chat_deployment: str | None = None
    azure_fast_deployment: str | None = None
    azure_vision_deployment: str | None = None

    # --- Cost guard: hard caps per run, plus local caches so the same text or
    # page image is never paid for twice (even after dropping a collection).
    max_embed_tokens_per_run: int = 200_000
    max_ocr_pages_per_run: int = 10
    max_chat_calls_per_run: int = 200
    # "run": caps above, per process run (CLI). "daily": caps below, shared by
    # every API process and worker through Redis (production).
    budget_scope: Literal["run", "daily"] = "run"
    max_embed_tokens_per_day: int = 250_000_000     # capacity plan needs ~180M
    max_ocr_pages_per_day: int = 20_000             # capacity plan needs ~15K
    max_chat_calls_per_day: int = 1_000_000       # ~4 calls per question
    # Shared rate limits (Redis), kept below the Azure deployment quota so bursts
    # from many workers queue up instead of producing 429 storms.
    rate_limit_enabled: bool = False
    embed_tpm_limit: int = 800_000
    vision_rpm_limit: int = 300
    chat_rpm_limit: int = 2_000           # chat + fast deployments together
    chat_tpm_limit: int = 2_000_000
    chat_rate_wait_s: float = 20          # wait this long for a slot, then answer 503
    ocr_enabled: bool = True
    embed_price_per_1m_tokens: float = 0.02    # set to your Azure price; used for estimates only
    ocr_price_per_page: float = 0.003          # rough; depends on vision model and image size

    # --- OpenAI (used when LLM_PROVIDER=openai)
    openai_api_key: SecretStr | None = None
    embedding_model: str = "text-embedding-3-small"
    embedding_dim: int = 1024
    embed_batch_size: int = 512
    openai_embed_tpm_limit: int = 1_000_000
    chat_model: str = "gpt-5"
    fast_model: str = "gpt-5-mini"
    vision_model: str = "gpt-5-mini"
    # "docling": layout model (headings, reading order, tables); ~2.5 s/page on CPU.
    # "fast": plain text per page via pdfium; ~7x faster, loses structure and tables.
    pdf_parser: Literal["docling", "fast"] = "docling"
    ocr_concurrency: int = 6              # parallel vision calls per worker (capacity report: 6 at peak)
    upsert_batch_size: int = 256          # chunks per embed + Qdrant upsert round trip

    # --- Retrieval and answering (query graph)
    retrieve_k: int = 40                  # hybrid candidates before reranking
    rerank_enabled: bool = True
    rerank_model: str = "Xenova/ms-marco-MiniLM-L-6-v2"   # local cross-encoder, ~80 MB
    rerank_top_n: int = 6
    # CPU cost of reranking grows with candidates x text length. 20 x 1000 chars
    # keeps it around 1-2 s on a laptop CPU; the best hits are almost always in the top 20.
    rerank_candidates: int = 20
    rerank_max_chars: int = 1000
    rerank_url: str | None = None         # remote rerank service (GPU pool); local model if unset
    rerank_concurrency: int = 4           # parallel local reranks per API process (CPU-bound)
    source_max_chars: int = 1500          # per source sent to the LLM: the main token cost per question
    grounding_sample_rate: float = 1.0    # fraction of answers checked; lower it to save calls at scale
    answer_cache_enabled: bool = True     # repeated first-turn questions answered from Redis
    answer_cache_ttl_s: int = 3600
    max_query_retries: int = 2            # rewrite-and-search-again attempts
    grounding_check: bool = True
    history_turns: int = 6                # past messages used to rewrite follow-ups

    # --- API and workers
    # "none": no authentication, tenant from the request (local development only).
    # "keys": every request needs an X-API-Key from the api_keys table (scripts/api_keys.py);
    # the key decides the tenant and role.
    auth_mode: Literal["none", "keys"] = "none"
    api_key: SecretStr | None = None      # optional bootstrap admin key (tenant "default")
    max_files_per_upload: int = 20
    checkpoint_pool_size: int = 20        # async Postgres connections for conversation memory
    io_threads: int = 64                  # threads for blocking calls (Qdrant, Redis) under async load
    # When conversation state is written: "exit" = once per question (fastest; a turn
    # interrupted by a crash is not saved), "async"/"sync" = after every graph step.
    checkpoint_durability: Literal["exit", "async", "sync"] = "exit"
    log_json: bool = False                # structured JSON logs (production)

    # --- PII and secrets (see rag/security/pii.py)
    # "regex": fast local patterns (emails, phones, cards, IDs, IPs, self-stated names).
    # "presidio": adds NER-based person/location detection (pip install -e .[pii]).
    pii_engine: Literal["regex", "presidio"] = "regex"
    pii_entities: str = "EMAIL,PHONE,CREDIT_CARD,IBAN,IP_ADDRESS,US_SSN,IN_PAN,IN_AADHAAR,PERSON"
    # Documents: off | tag (mark chunks, keep text) | redact (replace with <EMAIL> etc.) | block (reject file)
    pii_ingest_policy: Literal["off", "tag", "redact", "block"] = "tag"
    # Questions: off | storage (mask in saved conversations) | llm (also mask before search and the LLM)
    pii_query_policy: Literal["off", "storage", "llm"] = "storage"
    secret_redaction: bool = True         # API keys, private keys, passwords: removed from docs and answers
    output_redact_pii: bool = False       # also mask PII in answers (documents may legitimately contain it)

    # --- Guardrails (see rag/security/guardrails.py)
    guard_input_policy: Literal["off", "flag", "block"] = "block"    # prompt-injection / jailbreak questions
    guard_doc_injection: Literal["off", "tag", "drop"] = "tag"       # instructions hidden in documents
    blocked_topics: str = ""              # comma-separated regexes; matching questions are refused

    # --- Retention and platform security
    chat_retention_days: int = 30         # conversations older than this are purged (scripts/purge.py)
    ocr_cache_enabled: bool = True        # off for sensitive corpora: no transcribed text on disk
    max_pdf_pages: int = 2000             # larger PDFs are rejected at upload
    auth_fail_limit_per_min: int = 20     # failed API-key attempts per client IP before 429
    cors_origins: str = ""                # comma-separated allowed origins for browser apps
    hsts: bool = False                    # send Strict-Transport-Security (enable behind HTTPS)
    ingest_mode: Literal["queue", "inline"] = "queue"   # inline: no Celery, for local dev
    max_upload_mb: int = 100
    ingest_max_retries: int = 3

    # --- Qdrant
    qdrant_url: str = "http://127.0.0.1:6333"
    qdrant_api_key: SecretStr | None = None
    qdrant_collection: str = "docs"
    # Shard count is fixed at creation. 2 shards let a second node take half
    # the data later without re-indexing.
    qdrant_shards: int = 2
    qdrant_replicas: int = 1
    hnsw_m: int = 16
    hnsw_ef_construct: int = 128

    # --- Postgres
    postgres_dsn: str = "postgresql+psycopg://rag:rag@127.0.0.1:5432/rag"
    db_pool_size: int = 5
    db_max_overflow: int = 10

    # --- Redis (Celery broker)
    redis_url: str = "redis://127.0.0.1:6379/0"

    # --- Raw file storage
    blob_backend: Literal["local", "s3", "azure"] = "local"
    azure_storage_connection_string: SecretStr | None = None
    azure_blob_container: str = "rag-raw"
    blob_root: Path = PROJECT_ROOT / "data" / "blobs"
    s3_bucket: str | None = None
    s3_endpoint_url: str | None = None


    @property
    def collection(self) -> str:
        """Qdrant collection actually used. Fake vectors never share a collection with real ones."""
        name = self.qdrant_collection
        if self.llm_provider == "fake" and not name.endswith("_fake"):
            return f"{name}_fake"
        return name


@lru_cache
def get_settings() -> Settings:
    return Settings()
