"""Cross-encoder reranking: locally on CPU, or on a remote service (GPU pool).

Hybrid search is fast but coarse; a cross-encoder reads the question and each
candidate together and scores relevance much more precisely. Reranking the
top ~20 candidates down to ~6 gives the answer step fewer, better sources.

At high concurrency the local model competes with the API for CPU (about
1-3 s per question on a laptop). Set RERANK_URL to send the work to
`rag.rerank_service` running on GPU nodes instead.
"""
from __future__ import annotations

import asyncio
from functools import lru_cache

from langchain_core.documents import Document

from rag.config import CACHE_DIR, Settings


@lru_cache
def _encoder(model_name: str):
    from fastembed.rerank.cross_encoder import TextCrossEncoder

    return TextCrossEncoder(model_name=model_name, cache_dir=str(CACHE_DIR / "fastembed"))


def score(query: str, texts: list[str], model_name: str) -> list[float]:
    return [float(x) for x in _encoder(model_name).rerank(query, texts, batch_size=16)]


def _order(docs: list[Document], scores: list[float], top_n: int) -> list[Document]:
    ranked = sorted(zip(docs, scores), key=lambda pair: pair[1], reverse=True)
    out = []
    for doc, sc in ranked[:top_n]:
        doc.metadata["rerank_score"] = float(sc)
        out.append(doc)
    return out


def rerank(query: str, docs: list[Document], *, top_n: int, model_name: str,
           max_candidates: int = 20, max_chars: int = 1000) -> list[Document]:
    """Keep the best `top_n` of the first `max_candidates` hybrid results (local model).

    Only the start of each chunk is scored: it carries the heading/file context
    and most of the signal, and cost grows with text length.
    """
    docs = docs[:max_candidates]
    if len(docs) <= 1:
        return docs[:top_n]
    return _order(docs, score(query, [d.page_content[:max_chars] for d in docs], model_name), top_n)


_client = None


def _http():
    global _client
    if _client is None:
        import httpx
        _client = httpx.AsyncClient(timeout=10)
    return _client


async def arerank(query: str, docs: list[Document], s: Settings) -> list[Document]:
    docs = docs[:s.rerank_candidates]
    if len(docs) <= 1:
        return docs[:s.rerank_top_n]
    texts = [d.page_content[:s.rerank_max_chars] for d in docs]
    if s.rerank_url:
        try:
            r = await _http().post(s.rerank_url.rstrip("/") + "/rerank",
                                   json={"query": query, "texts": texts})
            r.raise_for_status()
            return _order(docs, r.json()["scores"], s.rerank_top_n)
        except Exception:
            # The reranker improves ranking but isn't essential: fall back to
            # hybrid order rather than failing the question.
            return docs[:s.rerank_top_n]
    scores = await asyncio.to_thread(score, query, texts, s.rerank_model)
    return _order(docs, scores, s.rerank_top_n)
