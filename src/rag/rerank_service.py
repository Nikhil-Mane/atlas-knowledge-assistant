"""Standalone rerank service, meant for GPU nodes (set RERANK_URL on the API).

  uvicorn rag.rerank_service:app --host 0.0.0.0 --port 8001

POST /rerank {"query": "...", "texts": ["...", ...]} -> {"scores": [...]}
"""
from __future__ import annotations

import asyncio

from fastapi import FastAPI
from pydantic import BaseModel, Field

from rag.config import get_settings
from rag.retrieval.rerank import score

app = FastAPI(title="Rerank service")
_slots = asyncio.Semaphore(max(1, get_settings().rerank_concurrency))


class RerankRequest(BaseModel):
    query: str = Field(max_length=4000)
    texts: list[str] = Field(max_length=100)


@app.post("/rerank")
async def rerank(req: RerankRequest) -> dict:
    async with _slots:
        scores = await asyncio.to_thread(score, req.query, req.texts, get_settings().rerank_model)
    return {"scores": scores}


@app.get("/health")
def health() -> dict:
    return {"ok": True}
