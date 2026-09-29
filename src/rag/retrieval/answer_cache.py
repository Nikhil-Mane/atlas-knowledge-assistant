"""Answer cache for repeated questions (the cheapest LLM call is the one not made).

Only first-turn questions are cached: a follow-up depends on its conversation.
Keys include the tenant's *index version*, which ingestion and deletes bump,
so an answer is never served after the documents behind it changed.
"""
from __future__ import annotations

import hashlib
import json
import logging
import re

from rag.config import Settings

log = logging.getLogger(__name__)


def _version_key(s: Settings, tenant: str) -> str:
    return f"rag:idxver:{s.collection}:{tenant}"


def bump_index_version(s: Settings, tenant: str, redis_client=None) -> None:
    """Called after a tenant's documents change. Never fails the caller."""
    try:
        import redis
        r = redis_client or redis.Redis.from_url(s.redis_url, socket_timeout=2)
        r.incr(_version_key(s, tenant))
    except Exception as e:  # cache staleness is bounded by the TTL anyway
        log.warning("could not bump index version for %s: %s", tenant, e)


def normalize(question: str) -> str:
    return re.sub(r"\s+", " ", question.strip().lower()).rstrip(" ?!.")


class AnswerCache:
    def __init__(self, s: Settings, redis_client=None):
        self.s = s
        if redis_client is None:
            import redis.asyncio as aioredis
            redis_client = aioredis.Redis.from_url(s.redis_url, socket_timeout=2)
        self.r = redis_client

    async def _key(self, tenant: str, question: str, file_types: list[str] | None) -> str:
        version = int(await self.r.get(_version_key(self.s, tenant)) or 0)
        raw = json.dumps([self.s.collection, tenant, version, normalize(question),
                          sorted(file_types or [])])
        return "rag:ans:" + hashlib.sha256(raw.encode()).hexdigest()

    async def get(self, tenant: str, question: str, file_types: list[str] | None) -> dict | None:
        if not self.s.answer_cache_enabled:
            return None
        try:
            raw = await self.r.get(await self._key(tenant, question, file_types))
            return json.loads(raw) if raw else None
        except Exception as e:
            log.warning("answer cache unavailable: %s", e)
            return None

    async def put(self, tenant: str, question: str, file_types: list[str] | None, result: dict) -> None:
        # Only confident answers: never cache "not found" or answers that failed the check.
        if (not self.s.answer_cache_enabled or result.get("status") != "answered"
                or result.get("grounded") is False):
            return
        try:
            await self.r.set(await self._key(tenant, question, file_types),
                             json.dumps(result, default=str), ex=self.s.answer_cache_ttl_s)
        except Exception as e:
            log.warning("answer cache unavailable: %s", e)
