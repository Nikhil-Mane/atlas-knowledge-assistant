"""Cost guard: spending caps, a shared rate limiter, and on-disk caches.

* Budgets count embedding tokens, OCR pages and chat calls actually sent to
  the API, and raise `BudgetExceeded` *before* a call that would pass a cap.
  - `RunBudget`: per process run (CLI, tests).       BUDGET_SCOPE=run
  - `RedisDailyBudget`: per calendar day (UTC), shared by every API process
    and worker through Redis (production).             BUDGET_SCOPE=daily
* `RedisRateLimiter`: tokens/minute and requests/minute windows shared by all
  workers, so bursts wait for the next minute instead of causing 429 storms.
* `SqliteCache` stores API results keyed by a hash of (deployment, input), so
  re-running tests, re-ingesting, or rebuilding a Qdrant collection costs
  nothing for content seen before.
"""
from __future__ import annotations

import hashlib
import sqlite3
import threading
import time
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path

import numpy as np
from langchain_core.embeddings import Embeddings

from rag.config import CACHE_DIR, Settings
METERS = ("embed_tokens", "ocr_pages", "chat_calls")


class BudgetExceeded(RuntimeError):
    pass


class _BudgetAPI:
    """Shared helpers; subclasses implement `charge`, `refund` and `used`."""
    scope_hint = ""
    embed_cache_hits = 0
    ocr_cache_hits = 0

    def charge_embed(self, tokens: int) -> None:
        self.charge("embed_tokens", tokens)

    def refund_embed(self, tokens: int) -> None:
        self.refund("embed_tokens", tokens)

    def charge_ocr(self, pages: int) -> None:
        self.charge("ocr_pages", pages)

    def refund_ocr(self, pages: int) -> None:
        self.refund("ocr_pages", pages)

    def charge_chat(self, calls: int = 1) -> None:
        self.charge("chat_calls", calls)

    @property
    def max_embed_tokens(self) -> int:
        return self.caps["embed_tokens"]

    @property
    def embed_tokens(self) -> int:
        return self.used("embed_tokens")

    @property
    def ocr_pages(self) -> int:
        return self.used("ocr_pages")

    @property
    def chat_calls(self) -> int:
        return self.used("chat_calls")

    def _exceeded(self, meter: str, used: int, n: int, cap: int) -> BudgetExceeded:
        label = {"embed_tokens": "embedding", "ocr_pages": "OCR", "chat_calls": "chat call"}[meter]
        return BudgetExceeded(f"{label} cap reached: {used:,} used + {n:,} needed > {cap:,} "
                              f"({self.scope_hint})")

    def summary(self, s: Settings) -> str:
        cost = (self.embed_tokens / 1e6 * s.embed_price_per_1m_tokens
                + self.ocr_pages * s.ocr_price_per_page)
        return (f"API usage this run: {self.embed_tokens:,} embedding tokens "
                f"({self.embed_cache_hits:,} texts from cache), {self.ocr_pages} OCR pages "
                f"({self.ocr_cache_hits} from cache), {self.chat_calls} chat calls "
                f"-> about ${cost:.4f} (embeddings + OCR)")


class RunBudget(_BudgetAPI):
    scope_hint = "raise MAX_*_PER_RUN in .env or the --max-* option"

    def __init__(self, max_embed_tokens: int, max_ocr_pages: int, max_chat_calls: int = 10**9):
        self.caps = {"embed_tokens": max_embed_tokens, "ocr_pages": max_ocr_pages,
                     "chat_calls": max_chat_calls}
        self._used = dict.fromkeys(METERS, 0)
        self._lock = threading.Lock()

    def used(self, meter: str) -> int:
        return self._used[meter]

    # Charges are reserved before a call so the cap can never be overshot, and
    # handed back if the call fails: a rejected request (404, 401, 429) is not billed.
    def charge(self, meter: str, n: int) -> None:
        with self._lock:
            if self._used[meter] + n > self.caps[meter]:
                raise self._exceeded(meter, self._used[meter], n, self.caps[meter])
            self._used[meter] += n

    def refund(self, meter: str, n: int) -> None:
        with self._lock:
            self._used[meter] -= n


_CHARGE_LUA = """
local used = tonumber(redis.call('GET', KEYS[1]) or '0')
if used + tonumber(ARGV[1]) > tonumber(ARGV[2]) then return -1 - used end
local n = redis.call('INCRBY', KEYS[1], ARGV[1])
redis.call('EXPIRE', KEYS[1], 172800)
return n
"""


class RedisDailyBudget(_BudgetAPI):
    """Daily caps shared by all processes. Check-and-charge is one atomic Lua call."""
    scope_hint = "daily cap shared by all workers; raise MAX_*_PER_DAY or wait until 00:00 UTC"

    def __init__(self, s: Settings, redis_client=None, prefix: str | None = None):
        import redis
        self.r = redis_client or redis.Redis.from_url(s.redis_url)
        # Per provider: fake-mode tests must never eat into the real daily caps.
        self.prefix = prefix or f"rag:budget:{s.llm_provider}"
        self.caps = {"embed_tokens": s.max_embed_tokens_per_day,
                     "ocr_pages": s.max_ocr_pages_per_day,
                     "chat_calls": s.max_chat_calls_per_day}
        self._charge = self.r.register_script(_CHARGE_LUA)

    def key(self, meter: str) -> str:
        return f"{self.prefix}:{datetime.now(timezone.utc):%Y-%m-%d}:{meter}"

    def used(self, meter: str) -> int:
        return int(self.r.get(self.key(meter)) or 0)

    def charge(self, meter: str, n: int) -> None:
        result = int(self._charge(keys=[self.key(meter)], args=[n, self.caps[meter]]))
        if result < 0:
            raise self._exceeded(meter, -1 - result, n, self.caps[meter])

    def refund(self, meter: str, n: int) -> None:
        self.r.decrby(self.key(meter), n)


_budget: RunBudget | None = None
_daily: dict[tuple, RedisDailyBudget] = {}


def get_budget(s: Settings) -> _BudgetAPI:
    """The budget for these settings' scope. Daily budgets are keyed by provider
    and Redis, so a per-run budget created earlier in the process (CLI, tests)
    can never stand in for the API's shared daily one, or the other way round."""
    global _budget
    if s.budget_scope == "daily":
        key = (s.llm_provider, s.redis_url, s.max_embed_tokens_per_day, s.max_ocr_pages_per_day,
               s.max_chat_calls_per_day)
        if key not in _daily:
            _daily[key] = RedisDailyBudget(s)
        return _daily[key]
    if _budget is None:
        _budget = RunBudget(s.max_embed_tokens_per_run, s.max_ocr_pages_per_run,
                            s.max_chat_calls_per_run)
    return _budget


def reset_budget(s: Settings) -> _BudgetAPI:
    """Start a fresh per-run budget (CLI runs, tests)."""
    global _budget
    _budget = None
    return get_budget(s)


_WINDOW_LUA = """
local cur = tonumber(redis.call('GET', KEYS[1]) or '0')
local amt = tonumber(ARGV[1])
if cur > 0 and cur + amt > tonumber(ARGV[2]) then return 0 end
redis.call('INCRBY', KEYS[1], amt)
redis.call('EXPIRE', KEYS[1], 120)
return 1
"""


class RedisRateLimiter:
    """Fixed one-minute windows shared by every process.

    `acquire` blocks until the amount fits in the current minute's window. A
    single request bigger than the whole limit is let through into an empty
    window, so it can never wait forever.
    """

    def __init__(self, s: Settings, redis_client=None, prefix: str = "rag:rl"):
        import redis
        self.r = redis_client or redis.Redis.from_url(s.redis_url)
        self.prefix = prefix
        self._take = self.r.register_script(_WINDOW_LUA)

    def try_take(self, name: str, amount: int, limit_per_minute: int) -> bool:
        """Non-blocking: take `amount` from this minute's window, or return False."""
        window = int(time.time() // 60)
        return bool(self._take(keys=[f"{self.prefix}:{name}:{window}"], args=[amount, limit_per_minute]))

    def acquire(self, name: str, amount: int, limit_per_minute: int,
                max_wait_s: float = 300) -> float:
        waited = 0.0
        while True:
            now = time.time()
            window = int(now // 60)
            if self._take(keys=[f"{self.prefix}:{name}:{window}"], args=[amount, limit_per_minute]):
                return waited
            pause = (window + 1) * 60 - now + 0.05
            if waited + pause > max_wait_s:
                raise TimeoutError(f"rate limit '{name}' still full after {waited:.0f}s")
            time.sleep(pause)
            waited += pause


_limiter: RedisRateLimiter | None = None


def get_rate_limiter(s: Settings) -> RedisRateLimiter | None:
    global _limiter
    if not s.rate_limit_enabled:
        return None
    if _limiter is None:
        _limiter = RedisRateLimiter(s)
    return _limiter


@lru_cache
def _encoding():
    import tiktoken
    return tiktoken.get_encoding("cl100k_base")   # tokenizer of text-embedding-3-*


def count_tokens(texts: list[str]) -> int:
    enc = _encoding()
    return sum(len(enc.encode(t, disallowed_special=())) for t in texts)


class SqliteCache:
    """Tiny thread-safe key/value store (one file per cache)."""

    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        # timeout: several worker processes may share one cache file on a volume.
        self._conn = sqlite3.connect(str(path), check_same_thread=False, timeout=30)
        self._conn.execute("pragma journal_mode=wal")
        self._conn.execute("create table if not exists kv (k text primary key, v blob not null)")
        self._conn.execute("create table if not exists owners (owner text not null, k text not null, "
                           "primary key (owner, k))")
        self._lock = threading.Lock()

    def get_many(self, keys: list[str]) -> dict[str, bytes]:
        out: dict[str, bytes] = {}
        with self._lock:
            for start in range(0, len(keys), 500):
                part = keys[start:start + 500]
                rows = self._conn.execute(
                    f"select k, v from kv where k in ({','.join('?' * len(part))})", part)
                out.update(rows)
        return out

    def add_owner(self, owner: str, keys: list[str]) -> None:
        with self._lock:
            self._conn.executemany("insert or ignore into owners values (?, ?)", [(owner, k) for k in keys])
            self._conn.commit()

    def delete_owned(self, owner: str) -> int:
        """Delete entries owned by `owner` that no other owner still uses. Returns the count."""
        with self._lock:
            keys = [r[0] for r in self._conn.execute("select k from owners where owner = ?", (owner,))]
            self._conn.execute("delete from owners where owner = ?", (owner,))
            orphans = [k for k in keys if not self._conn.execute(
                "select 1 from owners where k = ? limit 1", (k,)).fetchone()]
            self._conn.executemany("delete from kv where k = ?", [(k,) for k in orphans])
            self._conn.commit()
            return len(orphans)

    def put_many(self, items: dict[str, bytes]) -> None:
        with self._lock:
            self._conn.executemany("insert or replace into kv values (?, ?)", items.items())
            self._conn.commit()


def cache_key(*parts: str | bytes) -> str:
    h = hashlib.sha256()
    for p in parts:
        h.update(p if isinstance(p, bytes) else p.encode("utf-8"))
        h.update(b"\x1f")
    return h.hexdigest()


class GuardedEmbeddings(Embeddings):
    """Wraps a real embedding model with the disk cache and the run budget."""

    def __init__(self, inner: Embeddings, *, model_id: str, dim: int, settings: Settings,
                 cache: SqliteCache | None = None):
        self.inner, self.model_id, self.dim, self.settings = inner, model_id, dim, settings
        self.cache = cache or SqliteCache(CACHE_DIR / "embeddings.sqlite")

    def uncached_tokens(self, texts: list[str]) -> int:
        keys = [cache_key(self.model_id, str(self.dim), t) for t in texts]
        hits = self.cache.get_many(keys)
        return count_tokens([t for k, t in zip(keys, texts) if k not in hits])

    def precheck(self, texts: list[str]) -> None:
        """Raise before embedding a document that would not fit in the remaining budget."""
        needed = self.uncached_tokens(texts)
        budget = get_budget(self.settings)
        if budget.embed_tokens + needed > budget.max_embed_tokens:
            raise BudgetExceeded(
                f"document needs {needed:,} embedding tokens but only "
                f"{budget.max_embed_tokens - budget.embed_tokens:,} remain in this run's cap")

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        keys = [cache_key(self.model_id, str(self.dim), t) for t in texts]
        hits = self.cache.get_many(keys)
        budget = get_budget(self.settings)
        budget.embed_cache_hits += len(hits)

        misses = [(k, t) for k, t in zip(keys, texts) if k not in hits]
        if misses:
            tokens = count_tokens([t for _, t in misses])
            budget.charge_embed(tokens)
            limiter = get_rate_limiter(self.settings)
            if limiter:
                limiter.acquire("embed_tpm", tokens, self.settings.embed_tpm_limit)
            try:
                vectors = self.inner.embed_documents([t for _, t in misses])
            except Exception:
                budget.refund_embed(tokens)
                raise
            new = {k: np.asarray(v, dtype=np.float32).tobytes() for (k, _), v in zip(misses, vectors)}
            self.cache.put_many(new)
            hits.update(new)
        return [np.frombuffer(hits[k], dtype=np.float32).tolist() for k in keys]

    def embed_query(self, text: str) -> list[float]:
        return self.embed_documents([text])[0]
