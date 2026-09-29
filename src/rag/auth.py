"""API keys: who is calling, which tenant they may see, and how much they may do.

* Keys are random 32-byte tokens; only their SHA-256 is stored.
* The key fixes the tenant. A `user` key can never read or write another
  tenant's documents or conversations; an `admin` key may name any tenant.
* Per key: requests/minute (shared Redis window) and questions/day quota.
* Lookups are cached in-process for a minute, so a request normally costs no
  database round trip. Revoking a key takes effect within that minute.
"""
from __future__ import annotations

import hashlib
import secrets
import threading
import time
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone

from sqlalchemy import select

from rag.config import Settings
from rag.storage.postgres import ApiKey

KEY_PREFIX = "rag_"
CACHE_TTL_S = 60
TOUCH_EVERY_S = 300


class AuthError(Exception):
    def __init__(self, status: int, message: str):
        super().__init__(message)
        self.status = status


@dataclass(frozen=True)
class Principal:
    key_id: str
    name: str
    tenant: str
    role: str
    requests_per_minute: int
    questions_per_day: int

    @property
    def is_admin(self) -> bool:
        return self.role == "admin"

    def resolve_tenant(self, requested: str | None) -> str:
        """The tenant this request acts on. Users are pinned to their key's tenant."""
        if not requested or requested == self.tenant:
            return self.tenant
        if self.is_admin:
            return requested
        raise AuthError(403, "This API key cannot access another tenant")

    def thread_key(self, thread_id: str) -> str:
        # Conversations are namespaced by tenant and key: guessing another
        # caller's thread_id reveals nothing.
        return f"{self.tenant}/{self.key_id}/{thread_id}"


DEV_PRINCIPAL = Principal("dev", "local development", "default", "admin", 10**6, 10**9)


def hash_key(key: str) -> str:
    return hashlib.sha256(key.encode("utf-8")).hexdigest()


def create_key(sessions, *, name: str, tenant: str, role: str = "user",
               requests_per_minute: int = 60, questions_per_day: int = 500) -> tuple[str, ApiKey]:
    if role not in ("user", "admin"):
        raise ValueError("role must be 'user' or 'admin'")
    plain = KEY_PREFIX + secrets.token_urlsafe(32)
    row = ApiKey(name=name, key_prefix=plain[:12], key_hash=hash_key(plain), tenant=tenant,
                 role=role, requests_per_minute=requests_per_minute,
                 questions_per_day=questions_per_day, active=True)
    with sessions() as db:
        db.add(row)
        db.commit()
        db.refresh(row)
    return plain, row


class Authenticator:
    def __init__(self, settings: Settings, sessions, redis_client=None):
        self.s = settings
        self.sessions = sessions
        self._cache: dict[str, tuple[Principal | None, float]] = {}
        self._touched: dict[str, float] = {}
        self._lock = threading.Lock()
        self._redis = redis_client
        self._limiter = None

    @property
    def redis(self):
        if self._redis is None:
            import redis
            self._redis = redis.Redis.from_url(self.s.redis_url)
        return self._redis

    @property
    def limiter(self):
        if self._limiter is None:
            from rag.cost import RedisRateLimiter
            self._limiter = RedisRateLimiter(self.s, redis_client=self.redis, prefix="rag:rl:key")
        return self._limiter

    def _fail_key(self, ip: str) -> str:
        return f"rag:authfail:{ip}:{int(time.time() // 60)}"

    def authenticate(self, key: str | None, ip: str | None = None) -> Principal:
        if self.s.auth_mode == "none":
            return DEV_PRINCIPAL
        # Brute-force protection: too many bad keys from one address in a minute -> 429.
        if ip and int(self.redis.get(self._fail_key(ip)) or 0) >= self.s.auth_fail_limit_per_min:
            raise AuthError(429, "Too many failed authentication attempts; try again in a minute")
        try:
            return self._authenticate(key)
        except AuthError as e:
            if e.status == 401 and ip:
                k = self._fail_key(ip)
                if self.redis.incr(k) == 1:
                    self.redis.expire(k, 120)
            raise

    def _authenticate(self, key: str | None) -> Principal:
        if not key:
            raise AuthError(401, "Missing X-API-Key header")
        bootstrap = self.s.api_key.get_secret_value() if self.s.api_key else None
        if bootstrap and secrets.compare_digest(key, bootstrap):
            return Principal("bootstrap", "bootstrap admin", "default", "admin", 10**6, 10**9)

        digest = hash_key(key)
        now = time.monotonic()
        with self._lock:
            cached = self._cache.get(digest)
        if cached and cached[1] > now:
            principal = cached[0]
        else:
            with self.sessions() as db:
                row = db.scalar(select(ApiKey).where(ApiKey.key_hash == digest, ApiKey.active.is_(True)))
            principal = None if row is None else Principal(
                str(row.id), row.name, row.tenant, row.role, row.requests_per_minute,
                row.questions_per_day)
            with self._lock:
                self._cache[digest] = (principal, now + CACHE_TTL_S)
        if principal is None:
            raise AuthError(401, "Invalid or revoked API key")
        self._touch(principal, now)
        return principal

    def check_rate(self, p: Principal) -> None:
        if self.s.auth_mode == "none":
            return
        if not self.limiter.try_take(p.key_id, 1, p.requests_per_minute):
            raise AuthError(429, f"Rate limit: {p.requests_per_minute} requests/minute for this key")

    def charge_question(self, p: Principal) -> None:
        if self.s.auth_mode == "none":
            return
        key = f"rag:quota:{datetime.now(timezone.utc):%Y-%m-%d}:{p.key_id}"
        used = self.redis.incr(key)
        if used == 1:
            self.redis.expire(key, 172800)
        if used > p.questions_per_day:
            raise AuthError(429, f"Daily quota reached: {p.questions_per_day} questions for this key")

    def _touch(self, p: Principal, now: float) -> None:
        """Record last use, at most every few minutes per key (no write per request)."""
        if p.key_id in ("bootstrap", "dev") or self._touched.get(p.key_id, 0) > now:
            return
        self._touched[p.key_id] = now + TOUCH_EVERY_S
        with self.sessions() as db:
            row = db.get(ApiKey, uuid.UUID(p.key_id))
            if row is not None:
                row.last_used_at = datetime.now(timezone.utc)
                db.commit()
