"""Chat load test: many simultaneous users asking questions through the API.

Each virtual user asks a question, waits for the answer, and immediately asks
the next one (closed loop), so `--users` is the number of requests in flight.
A share of questions repeat (answer-cache hits); the rest are made unique.

Refuses to run against a paid provider unless --allow-paid: at scale the
LLM bill is the dominant cost. Run the API with LLM_PROVIDER=fake to measure
the platform itself (API, Qdrant, Postgres memory, Redis) at $0.

  .venv/Scripts/python scripts/chat_load_test.py --users 200 --duration 60
"""
import argparse
import asyncio
import random
import statistics
import sys
import time
from collections import Counter

import httpx

QUESTIONS = [
    "What does EMFILE mean?", "How many threads does the libuv thread pool have?",
    "Why use stream.pipeline instead of pipe?", "How do I get __dirname in an ES module?",
    "What caused the order-service memory leak?", "Which HTTP status for a duplicate unique value?",
    "When does Node.js 22 reach end of life?", "How often should API keys be rotated?",
]


def pct(values, p):
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, int(p / 100 * len(ordered)))]


async def user(client, deadline, repeat_ratio, rng, results):
    while time.perf_counter() < deadline:
        q = rng.choice(QUESTIONS)
        if rng.random() > repeat_ratio:
            q = f"{q} (variant {rng.randrange(10**9)})"      # defeats the answer cache
        t0 = time.perf_counter()
        try:
            r = await client.post("/chat", json={"question": q})
            ok = r.status_code == 200
            cached = ok and r.json().get("cached", False)
            results.append((time.perf_counter() - t0, r.status_code, cached))
        except httpx.HTTPError as e:
            results.append((time.perf_counter() - t0, type(e).__name__, False))


async def main_async(args) -> int:
    headers = {"X-API-Key": args.api_key} if args.api_key else {}
    limits = httpx.Limits(max_connections=args.users, max_keepalive_connections=args.users)
    async with httpx.AsyncClient(base_url=args.api_url, headers=headers, timeout=120, limits=limits) as client:
        health = (await client.get("/health")).json()
        if health.get("provider") != "fake" and not args.allow_paid:
            print(f"API provider is '{health.get('provider')}'. Every question costs about 4 LLM calls; "
                  "run with LLM_PROVIDER=fake, or pass --allow-paid if you mean it.")
            return 1
        results: list = []
        rng = random.Random(time.time_ns())      # fresh variants every run: no stale cache hits
        print(f"{args.users} concurrent users for {args.duration}s against {args.api_url} "
              f"({args.repeat_ratio:.0%} repeated questions)...")
        deadline = time.perf_counter() + args.duration
        t0 = time.perf_counter()
        await asyncio.gather(*(user(client, deadline, args.repeat_ratio,
                                    random.Random(rng.random()), results) for _ in range(args.users)))
        elapsed = time.perf_counter() - t0

    lat = [r[0] for r in results if r[1] == 200]
    codes = Counter(r[1] for r in results)
    cached = sum(1 for r in results if r[2])
    print(f"\nrequests {len(results)}   ok {len(lat)}   errors {dict((k, v) for k, v in codes.items() if k != 200)}")
    print(f"throughput {len(lat) / elapsed:.1f} answers/s   answer-cache hits {cached / max(1, len(lat)):.0%}")
    if lat:
        print(f"latency p50 {pct(lat, 50):.2f}s   p95 {pct(lat, 95):.2f}s   p99 {pct(lat, 99):.2f}s   "
              f"mean {statistics.mean(lat):.2f}s")
    return 0 if len(lat) == len(results) else 1


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--api-url", default="http://127.0.0.1:8000")
    ap.add_argument("--api-key")
    ap.add_argument("--users", type=int, default=100)
    ap.add_argument("--duration", type=int, default=60)
    ap.add_argument("--repeat-ratio", type=float, default=0.5)
    ap.add_argument("--allow-paid", action="store_true")
    return asyncio.run(main_async(ap.parse_args()))


if __name__ == "__main__":
    sys.exit(main())
