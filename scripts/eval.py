"""Evaluate retrieval and answers against a golden question set.

Metrics
  retrieval (cheap): hit@1, hit@k and MRR of the expected file after hybrid
                     search + rerank. Costs one query embedding per question.
  answers (--full):  answered rate, expected file cited, expected keywords in
                     the answer, grounded rate, latency. Up to 4 chat calls per
                     question (rewrite is skipped on first turns).

With a real provider nothing is sent until you add --yes; without it the
script prints the estimated number of calls and exits.

Usage:
  .venv/Scripts/python scripts/eval.py                         # estimate only
  .venv/Scripts/python scripts/eval.py --yes                   # retrieval metrics
  .venv/Scripts/python scripts/eval.py --full --yes            # + answer metrics
  LLM_PROVIDER=fake .venv/Scripts/python scripts/eval.py --full   # free harness check
"""
import argparse
import json
import statistics
import sys
import time
import uuid
from datetime import datetime
from pathlib import Path

from rag.config import PROJECT_ROOT, get_settings
from rag.cost import reset_budget

DEFAULT_SET = PROJECT_ROOT / "data" / "eval_set.json"


def retrieval_eval(items, s, k: int) -> list[dict]:
    from qdrant_client import models

    from rag.retrieval.rerank import rerank
    from rag.retrieval.service import query_deps

    store = query_deps(s).store
    rows = []
    for it in items:
        flt = models.Filter(must=[models.FieldCondition(
            key="metadata.tenant", match=models.MatchValue(value="default"))])
        docs = [d for d, _ in store.similarity_search_with_score(it["question"], k=s.retrieve_k, filter=flt)]
        if s.rerank_enabled:
            docs = rerank(it["question"], docs, top_n=k, model_name=s.rerank_model,
                          max_candidates=s.rerank_candidates, max_chars=s.rerank_max_chars)
        files = [d.metadata["filename"] for d in docs[:k]]
        rank = next((i + 1 for i, f in enumerate(files) if f.startswith(it["expected_file"])), None)
        rows.append({"id": it["id"], "rank": rank, "top": files[:3]})
    return rows


def answer_eval(items, s) -> list[dict]:
    from rag.retrieval.graph import run_turn
    from rag.retrieval.service import build_service_graph

    graph, pool = build_service_graph(s, memory="memory")
    rows = []
    try:
        for it in items:
            t0 = time.perf_counter()
            out = run_turn(graph, it["question"], str(uuid.uuid4()))
            answer = out["answer"].lower()
            cited = [c["filename"] for c in out.get("citations", [])]
            rows.append({
                "id": it["id"], "status": out["status"], "grounded": out.get("grounded"),
                "cited_expected": any(f.startswith(it["expected_file"]) for f in cited),
                "keywords_hit": all(kw.lower() in answer for kw in it["keywords"]),
                "latency_s": round(time.perf_counter() - t0, 2), "answer": out["answer"][:300],
            })
    finally:
        if pool is not None:
            pool.close()
    return rows


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", default=str(DEFAULT_SET))
    ap.add_argument("--k", type=int, default=6)
    ap.add_argument("--full", action="store_true", help="also run the answer graph")
    ap.add_argument("--yes", action="store_true", help="allow paid API calls")
    ap.add_argument("--only", help="comma-separated question ids")
    args = ap.parse_args()

    s = get_settings()
    items = json.loads(Path(args.set).read_text(encoding="utf-8"))
    if args.only:
        wanted = set(args.only.split(","))
        items = [it for it in items if it["id"] in wanted]

    if s.llm_provider != "fake" and not args.yes:
        calls = len(items) * (4 if args.full else 0)
        print(f"{len(items)} questions against '{s.collection}' ({s.llm_provider}).")
        print(f"Would send: {len(items)} query embeddings (~{len(items) * 20} tokens)"
              + (f" and up to {calls} chat calls" if args.full else "") + ".")
        print("Add --yes to run.")
        return 0

    s = s.model_copy(update={"max_chat_calls_per_run": len(items) * 5 + 5})
    reset_budget(s)
    report: dict = {"when": datetime.now().isoformat(timespec="seconds"), "provider": s.llm_provider,
                    "collection": s.collection, "k": args.k}

    ret = retrieval_eval(items, s, args.k)
    ranks = [r["rank"] for r in ret]
    report["retrieval"] = {
        "hit@1": sum(r == 1 for r in ranks) / len(ranks),
        f"hit@{args.k}": sum(r is not None for r in ranks) / len(ranks),
        "mrr": statistics.mean(1 / r if r else 0 for r in ranks),
        "rows": ret,
    }
    print(f"{'id':5} {'rank':>4}  top results")
    for r in ret:
        print(f"{r['id']:5} {r['rank'] or '-':>4}  {', '.join(r['top'])}")
    rr = report["retrieval"]
    print(f"\nretrieval: hit@1 {rr['hit@1']:.0%}   hit@{args.k} {rr[f'hit@{args.k}']:.0%}   MRR {rr['mrr']:.2f}")

    if args.full:
        ans = answer_eval(items, s)
        n = len(ans)
        report["answers"] = {
            "answered": sum(a["status"] == "answered" for a in ans) / n,
            "cited_expected": sum(a["cited_expected"] for a in ans) / n,
            "keywords_hit": sum(a["keywords_hit"] for a in ans) / n,
            "grounded": sum(a["grounded"] is True for a in ans) / n,
            "p50_latency_s": statistics.median(a["latency_s"] for a in ans),
            "rows": ans,
        }
        a = report["answers"]
        print(f"\n{'id':5} {'status':9} {'cited':5} {'kw':3} {'grnd':5} {'sec':>5}")
        for r in ans:
            print(f"{r['id']:5} {r['status']:9} {'yes' if r['cited_expected'] else 'no':5} "
                  f"{'yes' if r['keywords_hit'] else 'no':3} {str(r['grounded']):5} {r['latency_s']:>5}")
        print(f"\nanswers: answered {a['answered']:.0%}   expected file cited {a['cited_expected']:.0%}   "
              f"keywords {a['keywords_hit']:.0%}   grounded {a['grounded']:.0%}   "
              f"p50 {a['p50_latency_s']}s")
        if s.llm_provider == "fake":
            print("(fake mode: answers quote the top source, so keyword and OCR scores are not meaningful)")

    out = PROJECT_ROOT / "data" / "eval" / f"report-{datetime.now():%Y%m%d-%H%M%S}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    print(f"\nreport: {out.relative_to(PROJECT_ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
