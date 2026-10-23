"""Search evaluation.

1. Pick a random test subset of the corpus (50 docs per topic, fixed seed).
2. Ask the LLM to write one natural search query per test document (10 documents per LLM call).
3. Run every query with BM25, embeddings and hybrid search and check whether the source document is returned.
Metrics: Hit@1, Hit@5, Hit@10, MRR@10, average latency.  Results go to results/.
"""
import json
import os
import random
import sys
import time

from llm import call_llm
from search import SearchEngine

N_PER_TOPIC = 50
BATCH = 10
SEED = 42
WINDOW_CHARS = 700
TOP_K = 10
METHODS = ("bm25", "embedding", "hybrid")
QUERIES_PATH = "results/test_queries.json"

QUERY_PROMPT = """You are helping to evaluate a document search engine.
Below are {n} article excerpts. For EACH article write ONE natural search query (6 to 12 words) that a user might type to find that article.

Rules:
- Do NOT copy the article title into the query.
- Use specific details from the excerpt (names, places, concepts), not generic words.
- Output ONLY valid JSON (no markdown fences, no extra text):
{"queries": [{"id": <article id>, "query": "<search query>"}]}

Example
Article id 7: "The temple was built by a king in the 11th century and has a tall granite tower ..."
Output: {"queries": [{"id": 7, "query": "granite tower temple built by eleventh century king"}]}

Articles:
{articles}
"""


def parse_json(text):
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end == -1:
        raise ValueError("No JSON found in model output")
    return json.loads(text[start:end + 1])


def pick_test_docs(docs):
    rng = random.Random(SEED)
    test = []
    for topic in sorted({d["topic"] for d in docs}):
        pool = [d for d in docs if d["topic"] == topic]
        test += rng.sample(pool, min(N_PER_TOPIC, len(pool)))
    return test


def excerpt_for(doc, rng):
    """A random window from the first part of the article (what the user would have 'read')."""
    text = " ".join(doc["text"].split())
    start = rng.randint(0, max(0, min(len(text) - WINDOW_CHARS, 3000)))
    return text[start:start + WINDOW_CHARS]


def generate_queries(test_docs):
    rng = random.Random(SEED)
    queries = {}
    for i in range(0, len(test_docs), BATCH):
        batch = test_docs[i:i + BATCH]
        articles = "\n".join(f'Article id {d["id"]}: "{excerpt_for(d, rng)}"' for d in batch)
        prompt = QUERY_PROMPT.replace("{n}", str(len(batch))).replace("{articles}", articles)
        data = None
        for _ in range(2):
            try:
                data = parse_json(call_llm(prompt))
                break
            except ValueError:
                continue
        if data:
            for item in data.get("queries", []):
                queries[int(item["id"])] = str(item["query"]).strip()
        print(f"  generated queries for {min(i + BATCH, len(test_docs))}/{len(test_docs)} documents")
    return queries


def load_or_make_test_set(docs, regenerate=False):
    if os.path.exists(QUERIES_PATH) and not regenerate:
        with open(QUERIES_PATH, encoding="utf-8") as f:
            return json.load(f)
    test_docs = pick_test_docs(docs)
    print(f"Generating queries for {len(test_docs)} test documents ...")
    queries = generate_queries(test_docs)
    test, dropped = [], 0
    for d in test_docs:
        q = queries.get(d["id"])
        if not q or d["title"].lower() in q.lower():     # missing query, or the title leaked into the query
            dropped += 1
            continue
        test.append({"id": d["id"], "title": d["title"], "topic": d["topic"], "query": q})
    print(f"Kept {len(test)} test queries (dropped {dropped}: missing, or title copied into query)")
    os.makedirs("results", exist_ok=True)
    with open(QUERIES_PATH, "w", encoding="utf-8") as f:
        json.dump(test, f, indent=2, ensure_ascii=False)
    return test


def evaluate(engine, test):
    engine.search("warm up", top_n=1, method="embedding")   # load model + cache first, so timings exclude one-time loading
    report = {}
    for method in METHODS:
        ranks, times = [], []
        for t in test:
            t0 = time.time()
            res = engine.search(t["query"], top_n=TOP_K, method=method)
            times.append(time.time() - t0)
            ids = [r["id"] for r in res]
            ranks.append(ids.index(t["id"]) + 1 if t["id"] in ids else None)

        def hit(k, subset=None):
            pairs = [(r, t) for r, t in zip(ranks, test) if subset is None or t["topic"] == subset]
            return sum(1 for r, _ in pairs if r is not None and r <= k) / len(pairs)

        report[method] = {
            "hit@1": round(hit(1), 3),
            "hit@5": round(hit(5), 3),
            "hit@10": round(hit(10), 3),
            "mrr@10": round(sum(1 / r for r in ranks if r) / len(ranks), 3),
            "avg_ms": round(1000 * sum(times) / len(times), 1),
            "hit@5_by_topic": {tp: round(hit(5, tp), 3) for tp in sorted({t["topic"] for t in test})},
            "ranks": ranks,
        }
    return report


def main():
    regenerate = "--regen" in sys.argv
    engine = SearchEngine()
    test = load_or_make_test_set(engine.docs, regenerate)

    report = evaluate(engine, test)

    print(f"\n===== SEARCH EVALUATION ({len(test)} queries, corpus {len(engine.docs)} docs) =====")
    print(f"{'method':<10} {'Hit@1':>6} {'Hit@5':>6} {'Hit@10':>7} {'MRR@10':>7} {'avg ms':>7}")
    for m, r in report.items():
        print(f"{m:<10} {r['hit@1']:>6} {r['hit@5']:>6} {r['hit@10']:>7} {r['mrr@10']:>7} {r['avg_ms']:>7}")
    print("\nHit@5 by topic:")
    for m, r in report.items():
        print(f"  {m:<10} {r['hit@5_by_topic']}")

    # error analysis: queries the hybrid search did not return in the top 10
    misses = [{"query": t["query"], "expected": t["title"],
               "hybrid_top3": [r["title"] for r in engine.search(t["query"], top_n=3, method="hybrid")]}
              for t, rank in zip(test, report["hybrid"]["ranks"]) if rank is None]
    print(f"\nHybrid misses (not in top {TOP_K}): {len(misses)}")
    for m in misses[:5]:
        print(f"  Q: {m['query']}\n     expected: {m['expected']} | got: {m['hybrid_top3']}")

    os.makedirs("results", exist_ok=True)
    with open("results/search_eval.json", "w", encoding="utf-8") as f:
        json.dump({"n_queries": len(test), "metrics": report, "hybrid_misses": misses}, f,
                  indent=2, ensure_ascii=False)
    print("\nSaved results/search_eval.json and results/test_queries.json")


if __name__ == "__main__":
    main()