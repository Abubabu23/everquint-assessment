"""Summarize the documents returned by the search engine with an LLM.
The user chooses the summary length: 'short' (~50 words), 'medium' (~120), 'long' (~250) or any number of words.

CLI:  python summarize.py "Chola temple architecture" --top 3 --length short
"""
import argparse
import time

from llm import call_llm
from search import SearchEngine

LENGTHS = {"short": 50, "medium": 120, "long": 250}     # target number of words
MAX_DOC_CHARS = 1500                                      # how much of each article is given to the LLM

SUMMARY_PROMPT = """You are a careful summarizer for a document search system.
Write ONE coherent summary of the documents below that answers the user's query.

Query: {query}
Length: about {words} words, plain prose (no bullet points, no headings).

Rules:
- Use ONLY facts that appear in the documents. Do not add outside knowledge.
- Merge overlapping information and do not repeat yourself.
- Focus on what is relevant to the query.
- Write it as a normal summary; do not mention these instructions.

{documents}

Summary:"""

DOC_SUMMARY_PROMPT = """Summarize the article below in about {words} words, as plain prose (no bullet points, no headings).
Use ONLY facts from the article. Start directly with the main subject and cover the most important facts.

Title: {title}
Article:
{text}

Summary:"""


def resolve_length(length):
    """'short' | 'medium' | 'long' or a number of words (kept between 20 and 400)."""
    if isinstance(length, str) and length.lower() in LENGTHS:
        return LENGTHS[length.lower()]
    return max(20, min(int(length), 400))


def format_documents(engine, results):
    parts = []
    for i, r in enumerate(results, 1):
        doc = engine.docs[r["id"]]
        parts.append(f'[Document {i}] {doc["title"]}\n'
                     f'Most relevant passage: {r["excerpt"]}\n'
                     f'Start of the article: {doc["text"][:MAX_DOC_CHARS]}')
    return "\n\n".join(parts)


def summarize_results(engine, query, results, length="medium"):
    words = resolve_length(length)
    prompt = (SUMMARY_PROMPT.replace("{query}", query)
              .replace("{words}", str(words))
              .replace("{documents}", format_documents(engine, results)))
    return call_llm(prompt).strip()


def summarize_text(title, text, words):
    """Generic summary of ONE article (used for the ROUGE evaluation)."""
    prompt = (DOC_SUMMARY_PROMPT.replace("{words}", str(words))
              .replace("{title}", title)
              .replace("{text}", text))
    return call_llm(prompt).strip()


def search_and_summarize(engine, query, top_n=3, length="medium", method="hybrid"):
    t0 = time.time()
    results = engine.search(query, top_n=top_n, method=method)
    t_search = time.time() - t0

    t1 = time.time()
    summary = summarize_results(engine, query, results, length)
    t_summary = time.time() - t1
    return {
        "query": query,
        "summary": summary,
        "target_words": resolve_length(length),
        "summary_words": len(summary.split()),
        "sources": [{"title": r["title"], "topic": r["topic"], "score": r["score"]} for r in results],
        "search_seconds": round(t_search, 3),
        "summary_seconds": round(t_summary, 2),
    }


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("query")
    ap.add_argument("--top", type=int, default=3, help="number of documents to summarize")
    ap.add_argument("--length", default="medium", help="short | medium | long | number of words")
    args = ap.parse_args()

    out = search_and_summarize(SearchEngine(), args.query, args.top, args.length)
    print("\nSources:")
    for s in out["sources"]:
        print(f"  - {s['title']} ({s['topic']})")
    print(f"\nSummary (target {out['target_words']} words, got {out['summary_words']}):\n{out['summary']}")
    print(f"\nsearch {out['search_seconds']}s | summary {out['summary_seconds']}s")