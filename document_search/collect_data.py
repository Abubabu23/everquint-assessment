"""Collect a Wikipedia corpus (Tamil Nadu + AI/ML) for the document search assignment.

Each article is split into:
  - lead : the first paragraph(s) of the article -> used ONLY as the human reference summary for ROUGE
  - text : the rest of the article (cleaned)     -> the searchable corpus
Output: data/corpus.jsonl (one JSON object per line)
"""
import json
import os
import re
import time

import requests

API = "https://en.wikipedia.org/w/api.php"
HEADERS = {"User-Agent": "DocSearchAssignment/1.0 (student project; https://github.com/Abubabu23)"}

TOPICS = {
    "tamil_nadu": ["Category:Tamil Nadu"],
    "ai": ["Category:Artificial intelligence", "Category:Machine learning"],
}
PER_TOPIC = 1000        # target number of documents per topic
MAX_DEPTH = 3           # how many sub-category levels to follow
MAX_CATEGORIES = 400    # safety limit on category requests per topic
MIN_PAGE_BYTES = 5000   # skip tiny stub articles before downloading them
MIN_LEAD_CHARS = 200
MIN_TEXT_CHARS = 1000
MAX_TEXT_CHARS = 15000

STOP_SECTIONS = ("See also", "References", "External links", "Further reading",
                 "Notes", "Bibliography", "Sources", "Citations", "Footnotes")
SKIP_PREFIXES = ("List of", "Lists of", "Index of", "Outline of", "Timeline of")
NON_ARTICLE_PREFIXES = ("Template:", "Portal:", "Wikipedia:", "Draft:", "File:", "Help:",
                        "User:", "Talk:", "Module:", "Book:")


def api_get(params, tries=4):
    params = {**params, "format": "json", "formatversion": 2}
    for attempt in range(tries):
        try:
            r = requests.get(API, params=params, headers=HEADERS, timeout=30)
            r.raise_for_status()
            return r.json()
        except requests.RequestException:
            time.sleep(2 ** attempt)
    raise RuntimeError("Wikipedia API request failed repeatedly")


def category_members(category):
    pages, subcats, cont = [], [], {}
    while True:
        data = api_get({"action": "query", "list": "categorymembers", "cmtitle": category,
                        "cmlimit": 500, "cmtype": "page|subcat", "cmprop": "title|type|ids", **cont})
        for m in data["query"]["categorymembers"]:
            title, ns = m["title"], m.get("ns")      # some responses do not include "ns"
            if ns == 14 or title.startswith("Category:"):
                subcats.append(title)
            elif ns == 0 or (ns is None and not title.startswith(NON_ARTICLE_PREFIXES)):
                pages.append(title)
        if "continue" in data:
            cont = data["continue"]
        else:
            return pages, subcats


def collect_titles(roots, limit):
    titles, seen_titles, seen_cats = [], set(), set()
    queue = [(c, 0) for c in roots]
    while queue and len(titles) < limit and len(seen_cats) < MAX_CATEGORIES:
        cat, depth = queue.pop(0)
        if cat in seen_cats:
            continue
        seen_cats.add(cat)
        pages, subcats = category_members(cat)
        for t in pages:
            if t not in seen_titles:
                seen_titles.add(t)
                titles.append(t)
        if depth < MAX_DEPTH:
            queue.extend((s, depth + 1) for s in subcats)
        time.sleep(0.1)
    return titles[:limit]


def long_enough(titles):
    """Keep only titles whose page is big enough (prop=info returns the size for up to 50 titles at once)."""
    data = api_get({"action": "query", "prop": "info", "titles": "|".join(titles), "redirects": 1})
    return [p["title"] for p in data["query"]["pages"]
            if not p.get("missing") and p.get("length", 0) >= MIN_PAGE_BYTES]


def fetch_extract(title):
    """Plain-text article. Without 'exintro' the API returns only ONE extract per request,
    so full articles have to be fetched one by one."""
    data = api_get({"action": "query", "prop": "extracts", "explaintext": 1,
                    "titles": title, "redirects": 1})
    for p in data["query"]["pages"]:
        return p.get("extract", "")
    return ""


HEADING = re.compile(r"\n==+ (.+?) ==+\n")


def build_doc(title, extract):
    """Split into lead + cleaned body. Returns None if the article should be skipped."""
    if title.startswith(SKIP_PREFIXES) or "(disambiguation)" in title:
        return None
    m = HEADING.search(extract)
    if not m:
        return None
    lead, body = extract[:m.start()].strip(), extract[m.start():]
    if " may refer to" in lead[:300]:
        return None

    # cut the body at the first non-content section (References, External links, ...)
    for h in HEADING.finditer(body):
        if h.group(1).strip() in STOP_SECTIONS:
            body = body[:h.start()]
            break

    body = HEADING.sub(r"\n\1\n", body)          # keep heading text, drop the == markers
    body = re.sub(r"[ \t]+", " ", body)           # normalise spaces
    body = re.sub(r"\n{3,}", "\n\n", body).strip()
    body = body[:MAX_TEXT_CHARS]

    if len(lead) < MIN_LEAD_CHARS or len(body) < MIN_TEXT_CHARS:
        return None
    return {"title": title, "lead": lead, "text": body}


def main():
    os.makedirs("data", exist_ok=True)
    seen, total = set(), 0
    with open("data/corpus.jsonl", "w", encoding="utf-8") as f:
        for topic, roots in TOPICS.items():
            print(f"[{topic}] collecting titles ...")
            titles = collect_titles(roots, PER_TOPIC * 6)
            print(f"[{topic}] {len(titles)} candidate titles, downloading articles ...")
            kept = 0
            for i in range(0, len(titles), 50):
                for title in long_enough(titles[i:i + 50]):
                    if title in seen:
                        continue
                    try:
                        doc = build_doc(title, fetch_extract(title))
                    except RuntimeError:
                        continue
                    time.sleep(0.1)
                    if doc is None:
                        continue
                    seen.add(title)
                    f.write(json.dumps({"id": total, "topic": topic, **doc}, ensure_ascii=False) + "\n")
                    f.flush()
                    total += 1
                    kept += 1
                    if kept >= PER_TOPIC:
                        break
                print(f"[{topic}] checked {min(i + 50, len(titles))}/{len(titles)} titles, kept {kept}")
                if kept >= PER_TOPIC:
                    break

    with open("data/corpus.jsonl", encoding="utf-8") as f:
        docs = [json.loads(line) for line in f]
    print(f"\nSaved {len(docs)} documents to data/corpus.jsonl")
    for topic in TOPICS:
        n = [d for d in docs if d["topic"] == topic]
        if n:
            avg = sum(len(d["text"]) for d in n) // len(n)
            print(f"  {topic}: {len(n)} docs, average text length {avg} chars")


if __name__ == "__main__":
    main()