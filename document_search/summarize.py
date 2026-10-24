import json
import os
import re
import sys

from llm import generate

DOCS_PATH = os.path.join("data", "documents.json")
MAX_CHARS = 8000  # LLM-ku anuppura text limit


def load_docs():
    with open(DOCS_PATH, encoding="utf-8") as f:
        docs = json.load(f)
    for d in docs:
        d["text"] = clean_text(d["text"])
    return docs

def clean_text(text):
    
    lines = []
    for line in text.split("\n"):
        s = line.strip()
        if "displaystyle" in s or 0 < len(s) <= 2:
            continue
        lines.append(s)
    text = "\n".join(lines)
    text = re.sub(r"\n{3,}(?!\n*==)", " ", text)
    text = re.sub(r"[ \t]{2,}", " ", text)
    return text


def split_sentences(text):
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]


def baseline_summary(text, n=3):
    """Simple baseline: first n sentences (lead-n)."""
    return " ".join(split_sentences(text)[:n])


def llm_summary(text, sentences=3):
    prompt = (
        f"Summarize the following article in {sentences} clear sentences. "
        "Use only facts from the article. No bullet points, no preamble.\n\n"
        f"Article:\n{text[:MAX_CHARS]}"
    )
    return generate(prompt, temperature=0.2, max_tokens=2048)


def find_doc(docs, key):
    if key.isdigit():
        return docs[int(key)]
    for d in docs:
        if d["title"].lower() == key.lower():
            return d
    return None


if __name__ == "__main__":
    key = " ".join(sys.argv[1:]) or "Backpropagation"
    doc = find_doc(load_docs(), key)
    if doc is None:
        print("Doc kedaikala:", key)
        sys.exit(1)
    print("TITLE:", doc["title"])
    print("\nBASELINE:\n", baseline_summary(doc["text"]))
    print("\nLLM:\n", llm_summary(doc["text"]))