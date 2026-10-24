import json
import os
import random
import re
import time

from rouge_score import rouge_scorer

from summarize import baseline_summary, llm_summary, load_docs

N_DOCS = 20
SEED = 42
OUT_PATH = os.path.join("results", "summaries.json")


def split_intro(text):
    parts = re.split(r"\n\s*\n\s*\n==", text, maxsplit=1)
    if len(parts) < 2:
        return None
    intro = parts[0].strip()
    body = re.sub(r"^=+.*?=+\s*$", "", "==" + parts[1], flags=re.M).strip()
    if len(intro) < 300 or len(body) < 1500:
        return None
    return intro, body


def main():
    os.makedirs("results", exist_ok=True)
    docs = load_docs()
    random.Random(SEED).shuffle(docs)

    scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
    keys = ["rouge1", "rouge2", "rougeL"]
    totals = {"baseline": dict.fromkeys(keys, 0.0), "llm": dict.fromkeys(keys, 0.0)}
    rows = []

    for d in docs:
        if len(rows) >= N_DOCS:
            break
        parts = split_intro(d["text"])
        if parts is None:
            continue
        reference, body = parts
        try:
            llm = llm_summary(body)
        except Exception as e:
            print("skip", d["title"], e)
            continue
        base = baseline_summary(body)

        for name, pred in (("baseline", base), ("llm", llm)):
            sc = scorer.score(reference, pred)
            for k in keys:
                totals[name][k] += sc[k].fmeasure

        rows.append({"title": d["title"], "reference": reference, "baseline": base, "llm": llm})
        print(f"{len(rows)}/{N_DOCS} done: {d['title']}")
        time.sleep(4)  # rate limit-ku

    n = len(rows)
    print(f"\nDocs evaluated: {n}")
    for name in ("baseline", "llm"):
        r = {k: totals[name][k] / n for k in keys}
        print(f"{name:9s} ROUGE-1={r['rouge1']:.3f}  ROUGE-2={r['rouge2']:.3f}  ROUGE-L={r['rougeL']:.3f}")

    with open(OUT_PATH, "w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False, indent=2)
    print("Saved", OUT_PATH)


if __name__ == "__main__":
    main()