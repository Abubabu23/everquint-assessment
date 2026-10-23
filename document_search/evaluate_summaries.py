"""Summary evaluation: ROUGE against human-written summaries + a sheet for human evaluation.

For every test document the LLM summarizes the article body. The Wikipedia lead paragraph (which was removed
from the searchable text) is the human-written reference summary. The LLM never sees it.
Baseline (no LLM): the first N words of the article body, with the same length as the LLM target.
Metrics: ROUGE-1, ROUGE-2, ROUGE-L (F1).
"""
import csv
import json
import os
import random
import sys

from rouge_score import rouge_scorer

from summarize import summarize_text

CORPUS_PATH = "data/corpus.jsonl"
N_PER_TOPIC = 12
SEED = 7
MIN_WORDS, MAX_WORDS = 40, 250      # target summary length is the length of the reference, kept in this range
MAX_INPUT_CHARS = 6000              # how much of the article the LLM reads
HUMAN_EVAL_N = 10                   # rows in the human-evaluation sheet
METRICS = ("rouge1", "rouge2", "rougeL")


def load_docs():
    with open(CORPUS_PATH, encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def pick_test_docs(docs, n_per_topic):
    rng = random.Random(SEED)
    test = []
    for topic in sorted({d["topic"] for d in docs}):
        pool = [d for d in docs if d["topic"] == topic]
        test += rng.sample(pool, min(n_per_topic, len(pool)))
    return test


def mean(values):
    return sum(values) / len(values) if values else 0.0


def main():
    n_per_topic = int(sys.argv[1]) if len(sys.argv) > 1 else N_PER_TOPIC
    docs = load_docs()
    test = pick_test_docs(docs, n_per_topic)
    scorer = rouge_scorer.RougeScorer(list(METRICS), use_stemmer=True)

    rows = []
    for i, d in enumerate(test, 1):
        ref_words = len(d["lead"].split())
        target = max(MIN_WORDS, min(ref_words, MAX_WORDS))
        print(f"[{i}/{len(test)}] {d['title']} (target {target} words)")
        try:
            llm_summary = summarize_text(d["title"], d["text"][:MAX_INPUT_CHARS], target)
        except Exception as e:                                   # e.g. free quota used up: keep what we have
            print(f"  stopped early: {e}")
            break
        baseline = " ".join(d["text"].split()[:target])
        s_llm = scorer.score(d["lead"], llm_summary)
        s_base = scorer.score(d["lead"], baseline)
        rows.append({
            "id": d["id"], "title": d["title"], "topic": d["topic"],
            "reference": d["lead"], "llm_summary": llm_summary, "baseline_summary": baseline,
            "ref_words": ref_words, "target_words": target, "llm_words": len(llm_summary.split()),
            "llm": {m: round(s_llm[m].fmeasure, 4) for m in METRICS},
            "baseline": {m: round(s_base[m].fmeasure, 4) for m in METRICS},
        })

    if not rows:
        print("No summaries were generated.")
        return

    def table(subset, label):
        print(f"\n===== SUMMARY EVALUATION: {label} ({len(subset)} docs) =====")
        print(f"{'system':<24} {'ROUGE-1':>8} {'ROUGE-2':>8} {'ROUGE-L':>8} {'avg words':>10}")
        for name, key, wkey in (("LLM summary", "llm", "llm_words"),
                                ("Baseline (first N words)", "baseline", "target_words")):
            print(f"{name:<24} "
                  + " ".join(f"{mean([r[key][m] for r in subset]):>8.3f}" for m in METRICS)
                  + f" {mean([r[wkey] for r in subset]):>10.1f}")
        print(f"(reference summaries average {mean([r['ref_words'] for r in subset]):.1f} words)")

    table(rows, "all topics")
    for topic in sorted({r["topic"] for r in rows}):
        table([r for r in rows if r["topic"] == topic], topic)

    os.makedirs("results", exist_ok=True)
    summary = {k: {m: round(mean([r[k][m] for r in rows]), 4) for m in METRICS} for k in ("llm", "baseline")}
    with open("results/summary_eval.json", "w", encoding="utf-8") as f:
        json.dump({"n_docs": len(rows), "mean_f1": summary, "per_document": rows}, f, indent=2, ensure_ascii=False)

    with open("results/summary_examples.txt", "w", encoding="utf-8") as f:
        for r in rows[:5]:
            f.write(f"TITLE: {r['title']} ({r['topic']})\n\nLLM SUMMARY ({r['llm_words']} words):\n{r['llm_summary']}\n\n"
                    f"REFERENCE (Wikipedia lead, {r['ref_words']} words):\n{r['reference']}\n\n"
                    f"ROUGE-1/2/L F1: {r['llm']['rouge1']} / {r['llm']['rouge2']} / {r['llm']['rougeL']}\n"
                    + "-" * 70 + "\n")

    # human evaluation sheet: fill the three score columns (1 = poor, 5 = excellent)
    step = max(1, len(rows) // HUMAN_EVAL_N)
    with open("results/human_eval.csv", "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["id", "title", "llm_summary", "reference_lead",
                    "coherence_1to5", "faithfulness_1to5", "coverage_1to5", "comments"])
        for r in rows[::step][:HUMAN_EVAL_N]:
            w.writerow([r["id"], r["title"], r["llm_summary"], r["reference"], "", "", "", ""])

    print("\nSaved results/summary_eval.json, summary_examples.txt and human_eval.csv")


if __name__ == "__main__":
    main()