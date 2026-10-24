import csv
import json
import os
import random

with open(os.path.join("results", "summaries.json"), encoding="utf-8") as f:
    rows = json.load(f)[:10]

rng = random.Random(7)
key, out = [], []
for r in rows:
    pair = [("baseline", r["baseline"]), ("llm", r["llm"])]
    rng.shuffle(pair)  # blind rating-ku A/B shuffle
    key.append({"title": r["title"], "A": pair[0][0], "B": pair[1][0]})
    out.append([r["title"], pair[0][1], pair[1][1], "", ""])

with open(os.path.join("results", "human_eval.csv"), "w", newline="", encoding="utf-8-sig") as f:
    w = csv.writer(f)
    w.writerow(["title", "summary_A", "summary_B", "rating_A", "rating_B"])
    w.writerows(out)

with open(os.path.join("results", "human_eval_key.json"), "w", encoding="utf-8") as f:
    json.dump(key, f, indent=2)

print("Created results/human_eval.csv (10 rows)")