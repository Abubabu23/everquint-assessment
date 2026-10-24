import csv
import json
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

CSV_PATH = Path("results/human_eval.csv")
KEY_PATH = Path("results/human_eval_key.json")


def ask(label):
    while True:
        v = input(f"  {label} rating (1-5, q=quit): ").strip().lower()
        if v == "q":
            return None
        if v in {"1", "2", "3", "4", "5"}:
            return v
        print("  1 to 5 mattum podu")


def save(rows, fields):
    with open(CSV_PATH, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def averages(rows, key):
    scores = {"baseline": [], "llm": []}
    wins = {"baseline": 0, "llm": 0, "tie": 0}
    for r in rows:
        if not r["rating_A"] or not r["rating_B"]:
            continue
        k = key[r["title"]]
        a, b = float(r["rating_A"]), float(r["rating_B"])
        scores[k["A"]].append(a)
        scores[k["B"]].append(b)
        if a == b:
            wins["tie"] += 1
        elif a > b:
            wins[k["A"]] += 1
        else:
            wins[k["B"]] += 1
    print("\n=== Human eval results ===")
    for name, vals in scores.items():
        if vals:
            print(f"{name:9s} avg = {sum(vals)/len(vals):.2f} (n={len(vals)})")
    print("Wins:", wins)


def main():
    key = {k["title"]: k for k in json.loads(KEY_PATH.read_text(encoding="utf-8"))}
    with open(CSV_PATH, newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        fields = reader.fieldnames
        rows = list(reader)

    for i, r in enumerate(rows, 1):
        if r["rating_A"] and r["rating_B"]:
            continue
        print("\n" + "=" * 70)
        print(f"[{i}/{len(rows)}] {r['title']}\n")
        print("--- SUMMARY A ---")
        print(r["summary_A"])
        print("\n--- SUMMARY B ---")
        print(r["summary_B"])
        print()
        a = ask("A")
        if a is None:
            break
        b = ask("B")
        if b is None:
            break
        r["rating_A"], r["rating_B"] = a, b
        save(rows, fields)
        print("  saved")

    save(rows, fields)
    averages(rows, key)


if __name__ == "__main__":
    main()