import json
import os
import re
import sys
import time

from agent import solve

EASY = [
    {"question": "If a train leaves at 14:30 and arrives at 18:05, how long is the journey?",
     "expected": ["3 hours 35", "3 hours and 35", "3h 35", "215 minutes"]},
    {"question": "Alice has 3 red apples and twice as many green apples as red. How many apples does she have in total?",
     "expected": ["9"]},
    {"question": "A shop sells pens at 12 rupees each. How much do 7 pens cost?",
     "expected": ["84"]},
    {"question": "A rectangle is 8 cm long and 5 cm wide. What is its area?",
     "expected": ["40"]},
    {"question": "Ravi had 50 rupees. He spent 18 rupees on a notebook and 12 rupees on a pen. How much money is left?",
     "expected": ["20"]},
    {"question": "A bus leaves at 09:15 and the journey takes 2 hours 50 minutes. At what time does it arrive?",
     "expected": ["12:05"]},
    {"question": "There are 24 students in a class and one third of them are absent. How many students are present?",
     "expected": ["16"]},
    {"question": "A car travels at 60 km per hour for 2.5 hours. How far does it travel?",
     "expected": ["150"]},
]

TRICKY = [
    {"question": "A meeting needs 60 minutes. There are free slots: 09:00-09:30, 09:45-10:30, 11:00-12:00. Which slots can fit the meeting?",
     "expected": ["11:00"], "forbidden": ["09:00", "09:45"]},
    {"question": "A meeting needs 45 minutes. There are free slots: 10:00-10:45, 11:00-11:30. Which slots can fit the meeting?",
     "expected": ["10:00"], "forbidden": ["11:00"]},
    {"question": "A night train leaves at 22:45 and arrives at 05:10 the next day. How long is the journey?",
     "expected": ["6 hours 25", "6 hours and 25", "6h 25", "385 minutes"]},
    {"question": "A jacket costs 2000 rupees. It is discounted by 20%, and then a further 10% is taken off the discounted price. What is the final price?",
     "expected": ["1440", "1,440"], "forbidden": ["1400", "1,400"]},
    {"question": "A bat and a ball cost 110 rupees in total. The bat costs 100 rupees more than the ball. How much does the ball cost?",
     "expected": ["5"], "forbidden": ["10"]},
]


def matches(answer: str, token: str) -> bool:
    """Whole-number/time match, so '5' does not match '50' or '110'."""
    pattern = r"(?<![\d.:])" + re.escape(token) + r"(?![\d:]|\.\d)"
    return re.search(pattern, answer, flags=re.IGNORECASE) is not None


def run_case(level: str, case: dict) -> dict:
    result = solve(case["question"])
    answer = str(result["answer"])
    has_expected = any(matches(answer, t) for t in case["expected"])
    has_forbidden = any(matches(answer, t) for t in case.get("forbidden", []))
    return {
        "level": level,
        "question": case["question"],
        "expected": case["expected"],
        "result": result,
        "verifier_passed": result["status"] == "success",
        "retried": result["metadata"]["retries"] > 0,
        "correct": result["status"] == "success" and has_expected and not has_forbidden,
    }


def main():
    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    cases = []
    if which in ("all", "easy"):
        cases += [("easy", c) for c in EASY]
    if which in ("all", "tricky"):
        cases += [("tricky", c) for c in TRICKY]

    records = []
    for i, (level, case) in enumerate(cases, 1):
        print(f"[{i}/{len(cases)}] ({level}) {case['question']}")
        rec = run_case(level, case)
        records.append(rec)
        print(f"    answer: {rec['result']['answer']}")
        print(f"    correct: {rec['correct']} | verifier passed: {rec['verifier_passed']} | "
              f"retries: {rec['result']['metadata']['retries']}")
        time.sleep(1)

    os.makedirs("logs", exist_ok=True)
    with open("logs/test_run.json", "w", encoding="utf-8") as f:
        json.dump(records, f, indent=2, ensure_ascii=False)
    with open("logs/test_run.txt", "w", encoding="utf-8") as f:
        for r in records:
            f.write(f"Level: {r['level']}\n")
            f.write(f"Question: {r['question']}\n")
            f.write(f"Correct: {r['correct']} | Verifier passed: {r['verifier_passed']} | Retried: {r['retried']}\n")
            f.write(json.dumps(r["result"], indent=2, ensure_ascii=False) + "\n")
            f.write("-" * 70 + "\n")

    total = len(records)
    print("\n===== SUMMARY =====")
    print(f"Correct answers : {sum(r['correct'] for r in records)}/{total}")
    print(f"Verifier passed : {sum(r['verifier_passed'] for r in records)}/{total}")
    print(f"Questions retried: {sum(r['retried'] for r in records)}/{total}")
    print("Logs saved in logs/test_run.json and logs/test_run.txt")


if __name__ == "__main__":
    main()