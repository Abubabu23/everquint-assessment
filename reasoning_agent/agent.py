import json

from llm import call_llm
from prompts import (
    PLANNER_PROMPT,
    EXECUTOR_PROMPT,
    VERIFIER_SOLVE_PROMPT,
    VERIFIER_CHECK_PROMPT,
    render,
)

MAX_RETRIES = 2


# ---------- helpers ----------
def parse_json(text: str) -> dict:
    """Extract the JSON object from the model output (handles ```json fences)."""
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end == -1:
        raise ValueError(f"No JSON found in model output: {text[:200]}")
    return json.loads(text[start:end + 1])


def ask_json(prompt: str) -> dict:
    """Call the LLM and parse JSON; try twice if the output is not valid JSON."""
    last_error = None
    for _ in range(2):
        try:
            return parse_json(call_llm(prompt))
        except ValueError as e:
            last_error = e
    raise ValueError(f"Model did not return valid JSON: {last_error}")


# ---------- phase 1: planner ----------
def make_plan(question: str) -> list:
    data = ask_json(render(PLANNER_PROMPT, question=question))
    plan = data.get("plan")
    if not isinstance(plan, list) or not plan:
        raise ValueError("Planner returned no plan")
    return plan


# ---------- phase 2: executor ----------
def execute(question: str, plan: list, feedback: str = "") -> dict:
    fb = f"Previous attempt was rejected: {feedback}\nFix this issue." if feedback else ""
    data = ask_json(render(
        EXECUTOR_PROMPT,
        question=question,
        plan=" | ".join(str(s) for s in plan),
        feedback=fb,
    ))
    for key in ("steps", "answer", "explanation"):
        if key not in data:
            raise ValueError(f"Executor output is missing '{key}'")
    return data


# ---------- phase 3: verifier ----------
def verify(question: str, solution: dict) -> dict:
    independent = ask_json(render(VERIFIER_SOLVE_PROMPT, question=question)).get("answer", "")
    data = ask_json(render(
        VERIFIER_CHECK_PROMPT,
        question=question,
        steps=json.dumps(solution["steps"]),
        answer=solution["answer"],
        independent_answer=independent,
    ))
    checks = data.get("checks", [])
    approved = bool(data.get("approved")) and bool(checks) and all(c.get("passed") for c in checks)
    return {"approved": approved, "checks": checks, "issues": data.get("issues", "")}


# ---------- agent loop ----------
def solve(question: str) -> dict:
    plan, checks, retries, feedback = [], [], 0, ""
    try:
        for attempt in range(MAX_RETRIES + 1):
            retries = attempt
            plan = make_plan(question)
            solution = execute(question, plan, feedback)
            result = verify(question, solution)
            checks = result["checks"]

            if result["approved"]:
                return {
                    "answer": solution["answer"],
                    "status": "success",
                    "reasoning_visible_to_user": solution["explanation"],
                    "metadata": {
                        "plan": " -> ".join(str(s) for s in plan),
                        "checks": checks,
                        "retries": retries,
                    },
                }
            feedback = result["issues"]
        reason = f"Verification failed after {MAX_RETRIES} retries: {feedback}"
    except Exception as e:
        reason = f"Agent error: {e}"

    return {
        "answer": "Could not produce a verified answer.",
        "status": "failed",
        "reasoning_visible_to_user": reason,
        "metadata": {
            "plan": " -> ".join(str(s) for s in plan),
            "checks": checks,
            "retries": retries,
        },
    }


# ---------- CLI ----------
if __name__ == "__main__":
    print("Reasoning agent. Type a question (or 'exit').")
    while True:
        q = input("\nQuestion> ").strip()
        if q.lower() in ("exit", "quit"):
            break
        if q:
            print(json.dumps(solve(q), indent=2, ensure_ascii=False))