# Multi-Step Reasoning Agent with Self-Checking

A small agent that solves word problems (time, arithmetic, constraints) in three phases (Planner, Executor, Verifier) and returns a JSON result. Built with Python and the Gemini API.

## How it works
Question -> Planner -> Executor -> Verifier -> JSON answer

- **Planner**: writes a short numbered plan. It does not solve the question.
- **Executor**: follows the plan, writes the intermediate calculations and returns steps, a short answer and a 1-2 sentence explanation.
- **Verifier**: (1) solves the question independently, without seeing the executor's answer; (2) compares both answers; (3) checks constraints (valid times, non-negative counts, totals) and consistency of the steps.
- If any check fails, the agent retries (plan + execute again, with the verifier's feedback passed to the executor) up to 2 times. After that it returns `status: "failed"` with the reason.
- Long step-by-step reasoning is never shown to the user. Only `answer` and `reasoning_visible_to_user` are meant for the user; `plan`, `checks` and `retries` are for debugging.

## Output format
```json
{
  "answer": "84 rupees",
  "status": "success",
  "reasoning_visible_to_user": "7 pens at 12 rupees each cost 84 rupees.",
  "metadata": {
    "plan": "step 1 -> step 2 -> ...",
    "checks": [{"check_name": "independent_match", "passed": true, "details": "..."}],
    "retries": 0
  }
}
```

## Files
| File | Purpose |
|---|---|
| `agent.py` | Planner, executor, verifier functions, `solve(question)` agent loop and CLI |
| `prompts.py` | All prompts (nothing is hard-coded in `agent.py`) |
| `llm.py` | The only place that calls the LLM (easy to swap providers) |
| `tests.py` | 8 easy + 5 tricky questions, writes logs |
| `logs/` | Example run logs (`test_run.txt` readable, `test_run.json` full) |

## How to run
1. Install: `pip install -r requirements.txt`
2. Set the API key (PowerShell): `$env:GEMINI_API_KEY="your-key"` (key is never stored in the code)
3. CLI: `python agent.py` and type a question
4. From code: `from agent import solve; solve("A shop sells pens at 12 rupees each. How much do 7 pens cost?")`
5. Tests: `python tests.py` (or `python tests.py easy` / `python tests.py tricky`)

## Where prompts live
`prompts.py`: `PLANNER_PROMPT`, `EXECUTOR_PROMPT`, `VERIFIER_SOLVE_PROMPT`, `VERIFIER_CHECK_PROMPT`. Each prompt specifies the exact JSON output format and has 1-2 worked examples.

## Prompt design
- Planner and executor are separate so the plan can be inspected and the executor must follow it.
- Every prompt asks for JSON only, so the output can be parsed by code. `agent.py` strips markdown fences and retries once if the JSON is invalid.
- The verifier solves the question on its own, before it sees the executor's answer. This avoids simply agreeing with a wrong answer.
- The verifier prompt has one "approve" and one "reject" example (the "twice as many" mistake), so it knows what a rejection looks like.
- On a retry, the verifier's issue text is added to the executor prompt so the same mistake is not repeated.

## Reliability (llm.py)
- Temporary errors (503 high demand, per-minute 429) are retried with exponential backoff.
- If a model's daily free quota is over (or the model is not available), it switches to the next model in the list.
- Responses are cached in `.llm_cache.json` (git-ignored), so re-running tests does not spend API quota.

## Tests and results
13 questions: 8 easy (arithmetic, areas, time differences, fractions) and 5 tricky:
- slot fitting with several slots, and a slot that fits exactly at the boundary
- journey that crosses midnight (22:45 to 05:10)
- two stacked discounts (20% then 10% = 1440, not 1400)
- bat-and-ball puzzle (5 rupees, not 10)

Result: **13/13 correct, verifier passed 13/13, no question needed a retry.** Full question, final JSON, verifier status and retry flag for every test are in `logs/test_run.txt` and `logs/test_run.json`. Tests compare answers by number/time match (so "5" does not match "50"), not by exact string.

## Assumptions
- Questions are plain text word problems; times are 24-hour `HH:MM`.
- Answers are short text with units.
- A question counts as "success" only when the verifier approves all checks.

## What did not work well
- Gemini free tier allows only about 20 requests per day per model, and one question needs 4 LLM calls. A full test run exceeded the quota and every test failed with a 429 error. Fixed with model switching and the cache.
- Temporary 503 "high demand" errors crashed the agent loop. Fixed with retry and backoff.

## What I would improve with more time
- Let the executor use Python for arithmetic and add programmatic checks for time and money problems, instead of relying only on the LLM.
- Use the API's structured JSON output mode instead of parsing text.
- Add a test that forces a wrong solution, to exercise the retry path (no test triggered a retry so far).
- Add more tricky questions and run each several times to measure consistency.
- Reduce cost and latency (4 calls per question), for example by using a cheaper model for the verifier.