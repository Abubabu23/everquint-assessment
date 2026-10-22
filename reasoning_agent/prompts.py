"""All prompts for the reasoning agent live here """


def render(template: str, **values) -> str:
    """Fill {placeholders} without touching the JSON braces used in the examples."""
    for key, value in values.items():
        template = template.replace("{" + key + "}", str(value))
    return template


PLANNER_PROMPT = """You are the PLANNER of a reasoning agent.
Read the question and write a short numbered plan to solve it.
Do NOT solve the question and do NOT give the final answer.

Rules:
- 3 to 6 short steps.
- Cover: what to extract, what to compute, which unit/format to use, what to double-check.
- Output ONLY valid JSON (no markdown fences, no extra text):
{"plan": ["step 1", "step 2"]}

Example 1
Question: If a train leaves at 14:30 and arrives at 18:05, how long is the journey?
Output: {"plan": ["Extract departure 14:30 and arrival 18:05", "Convert both times to minutes since midnight", "Subtract departure from arrival", "Convert the difference to hours and minutes", "Check the result is positive and in 'X hours Y minutes' format"]}

Example 2
Question: Alice has 3 red apples and twice as many green apples as red. How many apples does she have in total?
Output: {"plan": ["Extract red apples = 3", "Compute green apples = 2 x red", "Add red and green for the total", "Check all counts are non-negative whole numbers"]}

Question: {question}
Output:"""


EXECUTOR_PROMPT = """You are the EXECUTOR of a reasoning agent.
Follow the plan step by step and solve the question. Do the arithmetic carefully and write each intermediate calculation.

Rules:
- "answer" is a short final answer with units.
- "explanation" is shown to the user: 1-2 sentences, no long reasoning.
- Output ONLY valid JSON (no markdown fences, no extra text):
{"steps": ["calculation 1", "calculation 2"], "answer": "short final answer", "explanation": "1-2 sentence summary"}

Example 1
Question: If a train leaves at 14:30 and arrives at 18:05, how long is the journey?
Plan: Convert times to minutes, subtract, convert back.
Output: {"steps": ["14:30 = 870 minutes", "18:05 = 1085 minutes", "1085 - 870 = 215 minutes", "215 minutes = 3 hours 35 minutes"], "answer": "3 hours 35 minutes", "explanation": "The journey from 14:30 to 18:05 takes 3 hours 35 minutes."}

Example 2
Question: A meeting needs 60 minutes. Free slots: 09:00-09:30, 09:45-10:30, 11:00-12:00. Which slots can fit the meeting?
Plan: Compute each slot length, keep slots with length >= 60.
Output: {"steps": ["09:00-09:30 = 30 minutes (too short)", "09:45-10:30 = 45 minutes (too short)", "11:00-12:00 = 60 minutes (fits exactly)"], "answer": "Only 11:00-12:00", "explanation": "Only the 11:00-12:00 slot is at least 60 minutes long."}

{feedback}
Question: {question}
Plan: {plan}
Output:"""


VERIFIER_SOLVE_PROMPT = """Solve the question independently and carefully.
Output ONLY valid JSON (no markdown fences, no extra text):
{"answer": "short final answer with units"}

Example
Question: A shop sells pens at 12 rupees each. How much do 7 pens cost?
Output: {"answer": "84 rupees"}

Question: {question}
Output:"""


VERIFIER_CHECK_PROMPT = """You are the VERIFIER of a reasoning agent.
You get a question, a proposed solution, and an independently computed answer.
Run these checks:
1. independent_match: does the independent answer agree with the proposed answer? (different formats are fine, e.g. "215 minutes" = "3 hours 35 minutes")
2. constraints: are the numbers valid for the question (non-negative counts, valid times, totals add up, boundary cases handled)?
3. consistency: do the steps support the final answer without contradictions?

Output ONLY valid JSON (no markdown fences, no extra text):
{"checks": [{"check_name": "independent_match", "passed": true, "details": "..."}, {"check_name": "constraints", "passed": true, "details": "..."}, {"check_name": "consistency", "passed": true, "details": "..."}], "approved": true, "issues": ""}
"approved" is true only if ALL checks passed. If not approved, "issues" explains the problem in 1-2 sentences.

Example 1 (approve)
Question: A shop sells pens at 12 rupees each. How much do 7 pens cost?
Proposed steps: ["7 x 12 = 84"]
Proposed answer: 84 rupees
Independent answer: 84 rupees
Output: {"checks": [{"check_name": "independent_match", "passed": true, "details": "Both answers are 84 rupees."}, {"check_name": "constraints", "passed": true, "details": "Cost is a positive number."}, {"check_name": "consistency", "passed": true, "details": "7 x 12 = 84 supports the answer."}], "approved": true, "issues": ""}

Example 2 (reject)
Question: Alice has 3 red apples and twice as many green apples as red. How many apples in total?
Proposed steps: ["green = 2 + 3 = 5", "total = 3 + 5 = 8"]
Proposed answer: 8
Independent answer: 9
Output: {"checks": [{"check_name": "independent_match", "passed": false, "details": "Proposed 8 but independent answer is 9."}, {"check_name": "constraints", "passed": true, "details": "Counts are non-negative whole numbers."}, {"check_name": "consistency", "passed": false, "details": "'Twice as many' means multiply by 2, but the steps added 2."}], "approved": false, "issues": "Green apples should be 2 x 3 = 6, so the total is 9, not 8."}

Question: {question}
Proposed steps: {steps}
Proposed answer: {answer}
Independent answer: {independent_answer}
Output:"""