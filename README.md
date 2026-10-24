# Everquint AI Engineer Assessment

Solutions to the 5 take-home tasks. Each task lives in its own folder with its own README (setup, run steps, notes).

| # | Folder | Task | Details |
|---|---|---|---|
| 1 | `max_profit/` | Maximum profit problem | [README](max_profit/README.md) |
| 2 | `water_tank/` | Water tank problem | [README](water_tank/README.md) |
| 3 | `cifar10/` | CIFAR-10 image classification | [README](cifar10/README.md) |
| 4 | `reasoning_agent/` | Reasoning agent | [README](reasoning_agent/README.md) |
| 5 | `document_search/` | Hybrid document search + summarization (Streamlit app) | [README](document_search/README.md), [REPORT](document_search/REPORT.md) |

## Task 5 highlights

- Hybrid search (BM25 + MiniLM embeddings + RRF) over a 1000-article AI/ML Wikipedia corpus: Hit@5 0.867, MRR 0.611 (BM25 alone: 0.767 / 0.510).
- Lead-3 vs LLM summaries evaluated with ROUGE and a blind human A/B test (4.60 vs 4.80 out of 5).

## Setup

Use a virtual environment and follow the README inside each task folder for its dependencies and run commands. API keys go in a local `.env` file (see `document_search/README.md`). `.env` is gitignored and never committed.