# Task 5: Document Search + Summarization

Hybrid search (BM25 + dense embeddings + RRF) over a 1000-article AI/ML Wikipedia corpus, with LLM and lead-3 summaries, evaluated with ROUGE and a blind human A/B test. A Streamlit app ties it together.

## Setup

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Create a `.env` file in this folder (never commit it):

```
LLM_PROVIDER=groq
GROQ_API_KEY=your_key_here
GROQ_MODEL=openai/gpt-oss-120b
# optional alternative provider
# LLM_PROVIDER=gemini
# GEMINI_API_KEY=your_key_here
# GEMINI_MODEL=gemini-3.8-flash
```

## Run

Run everything from this folder (paths are relative).

```powershell
python collect_data.py          # downloads Wikipedia articles -> data/documents.json
python search.py                # interactive CLI search (builds embedding cache on first run)
streamlit run app.py            # web UI at http://localhost:8501
python evaluate_search.py       # Hit@5 / MRR for bm25, dense, hybrid
python evaluate_summaries.py    # ROUGE: lead-3 baseline vs LLM -> results/summaries.json
python make_human_eval.py       # blind A/B sheet -> results/human_eval.csv (+ key)
python rate_human_eval.py       # rate in terminal, prints averages
```

Warning: `make_human_eval.py` reshuffles A/B and overwrites the key. Do not re-run it after rating.

## Files

| File | Purpose |
|---|---|
| `collect_data.py` | Wikipedia API crawler (categories + seed titles) |
| `search.py` | `HybridSearcher`: BM25, MiniLM chunk embeddings (cached), RRF fusion |
| `llm.py` | `generate()` wrapper, provider chosen via `.env` |
| `summarize.py` | `clean_text`, lead-3 baseline, LLM summary |
| `evaluate_search.py` | Retrieval evaluation on 30 queries |
| `evaluate_summaries.py` | ROUGE evaluation on 20 docs |
| `make_human_eval.py`, `rate_human_eval.py` | Blind human evaluation |
| `app.py` | Streamlit UI |
| `REPORT.md` | Full results and analysis |

## Design choices

- **Corpus:** AI/ML-only Wikipedia articles (1000 docs).
- **Hybrid retrieval:** BM25 handles exact keywords, dense embeddings handle paraphrases, Reciprocal Rank Fusion (k=60) merges the rankings without score normalization.
- **Chunking:** 150-word chunks with 120-word step for embeddings; the best chunk becomes the result snippet. Embeddings are cached in `data/chunk_embeddings.npy`.
- **Text cleaning:** Wikipedia LaTeX (`{\displaystyle ...}`) is removed before summarization.
- **Summaries:** lead-3 extractive baseline vs LLM abstractive summary.

## Results

Search (30 queries):

| Mode | Hit@5 | MRR |
|---|---|---|
| BM25 | 0.767 | 0.510 |
| Dense | 0.800 | 0.558 |
| Hybrid | 0.867 | 0.611 |

Summaries (20 docs, intro paragraph as reference):

| Method | ROUGE-1 | ROUGE-2 | ROUGE-L |
|---|---|---|---|
| Lead-3 | 0.274 | 0.066 | 0.155 |
| LLM | 0.305 | 0.051 | 0.153 |

Human eval (10 docs, blind A/B, 1-5): lead-3 avg 4.60, LLM avg 4.80. Details in `REPORT.md`.

## Screenshots

See `screenshots/`.