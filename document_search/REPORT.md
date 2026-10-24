# Task 5 Report: Document Search + Summarization

## 1. Overview

A search and summarization system over a 1000-article AI/ML Wikipedia corpus:

- **Search:** BM25 + MiniLM (all-MiniLM-L6-v2) chunk embeddings, fused with Reciprocal Rank Fusion (RRF).
- **Summarization:** lead-3 extractive baseline vs LLM abstractive summaries (Groq, openai/gpt-oss-120b).
- **Evaluation:** Hit@5 / MRR for search, ROUGE and a blind human A/B test for summaries.
- **UI:** Streamlit app (`app.py`).

## 2. Search Evaluation

30 queries, a result counts as a hit if the expected article is in the top 5.

| Mode | Hit@5 | MRR |
|---|---|---|
| BM25 | 0.767 | 0.510 |
| Dense | 0.800 | 0.558 |
| Hybrid (RRF) | 0.867 | 0.611 |

Hybrid is best on both metrics: +10 points Hit@5 over BM25 and +6.7 over dense. BM25 catches exact keyword matches, dense retrieval catches paraphrases, and RRF combines the two rankings without needing score normalization.

**Observed weakness:** for the broad query "supervised learning", the top hybrid result was "Representation learning" rather than the "Supervised learning" article. Broad topic queries match many related articles that mention the term often, and chunk-level retrieval can favor a chunk with dense keyword usage over the article that is actually about the topic.

## 3. Summarization Evaluation

### 3.1 ROUGE

20 documents, the article's intro paragraph as reference, LaTeX removed by `clean_text`.

| Method | ROUGE-1 | ROUGE-2 | ROUGE-L |
|---|---|---|---|
| Lead-3 baseline | 0.274 | 0.066 | 0.155 |
| LLM | 0.305 | 0.051 | 0.153 |

The LLM is higher on ROUGE-1, lower on ROUGE-2, and tied on ROUGE-L. The reference is the article's own intro, and lead-3 copies the first sentences of the same text, so lead-3 has a built-in advantage in exact bigram overlap. The LLM paraphrases, which lowers exact n-gram matches even when the content is good. ROUGE alone therefore understates the LLM.

### 3.2 Human evaluation

10 documents, blind A/B (positions randomly shuffled, key stored separately), 1-5 rating for accuracy and coverage of the main point.

| Method | Average rating |
|---|---|
| Lead-3 baseline | 4.60 |
| LLM | 4.80 |

Per-document comparison: LLM won 2, baseline won 1, tied 7.

The LLM has a slight edge, but with only 10 samples and 7 ties the difference is not statistically meaningful. The two methods perform similarly on this corpus. The LLM's advantage showed up on articles whose opening sentences are definitions with missing formulas or lack the main topic (for example WaveNet, where lead-3 only covered text-to-speech background). Lead-3 does well when the opening paragraph is already a good summary.

## 4. Limitations

- **Text cleaning gaps:** removing LaTeX leaves broken phrases such as "is where is an increasing function", which hurts both methods.
- **Reference choice:** using the intro paragraph as reference favors extractive methods under ROUGE.
- **Small human eval:** 10 documents, one rater, so no inter-rater agreement and low statistical power.
- **Small query set:** 30 queries, one expected article per query, so related-but-valid results count as misses.
- **Chunk snippets:** snippets can start mid-sentence and contain Wikipedia markup (`== Heading ==`).
- **LLM cost and dependency:** LLM summaries need an API key and are rate-limited (the Gemini free tier quota ran out, so Groq is used).

## 5. Possible Improvements

- Strip Wikipedia section markup from snippets.
- Add a cross-encoder reranker on the top 20 hybrid results.
- Boost exact title matches for short broad queries.
- Evaluate summaries with additional metrics (BERTScore) and more human raters.

## 6. Screenshots

| Search results | Summaries side by side | Search modes |
|---|---|---|
| ![search](screenshots/01_search_results.png) | ![summarize](screenshots/02_summarize.png) | ![modes](screenshots/03_modes.png) |