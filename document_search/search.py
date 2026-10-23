"""Hybrid document search: BM25 (keywords) + sentence embeddings (meaning), merged with Reciprocal Rank Fusion.

- BM25 indexes the whole article (title counted twice).
- Embeddings are computed per chunk (~130 words) because long articles do not fit in one embedding;
  a document's embedding score is the score of its best-matching chunk (that chunk is also the excerpt).
- Chunk embeddings are computed once and cached in data/chunk_embeddings.npy.
"""
import json
import os
import re
import sys
import time

import numpy as np
from rank_bm25 import BM25Okapi

CORPUS_PATH = "data/corpus.jsonl"
EMB_PATH = "data/chunk_embeddings.npy"
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

CHUNK_WORDS = 130       # ~800 characters, fits in the 256-token limit of the embedding model
CHUNK_STEP = 100        # 30 words of overlap between neighbouring chunks
MAX_CHUNKS = 10         # only the first ~1,000 words of each article are embedded (bounds compute)
RRF_K = 60              # Reciprocal Rank Fusion constant
CANDIDATES = 100        # how deep each ranking is used for fusion

STOPWORDS = set("""a an the and or of to in on at by for from with as is are was were be been it its this that these
those which who whom what when where why how not no but if then than so such also has have had into over under
about after before between during their his her they them he she we you i our your can could would should may
might will shall do does did""".split())


def tokenize(text):
    return [t for t in re.findall(r"[a-z0-9]+", text.lower()) if t not in STOPWORDS]


def make_chunks(doc):
    words = (doc["title"] + ". " + doc["text"]).split()
    chunks = []
    for start in range(0, max(len(words), 1), CHUNK_STEP):
        chunks.append(" ".join(words[start:start + CHUNK_WORDS]))
        if len(chunks) >= MAX_CHUNKS or start + CHUNK_WORDS >= len(words):
            break
    return chunks


def rrf(rankings, k=RRF_K):
    """rankings: list of lists of doc indices (best first). Returns {doc_index: fused score}."""
    fused = {}
    for ranking in rankings:
        for rank, idx in enumerate(ranking):
            fused[idx] = fused.get(idx, 0.0) + 1.0 / (k + rank + 1)
    return fused


class SearchEngine:
    def __init__(self, corpus_path=CORPUS_PATH):
        with open(corpus_path, encoding="utf-8") as f:
            self.docs = [json.loads(line) for line in f]

        # --- keyword index (BM25) ---
        self.bm25 = BM25Okapi([tokenize(d["title"] + " " + d["title"] + " " + d["text"]) for d in self.docs])

        # --- chunks for the embedding index ---
        self.chunks, self.starts = [], []
        for d in self.docs:
            self.starts.append(len(self.chunks))
            self.chunks.extend(make_chunks(d))
        self.starts = np.array(self.starts)
        self._model = None
        self.chunk_emb = None            # loaded lazily (only needed for embedding / hybrid search)

    # ---------- embeddings ----------
    def embed(self, texts, show_progress=False):
        if self._model is None:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(MODEL_NAME)
        return self._model.encode(texts, batch_size=64, normalize_embeddings=True,
                                  show_progress_bar=show_progress)

    def _ensure_embeddings(self):
        if self.chunk_emb is not None:
            return
        if os.path.exists(EMB_PATH):
            emb = np.load(EMB_PATH)
            if len(emb) == len(self.chunks):
                self.chunk_emb = emb
                return
        print(f"Embedding {len(self.chunks)} chunks (one-time, cached in {EMB_PATH}) ...")
        self.chunk_emb = np.asarray(self.embed(self.chunks, show_progress=True), dtype="float32")
        os.makedirs(os.path.dirname(EMB_PATH), exist_ok=True)
        np.save(EMB_PATH, self.chunk_emb)

    # ---------- the three rankings ----------
    def _bm25_scores(self, query):
        return np.asarray(self.bm25.get_scores(tokenize(query)))

    def _embedding_scores(self, query):
        """Returns (doc scores, similarity of every chunk)."""
        self._ensure_embeddings()
        q = np.asarray(self.embed([query]), dtype="float32")[0]
        sims = self.chunk_emb @ q                              # cosine (vectors are normalised)
        doc_scores = np.maximum.reduceat(sims, self.starts)    # best chunk per document
        return doc_scores, sims

    def _best_chunk(self, doc_idx, sims):
        start = self.starts[doc_idx]
        end = self.starts[doc_idx + 1] if doc_idx + 1 < len(self.starts) else len(self.chunks)
        return self.chunks[start + int(np.argmax(sims[start:end]))]

    def _overlap_chunk(self, doc_idx, query):
        q = set(tokenize(query))
        start = self.starts[doc_idx]
        end = self.starts[doc_idx + 1] if doc_idx + 1 < len(self.starts) else len(self.chunks)
        best = max(range(start, end), key=lambda i: len(q & set(tokenize(self.chunks[i]))))
        return self.chunks[best]

    # ---------- public API ----------
    def search(self, query, top_n=5, method="hybrid"):
        """method: 'bm25' | 'embedding' | 'hybrid'. Returns a list of dicts (best first)."""
        sims = None
        if method == "bm25":
            scores = self._bm25_scores(query)
            order = np.argsort(-scores)[:top_n]
            final = {int(i): float(scores[i]) for i in order}
        elif method == "embedding":
            scores, sims = self._embedding_scores(query)
            order = np.argsort(-scores)[:top_n]
            final = {int(i): float(scores[i]) for i in order}
        elif method == "hybrid":
            bm = self._bm25_scores(query)
            em, sims = self._embedding_scores(query)
            fused = rrf([list(np.argsort(-bm)[:CANDIDATES]), list(np.argsort(-em)[:CANDIDATES])])
            top = sorted(fused.items(), key=lambda kv: -kv[1])[:top_n]
            final = {int(i): float(s) for i, s in top}
        else:
            raise ValueError("method must be 'bm25', 'embedding' or 'hybrid'")

        results = []
        for idx, score in final.items():
            d = self.docs[idx]
            excerpt = self._best_chunk(idx, sims) if sims is not None else self._overlap_chunk(idx, query)
            results.append({"id": d["id"], "title": d["title"], "topic": d["topic"],
                            "score": round(score, 4), "excerpt": excerpt})
        return results


if __name__ == "__main__":
    engine = SearchEngine()
    print(f"Loaded {len(engine.docs)} documents, {len(engine.chunks)} chunks")
    query = " ".join(sys.argv[1:]) or input("Query> ")
    for method in ("bm25", "embedding", "hybrid"):
        t0 = time.time()
        res = engine.search(query, top_n=5, method=method)
        print(f"\n[{method}]  ({time.time() - t0:.2f}s)")
        for r in res:
            print(f"  {r['title']}  ({r['topic']}, score {r['score']})")