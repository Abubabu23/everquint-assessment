import json
import os
import re

import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

DOCS_PATH = os.path.join("data", "documents.json")
EMB_PATH = os.path.join("data", "chunk_embeddings.npy")
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
CHUNK_WORDS = 150
CHUNK_STEP = 120
RRF_K = 60
CANDIDATES = 50


def tokenize(text):
    return re.findall(r"\w+", text.lower())


def make_chunks(docs):
    chunks, owners = [], []
    for d_idx, d in enumerate(docs):
        words = d["text"].split()
        for start in range(0, max(len(words), 1), CHUNK_STEP):
            piece = " ".join(words[start:start + CHUNK_WORDS])
            if piece:
                chunks.append(piece)
                owners.append(d_idx)
            if start + CHUNK_WORDS >= len(words):
                break
    return chunks, np.array(owners)


class HybridSearcher:
    def __init__(self):
        with open(DOCS_PATH, encoding="utf-8") as f:
            self.docs = json.load(f)
        self.chunks, self.owners = make_chunks(self.docs)

        self.bm25 = BM25Okapi([tokenize(d["title"] + " " + d["text"]) for d in self.docs])

        self.model = SentenceTransformer(MODEL_NAME)
        if os.path.exists(EMB_PATH):
            self.emb = np.load(EMB_PATH)
            if len(self.emb) != len(self.chunks):
                self.emb = self._build_embeddings()
        else:
            self.emb = self._build_embeddings()

    def _build_embeddings(self):
        print(f"Embedding {len(self.chunks)} chunks (one-time, konja neram aagum)...")
        emb = self.model.encode(
            self.chunks, batch_size=64, normalize_embeddings=True, show_progress_bar=True
        )
        np.save(EMB_PATH, emb)
        return emb

    def _bm25_ranking(self, query):
        scores = self.bm25.get_scores(tokenize(query))
        order = np.argsort(scores)[::-1][:CANDIDATES]
        return [int(i) for i in order]

    def _dense_ranking(self, query):
        q = self.model.encode([query], normalize_embeddings=True)[0]
        sims = self.emb @ q
        order = np.argsort(sims)[::-1]
        ranking, best_chunk, seen = [], {}, set()
        for c in order:
            d = int(self.owners[c])
            if d in seen:
                continue
            seen.add(d)
            ranking.append(d)
            best_chunk[d] = int(c)
            if len(ranking) >= CANDIDATES:
                break
        return ranking, best_chunk

    def search(self, query, k=5, mode="hybrid"):
        bm25_rank = self._bm25_ranking(query)
        dense_rank, best_chunk = self._dense_ranking(query)

        if mode == "bm25":
            final = [(d, 1.0 / (i + 1)) for i, d in enumerate(bm25_rank)]
        elif mode == "dense":
            final = [(d, 1.0 / (i + 1)) for i, d in enumerate(dense_rank)]
        else:
            scores = {}
            for ranking in (bm25_rank, dense_rank):
                for i, d in enumerate(ranking):
                    scores[d] = scores.get(d, 0.0) + 1.0 / (RRF_K + i + 1)
            final = sorted(scores.items(), key=lambda x: x[1], reverse=True)

        results = []
        for d, score in final[:k]:
            doc = self.docs[d]
            snippet = self.chunks[best_chunk[d]] if d in best_chunk else doc["text"][:600]
            results.append(
                {
                    "doc_id": doc["id"],
                    "title": doc["title"],
                    "url": doc["url"],
                    "score": float(score),
                    "snippet": snippet,
                }
            )
        return results


if __name__ == "__main__":
    s = HybridSearcher()
    while True:
        q = input("\nQuery (exit na 'q'): ").strip()
        if q.lower() == "q":
            break
        for r in s.search(q, k=5):
            print(f"- {r['title']}  ({r['score']:.4f})")