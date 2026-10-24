import json
import os
import time

import requests

API = "https://en.wikipedia.org/w/api.php"
HEADERS = {"User-Agent": "everquint-assessment/1.0 (student project; abu@example.com)"}
OUT_PATH = os.path.join("data", "documents.json")
MAX_DOCS = 1000
MIN_CHARS = 500
MAX_SUBCATS = 15  # per category

CATEGORIES = [
    "Machine learning",
    "Artificial intelligence",
    "Deep learning",
    "Natural language processing",
    "Computer vision",
    "Artificial neural networks",
    "Reinforcement learning",
    "Large language models",
]

SEED_TITLES = [
    "Gradient descent", "Reinforcement learning", "K-means clustering",
    "Long short-term memory", "Generative adversarial network", "Q-learning",
    "Logistic regression", "Linear regression", "Dropout (neural networks)",
    "Batch normalization", "Named-entity recognition", "Machine translation",
    "Object detection", "Image segmentation", "Convolutional neural network",
    "Recurrent neural network", "Backpropagation", "Overfitting",
    "Support vector machine", "Random forest", "Principal component analysis",
    "Naive Bayes classifier", "Decision tree learning", "Word embedding",
    "Autoencoder", "Perceptron", "Activation function", "Transformer (deep learning)",
    "Attention (machine learning)", "BERT (language model)", "Word2vec",
    "Stochastic gradient descent", "Supervised learning", "Unsupervised learning",
    "Semi-supervised learning", "Transfer learning", "Feature engineering",
    "Regularization (mathematics)", "Deep learning", "Machine learning",
    "Hyperparameter optimization", "Cross-validation (statistics)",
    "Bias–variance tradeoff",
]


def get_json(params):
    for attempt in range(5):
        try:
            r = requests.get(API, params=params, headers=HEADERS, timeout=30)
            if r.status_code == 200:
                return r.json()
            print("status", r.status_code, "- waiting...")
        except Exception as e:
            print("error", e, "- retrying...")
        time.sleep(2 * (attempt + 1))
    raise RuntimeError("Wikipedia request failed")


def category_members(category):
    """Return (pages, subcategories) for a category."""
    pages, subcats = [], []
    params = {
        "action": "query",
        "list": "categorymembers",
        "cmtitle": f"Category:{category}",
        "cmlimit": 500,
        "format": "json",
    }
    while True:
        r = get_json(params)
        time.sleep(0.3)
        for m in r["query"]["categorymembers"]:
            if m["ns"] == 0:
                pages.append(m["title"])
            elif m["ns"] == 14:
                subcats.append(m["title"].replace("Category:", ""))
        if "continue" not in r:
            break
        params["cmcontinue"] = r["continue"]["cmcontinue"]
    return pages, subcats


def collect_titles():
    titles = list(SEED_TITLES)
    seen = set(titles)
    for cat in CATEGORIES:
        pages, subcats = category_members(cat)
        for sub in subcats[:MAX_SUBCATS]:  # one level deep
            try:
                p, _ = category_members(sub)
                pages.extend(p)
            except Exception as e:
                print("skip subcategory", sub, e)
        for t in pages:
            if t not in seen:
                seen.add(t)
                titles.append(t)
        print(f"{cat}: unique titles so far = {len(titles)}")
        if len(titles) >= MAX_DOCS * 1.5:
            break
    return titles


def fetch_text(title):
    params = {
        "action": "query",
        "prop": "extracts",
        "explaintext": 1,
        "titles": title,
        "redirects": 1,
        "format": "json",
    }
    r = get_json(params)
    page = next(iter(r["query"]["pages"].values()))
    return page.get("extract", "")


def main():
    os.makedirs("data", exist_ok=True)
    titles = collect_titles()
    docs = []
    for t in titles:
        if len(docs) >= MAX_DOCS:
            break
        try:
            text = fetch_text(t)
        except Exception as e:
            print("skip", t, e)
            continue
        if len(text) < MIN_CHARS:
            print("short/missing:", t)
            continue
        docs.append(
            {
                "id": len(docs),
                "title": t,
                "url": "https://en.wikipedia.org/wiki/" + t.replace(" ", "_"),
                "text": text,
            }
        )
        if len(docs) % 50 == 0:
            print(f"{len(docs)} docs collected")
        time.sleep(0.2)

    with open(OUT_PATH, "w", encoding="utf-8") as f:
        json.dump(docs, f, ensure_ascii=False)
    print(f"Saved {len(docs)} docs to {OUT_PATH}")


if __name__ == "__main__":
    main()