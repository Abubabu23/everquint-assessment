from search import HybridSearcher

# (query, expected Wikipedia title)
QUERIES = [
    ("algorithm that trains neural nets by propagating error backward", "Backpropagation"),
    ("optimization method that steps along the negative gradient", "Gradient descent"),
    ("model memorizes training data and fails on new data", "Overfitting"),
    ("agent learns by rewards and penalties from environment", "Reinforcement learning"),
    ("classifier that finds maximum-margin hyperplane", "Support vector machine"),
    ("ensemble of many decision trees voting together", "Random forest"),
    ("partition data into k clusters by nearest centroid", "K-means clustering"),
    ("reduce dimensionality by projecting onto principal axes", "Principal component analysis"),
    ("probabilistic classifier assuming feature independence", "Naive Bayes classifier"),
    ("tree-based model that splits data on feature values", "Decision tree learning"),
    ("dense vector representation of words", "Word embedding"),
    ("recurrent network with gates to remember long sequences", "Long short-term memory"),
    ("network that compresses input into a latent code and reconstructs it", "Autoencoder"),
    ("generator and discriminator networks trained against each other", "Generative adversarial network"),
    ("off-policy value-based reinforcement learning algorithm", "Q-learning"),
    ("model predicting binary outcome with sigmoid function", "Logistic regression"),
    ("fit a straight line to predict a numeric value", "Linear regression"),
    ("randomly dropping units during training to prevent overfitting", "Dropout (neural networks)"),
    ("normalizing layer inputs across a mini-batch", "Batch normalization"),
    ("simplest single-layer binary linear classifier", "Perceptron"),
    ("tuning hyperparameters of a learning algorithm", "Hyperparameter optimization"),
    ("splitting data into folds to estimate model performance", "Cross-validation (statistics)"),
    ("tradeoff between underfitting and overfitting error sources", "Bias–variance tradeoff"),
    ("identify people, places and organizations in text", "Named-entity recognition"),
    ("automatically translate text between languages", "Machine translation"),
    ("locate and classify objects in images with bounding boxes", "Object detection"),
    ("assign a class label to every pixel in an image", "Image segmentation"),
    ("neural network layers using convolution filters for images", "Convolutional neural network"),
    ("neural networks with loops for sequential data", "Recurrent neural network"),
    ("function deciding whether a neuron fires", "Activation function"),
]

K = 5


def evaluate(searcher, queries, mode):
    hits, rr = 0, 0.0
    misses = []
    for q, expected in queries:
        results = searcher.search(q, k=10, mode=mode)
        titles = [r["title"] for r in results]
        if expected in titles[:K]:
            hits += 1
        if expected in titles:
            rr += 1.0 / (titles.index(expected) + 1)
        else:
            misses.append((q, expected, titles[:3]))
    n = len(queries)
    return hits / n, rr / n, misses


if __name__ == "__main__":
    s = HybridSearcher()
    available = {d["title"] for d in s.docs}
    valid = [(q, t) for q, t in QUERIES if t in available]
    skipped = [t for q, t in QUERIES if t not in available]
    print(f"Using {len(valid)} queries. Skipped (not in corpus): {skipped}\n")

    for mode in ("bm25", "dense", "hybrid"):
        hit, mrr, misses = evaluate(s, valid, mode)
        print(f"{mode:7s} Hit@{K} = {hit:.3f}   MRR = {mrr:.3f}")
        if mode == "hybrid":
            for q, exp, got in misses:
                print("  MISS:", q, "| expected:", exp, "| got:", got)