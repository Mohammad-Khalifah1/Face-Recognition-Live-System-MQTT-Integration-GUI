from __future__ import annotations

import numpy as np


def normalize(vector: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(vector)
    if norm == 0:
        return vector
    return vector / norm


def match_embedding(
    embedding: np.ndarray,
    labels: np.ndarray,
    centroids: np.ndarray,
    threshold: float,
) -> dict[str, object]:
    scores = centroids @ embedding
    best_index = int(np.argmax(scores))
    similarity = float(scores[best_index])
    label = str(labels[best_index])
    accepted = similarity >= threshold
    return {
        "label": label if accepted else "unknown",
        "similarity": similarity,
        "accepted": accepted,
        "percent": max(0.0, min(100.0, similarity * 100.0)),
    }


def match_all_scores(
    embedding: np.ndarray,
    labels: np.ndarray,
    centroids: np.ndarray,
) -> list[dict[str, object]]:
    scores = centroids @ embedding
    results = []
    for i in range(len(labels)):
        similarity = float(scores[i])
        results.append({
            "label": str(labels[i]),
            "similarity": similarity,
            "percent": max(0.0, min(100.0, similarity * 100.0)),
        })
    results.sort(key=lambda r: r["similarity"], reverse=True)
    return results


def match_embedding_knn(
    embedding: np.ndarray,
    sample_labels: np.ndarray,
    sample_embeddings: np.ndarray,
    k: int = 3,
    threshold: float = 0.38,
) -> dict[str, object]:
    scores = sample_embeddings @ embedding
    top_indices = np.argsort(scores)[::-1]
    top_k = top_indices[:k]
    top_k_labels = [str(sample_labels[i]) for i in top_k]
    from collections import Counter
    label_counts = Counter(top_k_labels)
    best_label = label_counts.most_common(1)[0][0]
    mask = np.array([l == best_label for l in top_k_labels])
    avg_similarity = float(np.mean(scores[top_k[mask]]))
    accepted = avg_similarity >= threshold
    return {
        "label": best_label if accepted else "unknown",
        "similarity": avg_similarity,
        "accepted": accepted,
        "percent": max(0.0, min(100.0, avg_similarity * 100.0)),
        "votes": dict(label_counts),
    }
