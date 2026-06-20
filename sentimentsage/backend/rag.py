# backend/rag.py
# RAG — Retrieval-Augmented Generation via Cosine Similarity
# Week 12: Distance Measurement — Cosine Similarity
# EXPLAIN: "After classifying the review, I take its 512-dim latent vector
#           and compute cosine similarity against 1,000 reference vectors.
#           Cosine similarity measures the angle between vectors — not magnitude.
#           Two reviews about similar topics point in the same direction."

import pickle
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


def load_rag_store(path='rag_store.pkl'):
    """Load the pre-built RAG vector store."""
    with open(path, 'rb') as f:
        store = pickle.load(f)
    print(f'RAG store loaded: {len(store["labels"])} reference reviews')
    return store


def search(query_vector: np.ndarray, store: dict, top_k: int = 3) -> list:
    """
    Find top_k most similar reviews using cosine similarity.

    Week 12 concept:
      cosine_similarity(A, B) = (A · B) / (||A|| × ||B||)
      Result: 1.0 = identical direction, 0.0 = orthogonal, -1.0 = opposite

    Args:
        query_vector: 512-dim latent vector from LSTM for the input review
        store:        dict with 'vectors', 'labels', 'snippets', 'texts'
        top_k:        number of similar reviews to return

    Returns:
        list of dicts with snippet, label, similarity score
    """
    if query_vector.ndim == 1:
        query_vector = query_vector.reshape(1, -1)

    # Compute cosine similarity between query and all reference vectors
    similarities = cosine_similarity(query_vector, store['vectors'])[0]   # shape: (1000,)

    # Get top_k indices sorted by similarity (descending)
    top_indices = similarities.argsort()[-top_k:][::-1]

    results = []
    for idx in top_indices:
        results.append({
            'snippet':    store['snippets'][idx],
            'label':      store['labels'][idx],
            'similarity': round(float(similarities[idx]) * 100, 1),
        })

    return results
