"""Cosine-similarity query and top-k retrieval."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import faiss
import numpy as np

from src.retrieval.engine import normalise


@dataclass

class SearchResult:
    """Single retrieval result returned by query()."""
    rank: int
    score: float
    image_path: str
    metadata: dict[str, Any]


def query(
    index: faiss.IndexFlatIP,
    metadata: list[dict[str, Any]],
    query_vector: np.ndarray,
    top_k: int = 10,
) -> list[SearchResult]:
    """Return the top-k most similar items for a single query vector.

    query_vector can be shape (D,) or (1, D); will be L2-normalised automatically.
    top_k is clamped to index.ntotal if larger.
    """
    vec = _prepare_query(query_vector, expected_dim=index.d)
    k = min(top_k, index.ntotal)
    if k == 0:
        return []
    scores, indices = index.search(vec, k)
    results = []
    for rank, (idx, score) in enumerate(zip(indices[0], scores[0]), start=1):
        if idx == -1:
            break
        meta = metadata[idx]
        results.append(SearchResult(
            rank=rank,
            score=float(score),
            image_path=meta.get("image_path", ""),
            metadata=meta,
        ))
    return results


def batch_query(
    index: faiss.IndexFlatIP,
    metadata: list[dict[str, Any]],
    query_vectors: np.ndarray,
    top_k: int = 10,
) -> list[list[SearchResult]]:
    """Run query() for multiple vectors in a single FAISS call. Useful for eval recall@k."""
    vecs = np.vstack([
        _prepare_query(query_vectors[i], expected_dim=index.d)
        for i in range(len(query_vectors))
    ])
    k = min(top_k, index.ntotal)
    if k == 0:
        return [[] for _ in range(len(query_vectors))]
    all_scores, all_indices = index.search(vecs, k)
    output = []
    for scores_row, idx_row in zip(all_scores, all_indices):
        results = []
        for rank, (idx, score) in enumerate(zip(idx_row, scores_row), start=1):
            if idx == -1:
                break
            meta = metadata[idx]
            results.append(SearchResult(
                rank=rank,
                score=float(score),
                image_path=meta.get("image_path", ""),
                metadata=meta,
            ))
        output.append(results)
    return output


def _prepare_query(vec: np.ndarray, expected_dim: int) -> np.ndarray:
    """Validate, reshape to (1, D), and L2-normalise a query vector."""
    vec = np.array(vec, dtype=np.float32)
    if vec.ndim == 1:
        vec = vec.reshape(1, -1)
    elif not (vec.ndim == 2 and vec.shape[0] == 1):
        raise ValueError(f"query_vector must be 1-D or (1, D), got shape {vec.shape}")
    if vec.shape[1] != expected_dim:
        raise ValueError(
            f"query dimension {vec.shape[1]} does not match index dimension {expected_dim}"
        )
    return normalise(vec)


