"""FAISS index building and saving."""
 
from __future__ import annotations
 
import json
import os
from pathlib import Path
from typing import Any
 
import faiss
import numpy as np
 
 
def build_index(
    embeddings: np.ndarray,
    metadata: list[dict[str, Any]],
) -> tuple[faiss.IndexFlatIP, list[dict[str, Any]]]:
    """Build a flat IP index from L2-normalised embeddings. Raises ValueError if lengths differ."""
    if len(embeddings) != len(metadata):
        raise ValueError(
            f"embeddings and metadata must have the same length "
            f"({len(embeddings)} vs {len(metadata)})"
        )
    embeddings = _to_float32(embeddings)
    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)
    return index, metadata
 
 
def save_index(
    index: faiss.IndexFlatIP,
    metadata: list[dict[str, Any]],
    index_path: str | os.PathLike,
) -> None:
    """Write index to <index_path> and metadata to <index_path>.meta.json."""
    index_path = Path(index_path)
    index_path.parent.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(index_path))
    meta_path = index_path.with_suffix(index_path.suffix + ".meta.json")
    with open(meta_path, "w", encoding="utf-8") as fh:
        json.dump(metadata, fh, ensure_ascii=False, indent=2)
 
 
def load_index(
    index_path: str | os.PathLike,
) -> tuple[faiss.IndexFlatIP, list[dict[str, Any]]]:
    """Load index and metadata sidecar from disk. Raises FileNotFoundError if either is missing."""
    index_path = Path(index_path)
    meta_path = index_path.with_suffix(index_path.suffix + ".meta.json")
    if not index_path.exists():
        raise FileNotFoundError(f"FAISS index not found: {index_path}")
    if not meta_path.exists():
        raise FileNotFoundError(f"Metadata sidecar not found: {meta_path}")
    index = faiss.read_index(str(index_path))
    with open(meta_path, "r", encoding="utf-8") as fh:
        metadata = json.load(fh)
    return index, metadata
 
 
def normalise(embeddings: np.ndarray) -> np.ndarray:
    """L2-normalise embeddings in-place so IP search == cosine similarity."""
    embeddings = _to_float32(embeddings)
    faiss.normalize_L2(embeddings)
    return embeddings
 
 
def _to_float32(embeddings: np.ndarray) -> np.ndarray:
    """Cast to C-contiguous float32, as required by FAISS."""
    if embeddings.dtype != np.float32:
        embeddings = embeddings.astype(np.float32)
    if not embeddings.flags["C_CONTIGUOUS"]:
        embeddings = np.ascontiguousarray(embeddings)
    return embeddings
 
