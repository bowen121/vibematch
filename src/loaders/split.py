"""Genre-stratified train/val/test splitting for multi-label data.

Each item carries a list of genres. To avoid rare genres ending up entirely in
one split (which would break our generalization-to-unseen-styles eval), we
stratify on each item's *rarest* genre instead of doing a plain random split.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Sequence

import numpy as np


@dataclass(frozen=True)
class Split:
    """Indices into the original dataset for each split."""
    train: np.ndarray
    val: np.ndarray
    test: np.ndarray


def genre_stratified_split(
    genres: Sequence[Sequence[str]],
    val_frac: float = 0.1,
    test_frac: float = 0.1,
    seed: int = 42,
) -> Split:
    """Return train/val/test index arrays stratified by each item's rarest genre.

    Why rarest genre: if "Romance" appears 100x and "Western" 5x, stratifying on
    Western is what protects the tail. An item tagged ["Romance", "Western"] is
    grouped under "Western" so the 5 Westerns are split 4/0/1 (or 3/1/1) instead
    of clustering randomly.

    All three returned arrays are disjoint and union to range(len(genres)).
    """
    n = len(genres)
    if not 0.0 <= val_frac < 1.0 or not 0.0 <= test_frac < 1.0:
        raise ValueError("val_frac and test_frac must be in [0, 1)")
    if val_frac + test_frac >= 1.0:
        raise ValueError("val_frac + test_frac must be < 1")
    if n == 0:
        empty = np.array([], dtype=np.int64)
        return Split(train=empty, val=empty, test=empty)

    counts: Counter[str] = Counter()
    for g_list in genres:
        counts.update(g_list)

    keys: list[str] = []
    for g_list in genres:
        if not g_list:
            keys.append("__missing__")
        else:
            keys.append(min(g_list, key=lambda g: counts[g]))

    by_key: dict[str, list[int]] = {}
    for i, k in enumerate(keys):
        by_key.setdefault(k, []).append(i)

    rng = np.random.default_rng(seed)
    train: list[int] = []
    val: list[int] = []
    test: list[int] = []
    for _, idxs_list in by_key.items():
        idxs = np.array(idxs_list, dtype=np.int64)
        rng.shuffle(idxs)
        n_k = len(idxs)
        n_test = int(round(n_k * test_frac))
        n_val = int(round(n_k * val_frac))
        # Tiny groups: prefer leaving at least one item in train.
        if n_test + n_val >= n_k and n_k > 0:
            n_val = min(n_val, n_k - 1)
            n_test = max(0, n_k - 1 - n_val)
        test.extend(idxs[:n_test].tolist())
        val.extend(idxs[n_test : n_test + n_val].tolist())
        train.extend(idxs[n_test + n_val :].tolist())

    return Split(
        train=np.sort(np.array(train, dtype=np.int64)),
        val=np.sort(np.array(val, dtype=np.int64)),
        test=np.sort(np.array(test, dtype=np.int64)),
    )


def build_genre_vocab(genres: Sequence[Sequence[str]], min_count: int = 1) -> list[str]:
    """Return the sorted list of genres appearing at least `min_count` times.

    Index in the returned list is the position in the multi-hot label vector.
    """
    counts: Counter[str] = Counter()
    for g_list in genres:
        counts.update(g_list)
    return sorted(g for g, c in counts.items() if c >= min_count)


def multi_hot(genres: Sequence[str], vocab: Sequence[str]) -> np.ndarray:
    """Encode a single item's genre list as a multi-hot vector over `vocab`."""
    idx = {g: i for i, g in enumerate(vocab)}
    vec = np.zeros(len(vocab), dtype=np.float32)
    for g in genres:
        if g in idx:
            vec[idx[g]] = 1.0
    return vec
