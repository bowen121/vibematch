"""Tests for src/loaders/* (Member A's scope).

End-to-end pipeline smoke tests live in test_pipeline.py (Member C's scope).
These tests use only synthetic fixtures so they run without the Kaggle data.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
from PIL import Image

from src.loaders.dataset import (
    MediaDataset,
    build_image_transform,
    parse_genres,
    render_prompt,
)
from src.loaders.data_loader import (
    build_source_balanced_sampler,
    load_processed_frame,
    make_data_bundle,
)
from src.loaders.split import (
    build_genre_vocab,
    genre_stratified_split,
    multi_hot,
)


# ---------- split.py ----------


def test_split_is_deterministic_with_seed():
    genres = [["A", "B"], ["B"], ["C"], ["A"], ["A", "C"], ["B", "C"], ["A"], ["B"]]
    s1 = genre_stratified_split(genres, val_frac=0.2, test_frac=0.2, seed=7)
    s2 = genre_stratified_split(genres, val_frac=0.2, test_frac=0.2, seed=7)
    assert np.array_equal(s1.train, s2.train)
    assert np.array_equal(s1.val, s2.val)
    assert np.array_equal(s1.test, s2.test)


def test_split_has_no_leak_and_is_complete():
    genres = [["A"], ["B"], ["A", "B"], ["C"], ["A", "C"]] * 20
    s = genre_stratified_split(genres, val_frac=0.1, test_frac=0.1, seed=0)
    all_idx = np.concatenate([s.train, s.val, s.test])
    assert len(set(all_idx.tolist())) == len(genres)
    assert sorted(all_idx.tolist()) == list(range(len(genres)))


def test_split_keeps_rare_genre_in_train():
    # 2 rare ("Western"), 50 common ("Drama"). Rare should not vanish from train.
    genres = [["Western"]] * 2 + [["Drama"]] * 50
    s = genre_stratified_split(genres, val_frac=0.1, test_frac=0.1, seed=42)
    train_genres = [genres[i] for i in s.train]
    assert any("Western" in g for g in train_genres), "Rare genre lost from train split"


def test_split_rejects_bad_fractions():
    with pytest.raises(ValueError):
        genre_stratified_split([["A"]], val_frac=0.6, test_frac=0.6)
    with pytest.raises(ValueError):
        genre_stratified_split([["A"]], val_frac=-0.1, test_frac=0.1)


def test_split_handles_empty_input():
    s = genre_stratified_split([])
    assert len(s.train) == 0 and len(s.val) == 0 and len(s.test) == 0


def test_build_genre_vocab_min_count():
    genres = [["A", "B"], ["A"], ["A"], ["B"], ["C"]]
    vocab = build_genre_vocab(genres, min_count=2)
    assert vocab == ["A", "B"]


def test_multi_hot():
    vocab = ["Action", "Drama", "Romance"]
    vec = multi_hot(["Drama", "Romance", "Unknown"], vocab)
    assert vec.tolist() == [0.0, 1.0, 1.0]
    assert vec.dtype == np.float32


# ---------- dataset.py helpers ----------


def test_parse_genres_handles_pipe_and_nan():
    assert parse_genres("Drama|Romance") == ["Drama", "Romance"]
    assert parse_genres("Drama| Romance |Comedy") == ["Drama", "Romance", "Comedy"]
    assert parse_genres(float("nan")) == []
    assert parse_genres(None) == []
    assert parse_genres(["A", "B"]) == ["A", "B"]


def test_render_prompt_deterministic_when_rng_none():
    p = render_prompt(["Drama", "Romance"], "movie", rng=None)
    assert p == "a movie poster of a drama film"


def test_render_prompt_uses_rng_when_provided():
    rng = np.random.default_rng(0)
    p = render_prompt(["Drama", "Romance", "Action"], "movie", rng=rng)
    assert p.startswith("a movie poster of a ")
    assert p.endswith(" film")


def test_render_prompt_falls_back_for_empty():
    assert "generic" in render_prompt([], "movie", rng=None)


def test_image_transform_output_shape():
    t = build_image_transform(train=False)
    img = Image.new("RGB", (300, 200), color=(128, 64, 200))
    out = t(img)
    assert out.shape == (3, 224, 224)
    assert out.dtype == torch.float32


# ---------- MediaDataset + DataLoader ----------


class _FakeTokenizer:
    """Tiny tokenizer that matches the HF interface used by MediaDataset."""

    def __call__(self, text, padding, truncation, max_length, return_tensors):
        ids = [hash(tok) % 1000 for tok in text.split()][:max_length]
        ids = ids + [0] * (max_length - len(ids))
        mask = [1 if i < len(text.split()) and i < max_length else 0 for i in range(max_length)]
        return {
            "input_ids": torch.tensor([ids], dtype=torch.long),
            "attention_mask": torch.tensor([mask], dtype=torch.long),
        }


def _make_synthetic_data(tmp_path: Path, n_movies: int = 6, n_books: int = 6) -> Path:
    """Build data/raw images + data/processed CSVs under tmp_path. Returns repo-style root."""
    img_dir = tmp_path / "data" / "raw" / "imgs"
    img_dir.mkdir(parents=True)
    proc_dir = tmp_path / "data" / "processed"
    proc_dir.mkdir(parents=True)

    rng = np.random.default_rng(0)
    rows = []
    movie_genres = [["Drama"], ["Action"], ["Drama", "Romance"], ["Comedy"], ["Action"], ["Western"]]
    book_genres = [["Fiction"], ["Mystery"], ["Fiction", "Romance"], ["Sci-Fi"], ["Fiction"], ["Fantasy"]]

    for i in range(n_movies):
        path = img_dir / f"movie_{i}.png"
        Image.fromarray(rng.integers(0, 255, (32, 32, 3), dtype=np.uint8)).save(path)
        rows.append({
            "id": f"movie_{i}",
            "image_path": str(path.relative_to(tmp_path)),
            "title": f"Movie {i}",
            "genres": "|".join(movie_genres[i % len(movie_genres)]),
            "source": "movie",
        })
    movies_df = pd.DataFrame(rows[:n_movies])
    movies_df.to_csv(proc_dir / "movies.csv", index=False)

    rows2 = []
    for i in range(n_books):
        path = img_dir / f"book_{i}.png"
        Image.fromarray(rng.integers(0, 255, (32, 32, 3), dtype=np.uint8)).save(path)
        rows2.append({
            "id": f"book_{i}",
            "image_path": str(path.relative_to(tmp_path)),
            "title": f"Book {i}",
            "genres": "|".join(book_genres[i % len(book_genres)]),
            "source": "book",
        })
    pd.DataFrame(rows2).to_csv(proc_dir / "books.csv", index=False)
    return tmp_path


def test_load_processed_frame_concatenates(tmp_path: Path):
    root = _make_synthetic_data(tmp_path)
    df = load_processed_frame(root / "data" / "processed")
    assert len(df) == 12
    assert set(df["source"]) == {"movie", "book"}


def test_media_dataset_returns_correct_shapes(tmp_path: Path):
    root = _make_synthetic_data(tmp_path)
    df = load_processed_frame(root / "data" / "processed")
    vocab = ["Action", "Comedy", "Drama", "Fantasy", "Fiction", "Mystery", "Romance", "Sci-Fi", "Western"]
    ds = MediaDataset(df, data_root=root, genre_vocab=vocab, tokenizer=_FakeTokenizer(), train=False)
    assert len(ds) == 12
    sample = ds[0]
    assert sample["image"].shape == (3, 224, 224)
    assert sample["input_ids"].shape == (32,)
    assert sample["attention_mask"].shape == (32,)
    assert sample["labels"].shape == (len(vocab),)
    assert sample["labels"].dtype == torch.float32


def test_media_dataset_rejects_bad_schema(tmp_path: Path):
    bad = pd.DataFrame({"id": [1], "title": ["x"]})
    with pytest.raises(ValueError, match="missing required columns"):
        MediaDataset(bad, data_root=tmp_path, genre_vocab=["A"], tokenizer=_FakeTokenizer())


def test_make_data_bundle_end_to_end(tmp_path: Path):
    root = _make_synthetic_data(tmp_path)
    bundle = make_data_bundle(
        processed_dir=root / "data" / "processed",
        data_root=root,
        tokenizer=_FakeTokenizer(),
        batch_size=4,
        num_workers=0,
        val_frac=0.2,
        test_frac=0.2,
        seed=0,
        min_genre_count=1,
        pin_memory=False,
    )
    assert len(bundle.genre_vocab) > 0
    n_total = len(bundle.train_df) + len(bundle.val_df) + len(bundle.test_df)
    assert n_total == 12
    batch = next(iter(bundle.train_loader))
    assert batch["image"].shape[1:] == (3, 224, 224)
    assert batch["labels"].shape[1] == len(bundle.genre_vocab)
    assert batch["image"].shape[0] == batch["labels"].shape[0]


# ---------- WeightedRandomSampler ----------


def test_balanced_sampler_evens_out_imbalanced_sources():
    # 90 books, 10 movies. Without weighting, books would dominate ~9:1.
    sources = ["book"] * 90 + ["movie"] * 10
    sampler = build_source_balanced_sampler(
        sources, generator=torch.Generator().manual_seed(0)
    )
    drawn = [sources[i] for i in list(sampler)]
    n_movie = drawn.count("movie")
    n_book = drawn.count("book")
    # Expected ~50/50; allow generous tolerance for the small N.
    assert 30 <= n_movie <= 70, f"movie count {n_movie} far from balanced"
    assert 30 <= n_book <= 70, f"book count {n_book} far from balanced"


def test_make_data_bundle_uses_balanced_sampler(tmp_path: Path):
    root = _make_synthetic_data(tmp_path, n_movies=2, n_books=10)
    bundle = make_data_bundle(
        processed_dir=root / "data" / "processed",
        data_root=root,
        tokenizer=_FakeTokenizer(),
        batch_size=4,
        num_workers=0,
        val_frac=0.0,
        test_frac=0.0,
        seed=0,
        min_genre_count=1,
        pin_memory=False,
        balance_train_sources=True,
    )
    seen = []
    for batch in bundle.train_loader:
        # When sampler is set, shuffle is False but the sampler does the work.
        # We just need the loader to actually iterate.
        seen.append(batch["image"].shape[0])
    assert sum(seen) > 0, "balanced loader produced no batches"
