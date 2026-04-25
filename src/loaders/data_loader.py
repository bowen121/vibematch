"""DataLoader factory: turn processed CSVs into train/val/test loaders.

Public entry point is `make_data_bundle(config)` → DataBundle. Member B's CLIP
trainer and Member C's classifier trainer both consume this same bundle.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import pandas as pd
from torch.utils.data import DataLoader

from src.loaders.dataset import MediaDataset, parse_genres
from src.loaders.split import build_genre_vocab, genre_stratified_split

PROCESSED_CSVS = ("movies.csv", "books.csv")


@dataclass
class DataBundle:
    """Bundles every artifact downstream training code needs."""
    train_loader: DataLoader
    val_loader: DataLoader
    test_loader: DataLoader
    genre_vocab: list[str]
    train_df: pd.DataFrame
    val_df: pd.DataFrame
    test_df: pd.DataFrame


def load_processed_frame(processed_dir: str | Path) -> pd.DataFrame:
    """Concat movies.csv and books.csv, dropping rows whose images are missing."""
    processed_dir = Path(processed_dir)
    frames: list[pd.DataFrame] = []
    for fname in PROCESSED_CSVS:
        path = processed_dir / fname
        if not path.exists():
            raise FileNotFoundError(
                f"{path} not found. Run scripts/preprocess.py first."
            )
        frames.append(pd.read_csv(path))
    df = pd.concat(frames, ignore_index=True)
    if df.empty:
        raise ValueError("Processed CSVs are empty.")
    return df


def make_data_bundle(
    processed_dir: str | Path,
    data_root: str | Path,
    tokenizer: Any,
    *,
    batch_size: int = 64,
    num_workers: int = 2,
    val_frac: float = 0.1,
    test_frac: float = 0.1,
    seed: int = 42,
    min_genre_count: int = 5,
    pin_memory: bool = True,
) -> DataBundle:
    """Build the train/val/test loaders + genre vocabulary in one call.

    The split happens here (not at preprocess time) so seed changes don't
    require re-running preprocess.
    """
    df = load_processed_frame(processed_dir)
    genres_per_item = [parse_genres(g) for g in df["genres"]]
    vocab = build_genre_vocab(genres_per_item, min_count=min_genre_count)
    if not vocab:
        raise ValueError(
            f"No genres met min_count={min_genre_count}. "
            f"Lower the threshold or check preprocess output."
        )

    split = genre_stratified_split(
        genres_per_item, val_frac=val_frac, test_frac=test_frac, seed=seed
    )
    train_df = df.iloc[split.train].reset_index(drop=True)
    val_df = df.iloc[split.val].reset_index(drop=True)
    test_df = df.iloc[split.test].reset_index(drop=True)

    def _make(loader_df: pd.DataFrame, train: bool, shuffle: bool) -> DataLoader:
        ds = MediaDataset(
            loader_df,
            data_root=data_root,
            genre_vocab=vocab,
            tokenizer=tokenizer,
            train=train,
            seed=seed,
        )
        return DataLoader(
            ds,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=pin_memory,
            drop_last=shuffle,
        )

    return DataBundle(
        train_loader=_make(train_df, train=True, shuffle=True),
        val_loader=_make(val_df, train=False, shuffle=False),
        test_loader=_make(test_df, train=False, shuffle=False),
        genre_vocab=vocab,
        train_df=train_df,
        val_df=val_df,
        test_df=test_df,
    )
