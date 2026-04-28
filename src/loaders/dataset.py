"""Dataset classes and image/text transforms for VibeMatch.

Canonical processed-CSV schema (one row = one media item):
    id          str   stable id (imdbId, ISBN, or hash)
    image_path  str   path to the image, relative to data_root
    title       str   item title
    genres      str   pipe-separated list, e.g. "Drama|Romance"
    source      str   "movie" or "book"

`MediaDataset` returns a dict so downstream code can pull whichever fields it
needs without unpacking tuples. Member B (contrastive) consumes image +
input_ids/attention_mask; Member C (classifier) consumes image + labels.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

from src.loaders.split import multi_hot

IMAGE_SIZE = 224
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

PROMPT_TEMPLATES = {
    "movie": "a movie poster of a {genre} film",
    "book": "a book cover for a {genre} book",
}


def build_image_transform(train: bool) -> transforms.Compose:
    """Standard ResNet-style preprocessing. Light augmentation in train mode."""
    if train:
        return transforms.Compose([
            transforms.Resize(256),
            transforms.RandomResizedCrop(IMAGE_SIZE, scale=(0.85, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ])
    return transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(IMAGE_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


def render_prompt(genres: Sequence[str], source: str, rng: np.random.Generator | None) -> str:
    """Pick one genre and render it through the source-specific template.

    rng=None → deterministic (first listed genre); used for val/test.
    """
    template = PROMPT_TEMPLATES.get(source, "a {genre} cover")
    if not genres:
        return template.format(genre="generic")
    if rng is None:
        chosen = genres[0]
    else:
        chosen = genres[int(rng.integers(0, len(genres)))]
    return template.format(genre=chosen.lower())


def parse_genres(cell: Any) -> list[str]:
    """Split a pipe-separated genres cell into a clean list. Handles NaN."""
    if cell is None or (isinstance(cell, float) and np.isnan(cell)):
        return []
    if isinstance(cell, list):
        return [g.strip() for g in cell if str(g).strip()]
    return [g.strip() for g in str(cell).split("|") if g.strip()]


@dataclass
class MediaSample:
    """One item returned by MediaDataset.__getitem__ (kept as dict for collate)."""
    image: torch.Tensor
    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    labels: torch.Tensor
    source: str
    item_id: str


class MediaDataset(Dataset):
    """Unified dataset over the processed movies+books CSVs.

    Args:
        df:           DataFrame with the canonical schema above (already split).
        data_root:    directory that image_path is relative to.
        genre_vocab:  ordered list of genres → multi-hot indices.
        tokenizer:    HuggingFace tokenizer (e.g. DistilBertTokenizerFast).
        train:        if True, use augmenting transform + random prompt genre.
        max_text_len: max token length passed to tokenizer.
        seed:         RNG seed for reproducible prompt sampling in train mode.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        data_root: str | Path,
        genre_vocab: Sequence[str],
        tokenizer: Any,
        train: bool = False,
        max_text_len: int = 77,
        seed: int = 0,
    ) -> None:
        required_cols = {"id", "image_path", "genres", "source"}
        missing = required_cols - set(df.columns)
        if missing:
            raise ValueError(f"DataFrame missing required columns: {sorted(missing)}")
        self.df = df.reset_index(drop=True)
        self.data_root = Path(data_root)
        self.genre_vocab = list(genre_vocab)
        self.tokenizer = tokenizer
        self.train = train
        self.max_text_len = max_text_len
        self.transform = build_image_transform(train=train)
        self._rng = np.random.default_rng(seed) if train else None

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        row = self.df.iloc[idx]
        genres = parse_genres(row["genres"])

        img_path = self.data_root / str(row["image_path"])
        with Image.open(img_path) as img:
            image = self.transform(img.convert("RGB"))

        caption_cols = [c for c in ("caption_1", "caption_2", "caption_3") if c in self.df.columns]
        available = [str(row[c]) for c in caption_cols if pd.notna(row[c]) and str(row[c]).strip()]
        if available:
            chosen = available[int(self._rng.integers(0, len(available)))] if self._rng else available[0]
            prompt = chosen
        else:
            prompt = render_prompt(genres, str(row["source"]), self._rng)
        encoded = self.tokenizer(
            prompt,
            padding="max_length",
            truncation=True,
            max_length=self.max_text_len,
            return_tensors="pt",
        )

        labels = torch.from_numpy(multi_hot(genres, self.genre_vocab))

        return {
            "image": image,
            "input_ids": encoded["input_ids"].squeeze(0),
            "attention_mask": encoded["attention_mask"].squeeze(0),
            "labels": labels,
            "source": str(row["source"]),
            "item_id": str(row["id"]),
        }
