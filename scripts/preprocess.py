"""Convert raw Kaggle dumps in data/raw/ into canonical CSVs in data/processed/.

Output schema (one row = one media item):
    id, image_path, title, genres, source

Run after `scripts/download_data.py`. Idempotent — overwrites processed CSVs.

Schema-detection: each Kaggle dump has slightly different column names across
versions, so we look for any column whose lowercased name contains a known
keyword (e.g., "genre", "title"). If detection fails, the script raises with a
clear message listing the columns it actually saw.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = REPO_ROOT / "data" / "raw"
PROCESSED_DIR = REPO_ROOT / "data" / "processed"

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp"}


def _find_column(df: pd.DataFrame, keywords: list[str], required: bool = True) -> str | None:
    """Return the first column whose lowercased name contains any keyword."""
    lower_map = {c.lower(): c for c in df.columns}
    for kw in keywords:
        for low, original in lower_map.items():
            if kw in low:
                return original
    if required:
        raise KeyError(
            f"None of {keywords} found in columns {list(df.columns)}. "
            f"Inspect the raw CSV and update preprocess.py keywords."
        )
    return None


def _index_images(root: Path) -> dict[str, Path]:
    """Map filename stem (no ext) → absolute image path. Used to attach images to rows."""
    out: dict[str, Path] = {}
    if not root.exists():
        return out
    for p in root.rglob("*"):
        if p.is_file() and p.suffix.lower() in IMAGE_EXTS:
            out.setdefault(p.stem, p)
    return out


def _normalise_genres(cell: object) -> str:
    """Whatever the source uses (comma, pipe, list-string), output 'Drama|Romance'."""
    if cell is None:
        return ""
    if isinstance(cell, float) and pd.isna(cell):
        return ""
    s = str(cell).strip()
    if not s:
        return ""
    for sep in ("|", ",", ";", "/"):
        if sep in s:
            parts = s.split(sep)
            break
    else:
        parts = [s]
    return "|".join(p.strip() for p in parts if p.strip())


def _first_csv(directory: Path) -> Path:
    """Return the first .csv (case-insensitive) under `directory`, recursively."""
    for p in sorted(directory.rglob("*")):
        if p.is_file() and p.suffix.lower() == ".csv":
            return p
    raise FileNotFoundError(f"No CSV found under {directory}.")


def preprocess_movies() -> pd.DataFrame:
    """Build the canonical movies frame from data/raw/movies/."""
    raw_root = RAW_DIR / "movies"
    csv_path = _first_csv(raw_root)
    df = pd.read_csv(csv_path, encoding_errors="ignore")

    id_col = _find_column(df, ["imdbid", "id"])
    title_col = _find_column(df, ["title"])
    genre_col = _find_column(df, ["genre"])

    images = _index_images(raw_root)
    rows = []
    for _, r in df.iterrows():
        raw_id = str(r[id_col]).strip()
        if not raw_id or raw_id.lower() == "nan":
            continue
        img = images.get(raw_id) or images.get(f"tt{raw_id}") or images.get(raw_id.lstrip("0"))
        if img is None:
            continue
        rows.append({
            "id": f"movie_{raw_id}",
            "image_path": str(img.relative_to(REPO_ROOT)),
            "title": str(r[title_col]) if pd.notna(r[title_col]) else "",
            "genres": _normalise_genres(r[genre_col]),
            "source": "movie",
        })
    out = pd.DataFrame(rows)
    out = out[out["genres"].str.len() > 0].reset_index(drop=True)
    return out


def preprocess_books() -> pd.DataFrame:
    """Build the canonical books frame from data/raw/books/."""
    raw_root = RAW_DIR / "books"
    csv_path = _first_csv(raw_root)
    df = pd.read_csv(csv_path, encoding_errors="ignore")

    id_col = _find_column(df, ["isbn", "id"], required=False)
    title_col = _find_column(df, ["title", "name"])
    genre_col = _find_column(df, ["genre", "category", "categories"])
    image_col = _find_column(df, ["image", "cover", "filename", "file_name"], required=False)

    images = _index_images(raw_root)
    rows = []
    for i, r in df.iterrows():
        if id_col and pd.notna(r[id_col]):
            raw_id = str(r[id_col]).strip()
        else:
            raw_id = str(i)
        img = None
        if image_col and pd.notna(r[image_col]):
            cand = str(r[image_col]).strip()
            stem = Path(cand).stem
            img = images.get(stem)
            if img is None and (raw_root / cand).exists():
                img = raw_root / cand
        if img is None:
            img = images.get(raw_id)
        if img is None:
            continue
        rows.append({
            "id": f"book_{raw_id}",
            "image_path": str(img.relative_to(REPO_ROOT)),
            "title": str(r[title_col]) if pd.notna(r[title_col]) else "",
            "genres": _normalise_genres(r[genre_col]),
            "source": "book",
        })
    out = pd.DataFrame(rows)
    out = out[out["genres"].str.len() > 0].reset_index(drop=True)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--movies-only", action="store_true")
    parser.add_argument("--books-only", action="store_true")
    args = parser.parse_args()

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    do_movies = not args.books_only
    do_books = not args.movies_only

    if do_movies:
        try:
            movies = preprocess_movies()
        except FileNotFoundError as e:
            print(f"[skip movies] {e}", file=sys.stderr)
        else:
            out = PROCESSED_DIR / "movies.csv"
            movies.to_csv(out, index=False)
            print(f"[movies] {len(movies)} rows -> {out}")

    if do_books:
        try:
            books = preprocess_books()
        except FileNotFoundError as e:
            print(f"[skip books] {e}", file=sys.stderr)
        else:
            out = PROCESSED_DIR / "books.csv"
            books.to_csv(out, index=False)
            print(f"[books] {len(books)} rows -> {out}")


if __name__ == "__main__":
    main()
