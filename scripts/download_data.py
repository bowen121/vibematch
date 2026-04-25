"""Download the Kaggle Movie Poster + Goodreads Book Cover datasets into data/raw/.

Usage:
    python scripts/download_data.py              # download both
    python scripts/download_data.py --movies     # download only movies
    python scripts/download_data.py --books      # download only books
    python scripts/download_data.py --force      # re-download even if present

Auth: requires either ~/.kaggle/kaggle.json (chmod 600) or KAGGLE_USERNAME +
KAGGLE_KEY env vars. See https://www.kaggle.com/docs/api for setup.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = REPO_ROOT / "data" / "raw"


def _load_dotenv() -> None:
    """Read REPO_ROOT/.env into os.environ. Existing env vars win. No-op if missing."""
    env_path = REPO_ROOT / ".env"
    if not env_path.exists():
        return
    for line in env_path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        os.environ.setdefault(key, value)


_load_dotenv()

DATASETS = {
    "movies": {
        "slug": "neha1703/movie-genre-from-its-poster",
        "subdir": "movies",
        "url": "https://www.kaggle.com/datasets/neha1703/movie-genre-from-its-poster",
    },
    "books": {
        "slug": "mexwell/book-cover-dataset",
        "subdir": "books",
        "url": "https://www.kaggle.com/datasets/mexwell/book-cover-dataset",
    },
}


def _check_credentials() -> None:
    """Fail fast with an actionable message if Kaggle credentials are missing."""
    username = os.environ.get("KAGGLE_USERNAME", "").strip()
    key = os.environ.get("KAGGLE_KEY", "").strip()
    has_env = bool(username) and bool(key)
    has_file = (Path.home() / ".kaggle" / "kaggle.json").exists()
    if has_env or has_file:
        return
    missing = []
    if not username:
        missing.append("KAGGLE_USERNAME")
    if not key:
        missing.append("KAGGLE_KEY")
    sys.exit(
        f"Kaggle credentials not found (missing: {', '.join(missing) or 'all'}).\n"
        "Either:\n"
        "  (a) fill both KAGGLE_USERNAME and KAGGLE_KEY in .env (see .env.example), or\n"
        "  (b) place kaggle.json from https://www.kaggle.com/settings/account at\n"
        "      ~/.kaggle/kaggle.json (chmod 600).\n"
        "Both username and key are required — a key alone will not authenticate."
    )


def _is_already_downloaded(target: Path) -> bool:
    """Treat the dataset as present if the directory exists and has any non-zip files."""
    if not target.exists():
        return False
    return any(p.suffix.lower() != ".zip" for p in target.rglob("*") if p.is_file())


def download_one(name: str, force: bool) -> None:
    """Download and unzip a single dataset by short name (movies | books)."""
    spec = DATASETS[name]
    target = RAW_DIR / spec["subdir"]
    target.mkdir(parents=True, exist_ok=True)

    if not force and _is_already_downloaded(target):
        print(f"[skip] {name}: already present at {target}")
        return

    # Import lazily so --help works without the kaggle package installed.
    from kaggle.api.kaggle_api_extended import KaggleApi

    api = KaggleApi()
    api.authenticate()

    print(f"[download] {name}: {spec['slug']} -> {target}")
    print(f"           if this 403s, accept the dataset rules at {spec['url']}")
    api.dataset_download_files(spec["slug"], path=str(target), unzip=True, quiet=False)
    print(f"[done] {name}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--movies", action="store_true", help="download movie posters only")
    parser.add_argument("--books", action="store_true", help="download book covers only")
    parser.add_argument("--force", action="store_true", help="re-download even if present")
    args = parser.parse_args()

    _check_credentials()

    selected: list[str] = []
    if args.movies:
        selected.append("movies")
    if args.books:
        selected.append("books")
    if not selected:
        selected = list(DATASETS.keys())

    for name in selected:
        download_one(name, force=args.force)


if __name__ == "__main__":
    main()
