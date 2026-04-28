"""Parallel downloader for movie poster images referenced in MovieGenre.csv.

The Kaggle `movie-genre-from-its-poster` dump only ships ~1k sample images.
The CSV has ~39k Amazon-hosted URLs. This script fetches the rest.

Resumes on rerun (skips files already on disk). Failures are silent — the
exact set of working URLs varies (Amazon link rot since 2017).

Usage:
    python scripts/download_posters.py                      # all URLs
    python scripts/download_posters.py --limit 8000         # first 8000 only
    python scripts/download_posters.py --workers 16
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import requests

REPO_ROOT = Path(__file__).resolve().parents[1]
CSV_PATH = REPO_ROOT / "data" / "raw" / "movies" / "MovieGenre.csv"
OUT_DIR = REPO_ROOT / "data" / "raw" / "movies" / "SampleMoviePosters" / "SampleMoviePosters"

TIMEOUT_S = 6
HEADERS = {
    "User-Agent": "Mozilla/5.0 (compatible; VibeMatch coursework downloader)"
}


def _read_targets(csv_path: Path) -> list[tuple[str, str]]:
    """Yield (imdbId, url) pairs from MovieGenre.csv. Skips rows missing either."""
    targets: list[tuple[str, str]] = []
    with open(csv_path, encoding="latin-1", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            imdb = (row.get("imdbId") or "").strip()
            url = (row.get("Poster") or "").strip()
            if imdb and url and url.startswith("http"):
                targets.append((imdb, url))
    return targets


def _download_one(imdb: str, url: str, out_dir: Path) -> str:
    """Returns one of: 'ok', 'skip', 'fail'."""
    out_path = out_dir / f"{imdb}.jpg"
    if out_path.exists() and out_path.stat().st_size > 0:
        return "skip"
    try:
        r = requests.get(url, timeout=TIMEOUT_S, headers=HEADERS, stream=True)
        if r.status_code != 200:
            return "fail"
        content = r.content
        if len(content) < 1024:
            return "fail"
        out_path.write_bytes(content)
        return "ok"
    except requests.RequestException:
        return "fail"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument("--limit", type=int, default=None,
                        help="cap the number of URLs attempted (after resume-skip)")
    args = parser.parse_args()

    if not CSV_PATH.exists():
        sys.exit(f"{CSV_PATH} not found. Run scripts/download_data.py first.")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    targets = _read_targets(CSV_PATH)
    if args.limit is not None:
        targets = targets[: args.limit]
    print(f"[start] {len(targets)} URLs, {args.workers} workers, out={OUT_DIR}")

    counts = {"ok": 0, "skip": 0, "fail": 0}
    t0 = time.monotonic()
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(_download_one, imdb, url, OUT_DIR) for imdb, url in targets]
        for i, fut in enumerate(as_completed(futures), 1):
            counts[fut.result()] += 1
            if i % 500 == 0 or i == len(futures):
                dt = time.monotonic() - t0
                rate = i / dt if dt > 0 else 0
                print(
                    f"[{i:5d}/{len(futures)}] ok={counts['ok']} "
                    f"skip={counts['skip']} fail={counts['fail']} "
                    f"({rate:.1f}/s)"
                )

    print(f"[done] {counts}")


if __name__ == "__main__":
    main()
