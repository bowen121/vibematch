"""One-command data pipeline: Kaggle download -> URL download -> preprocess.

Each sub-step is idempotent, so re-running this is safe and skips work
already done. The expensive step is poster URL fetching (~5-30 min depending
on success rate); pass --skip-posters to use only the ~1k Kaggle samples.

Usage:
    python scripts/prepare_data.py
    python scripts/prepare_data.py --skip-posters       # quick path, ~1k movies
    python scripts/prepare_data.py --poster-limit 8000  # cap fetch at 8k URLs
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = REPO_ROOT / "scripts"


def run(cmd: list[str]) -> None:
    """Run a sub-script; abort on non-zero exit."""
    print(f"\n$ {' '.join(cmd)}", flush=True)
    result = subprocess.run(cmd, cwd=REPO_ROOT)
    if result.returncode != 0:
        sys.exit(f"\n[prepare_data] step failed: {' '.join(cmd)}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--skip-posters", action="store_true",
                        help="don't fetch additional posters from URLs")
    parser.add_argument("--poster-limit", type=int, default=12000,
                        help="cap on poster URLs to attempt (default 12000)")
    parser.add_argument("--poster-workers", type=int, default=16)
    args = parser.parse_args()

    py = sys.executable

    print("[1/3] Kaggle dataset download")
    run([py, str(SCRIPTS / "download_data.py")])

    if not args.skip_posters:
        print("\n[2/3] Movie poster URL fetch (this is the slow step)")
        run([
            py, str(SCRIPTS / "download_posters.py"),
            "--workers", str(args.poster_workers),
            "--limit", str(args.poster_limit),
        ])
    else:
        print("\n[2/3] Movie poster URL fetch — SKIPPED (--skip-posters)")

    print("\n[3/3] Preprocess raw -> processed CSVs")
    run([py, str(SCRIPTS / "preprocess.py")])

    print("\n[done] data/processed/{movies,books}.csv ready.")
    print("       Next: open notebooks/exploration.ipynb")


if __name__ == "__main__":
    main()
