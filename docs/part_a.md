# Part A — Data & Evaluation

Owner: **Ethan Pan**.

This module produces the train/val/test data that Members B (CLIP) and C
(classifier) consume. Everything below is reproducible from a fresh clone with
one command (`scripts/prepare_data.py`).

---

## TL;DR — get a working dataset

```bash
pip install -r requirements.txt

# one-time: put your Kaggle token in .env
cp .env.example .env
# edit .env -> KAGGLE_API_TOKEN=KGAT_...   (or use legacy kaggle.json, see README)

# accept dataset rules in a browser (signed in):
#   https://www.kaggle.com/datasets/neha1703/movie-genre-from-its-poster
#   https://www.kaggle.com/datasets/mexwell/book-cover-dataset

python scripts/prepare_data.py            # ~30 min, mostly the poster fetch
```

After this, `data/processed/movies.csv` and `data/processed/books.csv` exist
and the EDA notebook at `notebooks/exploration.ipynb` runs end-to-end.

---

## Pipeline stages

```
scripts/prepare_data.py
 ├── scripts/download_data.py     [Kaggle API → data/raw/]
 ├── scripts/download_posters.py  [HTTP from URLs in MovieGenre.csv → data/raw/movies/.../*.jpg]
 └── scripts/preprocess.py        [data/raw/ → data/processed/{movies,books}.csv]
```

Each stage is **idempotent** — re-running skips work that's already done. You
can also run them individually:

```bash
python scripts/download_data.py          # both
python scripts/download_data.py --movies # just one
python scripts/download_posters.py --workers 16 --limit 12000
python scripts/preprocess.py
```

---

## Processed CSV schema

`data/processed/movies.csv` and `data/processed/books.csv` share one schema:

| column        | type   | description                                    |
|---------------|--------|------------------------------------------------|
| `id`          | str    | stable id, prefixed `movie_<imdbId>` / `book_<asin>` |
| `image_path`  | str    | path relative to repo root                     |
| `title`       | str    | item title (free-form)                         |
| `genres`      | str    | pipe-separated, e.g. `"Drama|Romance"`         |
| `source`      | str    | `"movie"` or `"book"`                          |

Rows with no local image or empty genres are dropped at preprocess time.

---

## DataBundle API — for Members B and C

This is the **only** thing downstream code needs to import.

```python
from src.loaders import make_data_bundle
from transformers import DistilBertTokenizerFast

tokenizer = DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased")

bundle = make_data_bundle(
    processed_dir="data/processed",
    data_root=".",                 # image_path is relative to this
    tokenizer=tokenizer,
    batch_size=64,
    num_workers=4,
    seed=42,
    balance_train_sources=True,    # see "Class imbalance" below
)

bundle.train_loader   # torch.utils.data.DataLoader
bundle.val_loader
bundle.test_loader
bundle.genre_vocab    # list[str], multi-hot label index = vocab.index(genre)
```

Each batch is a `dict` with these keys:

| key              | shape                | dtype       | notes                              |
|------------------|----------------------|-------------|------------------------------------|
| `image`          | `(B, 3, 224, 224)`   | float32     | ImageNet-normalized                |
| `input_ids`      | `(B, 32)`            | long        | DistilBERT-tokenized prompt        |
| `attention_mask` | `(B, 32)`            | long        |                                    |
| `labels`         | `(B, |vocab|)`       | float32     | multi-hot genre vector             |
| `source`         | list[str], len B     | —           | `"movie"` / `"book"`               |
| `item_id`        | list[str], len B     | —           | for retrieval-eval traceback       |

The text in `input_ids` is a templated genre prompt, not the title. In train
mode a random genre is sampled per call; in val/test the first listed genre is
used (deterministic).

---

## Class imbalance — important for Member B

Raw split sizes (with movie poster URL download):

| split   | movies  | books   | ratio   |
|---------|---------|---------|---------|
| train   | ~6,000  | ~45,600 | ~1 : 8  |
| val     |   ~750  |  ~5,700 | ~1 : 8  |
| test    |   ~750  |  ~5,700 | ~1 : 8  |

Without intervention the contrastive trainer almost only sees book/text
pairs and converges to "any cover ≈ a book." Two recommended fixes:

1. **Use `balance_train_sources=True`** when calling `make_data_bundle`.
   Internally this swaps the train DataLoader's shuffler for a
   `WeightedRandomSampler` that draws movies and books with equal expected
   frequency. Implemented in `build_source_balanced_sampler` in
   `src/loaders/data_loader.py`. `replacement=True`, so movies are seen
   ~8× per epoch — the trainer sees more poster aesthetics per step.

2. **Lower `min_genre_count`** if you want rarer movie genres in the vocab.
   The default is `5`; with the smaller movie set you may want `min_genre_count=3`.

---

## Genre-stratified split — what's special

Plain `train_test_split` with shuffle would put all 5 "Western" rows into one
split. Our split (`src/loaders/split.py:genre_stratified_split`) keys each
row by its **rarest** genre and stratifies on that, so small-genre
representation is preserved across train/val/test. This is what makes the
"generalize to unseen visual styles" eval meaningful.

The split is deterministic given a seed and is regenerated per run from the
processed CSVs — no `splits/` files on disk to drift out of sync.

---

## Known limitations

- **Movie poster URL rot.** ~30% of the Amazon URLs in the Kaggle CSV are
  dead (they're from 2017). `download_posters.py` silently skips failures.
  Expect ~5,000–8,000 movie images successfully downloaded out of ~39,000
  rows. Plenty for the proposal's "~7,000" target.
- **Books are single-label.** The mexwell dataset has one category per book,
  so book labels look multi-hot but only one bit is set. Member C's
  classifier should still treat labels as multi-label (BCE) since movies
  are genuinely multi-label.
- **Sample posters folder is nested.** Kaggle ships the images at
  `data/raw/movies/SampleMoviePosters/SampleMoviePosters/*.jpg` (yes, the
  folder name is duplicated). We write our own downloads into the same
  inner directory.
- **Books CSV is semicolon-delimited.** Handled in `preprocess.py`. If
  Kaggle ever changes the format you'll see a column-detection error
  pointing you to update the keyword list.

---

## Tests

```bash
pytest tests/test_loaders.py -v
```

18 tests covering: split determinism, no-leak, rare-genre survival, vocab
construction, multi-hot encoding, prompt rendering, image transform shapes,
end-to-end DataBundle on synthetic fixtures, and the balanced sampler.

All synthetic — they run without the Kaggle data.
