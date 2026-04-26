# VibeMatch

Aesthetic-driven media retrieval system that matches movie/book vibes using a custom-built CLIP framework with contrastive learning, cosine similarity search, and a live MLP genre classifier.

## Setup

```bash
pip install -r requirements.txt
```

## Dataset setup

VibeMatch uses two Kaggle datasets:

- [Movie Posters](https://www.kaggle.com/datasets/neha1703/movie-genre-from-its-poster) (~7k images)
- [Book Covers](https://www.kaggle.com/datasets/mexwell/book-cover-dataset) (~20k images)

**1. Get a Kaggle API token** — go to https://www.kaggle.com/settings/account. Two options:

- **(Recommended) New API Token** → click "Generate New Token", copy the `KGAT_…` string, then put it in `.env`:
  ```bash
  cp .env.example .env  # then edit .env and set KAGGLE_API_TOKEN=KGAT_...
  ```
  Requires `kaggle>=1.8.0` (already pinned in `requirements.txt`).
- **(Legacy) kaggle.json** → click "Create Legacy API Key", then:
  ```bash
  mkdir -p ~/.kaggle
  mv ~/Downloads/kaggle.json ~/.kaggle/
  chmod 600 ~/.kaggle/kaggle.json
  ```

**2. Accept dataset rules** — open each dataset page above in a browser (while signed in) and click *"I Understand and Accept"*. Without this, downloads return 403.

**3. One-command pipeline** — runs Kaggle fetch + poster URL download + preprocess in order, idempotent:

```bash
python scripts/prepare_data.py
# faster path skipping the URL fetch (only ~1k movies):
python scripts/prepare_data.py --skip-posters
```

This produces `data/processed/{movies,books}.csv` with a unified schema (`id, image_path, title, genres, source`) and is what Member B and Member C depend on.

**4. EDA** — open `notebooks/exploration.ipynb` for genre distribution, image-size stats, sample grids, and split balance.

**Sub-scripts (run individually if you want):**

```bash
python scripts/download_data.py            # Kaggle datasets
python scripts/download_posters.py         # ~39k poster URLs from MovieGenre.csv
python scripts/preprocess.py               # raw -> canonical CSVs
```

For Part A details (DataBundle API, sampler, schema), see [`docs/part_a.md`](docs/part_a.md).

## Project Structure

```
vibematch/
|-- README.md                  # Project overview, setup, and usage instructions
|-- requirements.txt           # Python dependencies
|-- app.py                     # Streamlit entry point: launches the web app
|-- configs/
|   |-- train_config.yaml      # Training hyperparameters
|   +-- app_config.yaml        # App settings (top-k, index path, model weight path)
|-- scripts/
|   |-- train_clip.py          # Trains projection layers with contrastive loss
|   |-- train_classifier.py    # Trains the MLP genre classifier on frozen embeddings
|   +-- build_index.py         # Generates the FAISS index from image embeddings
|-- src/
|   |-- loaders/
|   |   |-- dataset.py         # Dataset classes and image/text transforms
|   |   |-- data_loader.py     # DataLoader factory with batching and sampling logic
|   |   +-- split.py           # Genre-stratified train/val/test splitting logic
|   |-- model/
|   |   |-- encoder.py         # Frozen ResNet + DistilBERT w/ projection layers
|   |   |-- classifier.py      # MLP genre classifier for live genre tagging
|   |   +-- loss.py            # Symmetric contrastive loss function
|   +-- retrieval/
|       |-- engine.py          # FAISS index building and saving
|       +-- search.py          # Cosine-similarity query and top-k retrieval
|-- data/
|   |-- raw/                   # Original downloaded datasets (posters, covers)
|   +-- processed/             # Cleaned images and metadata CSVs
|-- models/                    # Saved checkpoints (.pt) and FAISS index (.bin)
|-- notebooks/
|   +-- exploration.ipynb      # EDA: dataset stats, sample visualizations
+-- tests/
    +-- test_pipeline.py       # Smoke tests for data loading, encoding, and retrieval
```

## Usage

```bash
# Train CLIP projection layers
python scripts/train_clip.py

# Train MLP genre classifier
python scripts/train_classifier.py

# Build FAISS index
python scripts/build_index.py

# Launch the app
streamlit run app.py
```