# VibeMatch

Aesthetic-driven media retrieval system that matches movie/book vibes using a custom-built CLIP framework with contrastive learning, cosine similarity search, and a live MLP genre classifier.

Describe a vibe or upload an image, and VibeMatch returns the 12 most aesthetically similar movies and books from a 66,604-item index. Results are ranked by cosine similarity in a shared 256-dimensional embedding space trained jointly on visual and textual signals.

## Live Demo

**[vibematch-xprf2kfhsbfjz8ecyyugvd.streamlit.app](https://vibematch-xprf2kfhsbfjz8ecyyugvd.streamlit.app)**

## How It Works

```
Query (text or image)
        ↓
  DistilBERT / ResNet-50 backbone  (frozen)
        ↓
  Learned projection head  →  256-d L2-normalized embedding
        ↓
  FAISS IndexFlatIP
        ↓
  Top-12 results  +  live MLP genre tags
```

1. **Text query** — tokenized with DistilBERT, CLS token projected to 256-d space.
2. **Image query** — resized to 224×224, passed through ResNet-50 avgpool, projected to the same 256-d space.
3. **Retrieval** — inner product search over a pre-built FAISS flat index. Because all embeddings are L2-normalized, inner product equals cosine similarity.
4. **Genre tagging** — a lightweight MLP classifier (trained on image embeddings) runs on the query embedding first, predicting "you might be looking for: 82% Thriller, 61% Crime" to demonstrate cross-modal alignment. It also annotates each retrieved result independently.

## Model Architecture

### Dual Encoder (CLIP-style)

| Component | Backbone | Output dim | Trainable |
|---|---|---|---|
| Image encoder | ResNet-50 (ImageNet pretrained) | 2048 → 256 | Projection head only |
| Text encoder | DistilBERT-base-uncased | 768 → 256 | Projection head only |

Both projection heads share the same structure: `Linear → LayerNorm → GELU → Dropout → Linear`. Only the projection heads are trained; backbone weights are frozen throughout.

**Loss:** symmetric InfoNCE contrastive loss over (image, caption) pairs within each batch. Positive pairs are matched (image, caption); all other combinations in the batch are negatives. Temperature is fixed at τ=0.07 — learnable temperature was found to collapse from 0.07 → 0.027 in every run, causing overconfident gradients and val divergence.

**Best checkpoint:** `models/runs/20260428_050849/best.pt` — val_loss=3.2685, R@1=5.3%, R@10=21.3%, med_rank=84, align=0.438

### Genre Classifier

A 4-layer MLP (`256 → 512 → 256 → num_genres`) trained on frozen image embeddings. Because image and text projections share the same 256-d space, the classifier also works on text query embeddings at inference — proving cross-modal alignment. Used for the "you might be looking for" readout on each query and for annotating retrieved results.

## Features

- **Text search** — describe any mood, aesthetic, era, or feeling
- **Image search** — upload a still, poster, or photo and find visually similar media
- **Cross-modal retrieval** — text queries match books and movies in the same embedding space
- **Live query genre prediction** — MLP trained on image embeddings predicts genres from text queries in real time, proving the learned space aligns both modalities ("you might be looking for: 82% Thriller")
- **Live result genre tagging** — genres predicted per result via the same MLP classifier
- **Dominant-color theming** — each result card is tinted by the poster's extracted dominant color
- **High-res posters** — CDN images for all 66k items (no local image files required at runtime)
- **Interactive UI** — particle canvas background with cursor tracking, built in Streamlit with a custom HTML/JS search component

## Setup

```bash
pip install -r requirements.txt
```

## Dataset setup

VibeMatch uses two Kaggle datasets:

- [Movie Posters](https://www.kaggle.com/datasets/neha1703/movie-genre-from-its-poster) (~9k images)
- [Book Covers](https://www.kaggle.com/datasets/mexwell/book-cover-dataset) (~57k images)

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
|-- README.md
|-- requirements.txt
|-- app.py                         # Streamlit entry point
|-- configs/
|   |-- train_config.yaml          # Hyperparameters for training scripts
|   +-- app_config.yaml            # App settings (top-k, index path, weight paths)
|-- scripts/
|   |-- prepare_data.py            # End-to-end data pipeline (idempotent)
|   |-- train_clip.py              # Trains projection heads with contrastive loss
|   |-- train_classifier.py        # Trains MLP genre classifier on frozen embeddings
|   |-- extract_embeddings.py      # Saves (N, 256) embeddings.pt for classifier training
|   |-- build_index.py             # Builds FAISS index + dominant-color metadata
|   |-- generate_descriptions.py   # Qwen2.5-VL caption generation (parallel splits)
|   |-- download_data.py           # Kaggle dataset fetch
|   |-- download_posters.py        # Poster CDN URL scrape
|   +-- preprocess.py              # Raw → canonical CSV
|-- src/
|   |-- loaders/
|   |   |-- dataset.py             # MediaDataset + image/text transforms
|   |   |-- data_loader.py         # DataLoader factory with WeightedRandomSampler
|   |   +-- split.py               # Genre-stratified train/val/test splitting
|   |-- model/
|   |   |-- encoder.py             # Frozen ResNet + DistilBERT w/ projection heads
|   |   |-- classifier.py          # MLP genre classifier + predict_genres_with_scores()
|   |   +-- loss.py                # Symmetric InfoNCE contrastive loss
|   +-- retrieval/
|       |-- engine.py              # FAISS index building, normalisation, save/load
|       +-- search.py              # Cosine-similarity query + top-k SearchResult
|-- data/
|   |-- raw/                       # Downloaded datasets (git-ignored)
|   +-- processed/                 # Cleaned CSVs used by all training scripts
|-- models/                        # Checkpoints (.pt), FAISS index (.bin), vocab JSON
|-- docs/
|   |-- part_a.md                  # DataBundle API spec
|   |-- part_b.md                  # CLIP training details and extract_embeddings usage
|   +-- part_c.md                  # MLP classifier details
|-- notebooks/
|   +-- exploration.ipynb          # EDA: genre distribution, image stats, sample grids
+-- tests/
    |-- test_loaders.py            # 18 tests for DataBundle, split, sampler
    +-- test_pipeline.py           # Smoke tests for encoding and retrieval
```

## Usage

```bash
# Train CLIP projection layers
python scripts/train_clip.py

# Extract image embeddings for classifier training
python scripts/extract_embeddings.py

# Train MLP genre classifier
python scripts/train_classifier.py

# Build FAISS index (adds dominant-color metadata per item)
python scripts/build_index.py

# Launch the app
streamlit run app.py
```
