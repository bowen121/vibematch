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

**1. Get a Kaggle API token** — go to https://www.kaggle.com/settings/account, click "Create New Token", then:

```bash
mkdir -p ~/.kaggle
mv ~/Downloads/kaggle.json ~/.kaggle/
chmod 600 ~/.kaggle/kaggle.json
```

**2. Accept dataset rules** — open each dataset page above in a browser (while signed in) and click *"I Understand and Accept"*. Without this, downloads return 403.

**3. Download** — pulls both datasets into `data/raw/{movies,books}/` (skips if already present):

```bash
python scripts/download_data.py
# or just one: --movies / --books     re-download: --force
```

**4. Preprocess** — flattens both raw dumps into `data/processed/{movies,books}.csv` with a unified schema (`id, image_path, title, genres, source`):

```bash
python scripts/preprocess.py
```

**5. EDA** — open `notebooks/exploration.ipynb` to see genre distribution, image-size stats, sample grids, and split balance.

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