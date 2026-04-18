# VibeMatch

Aesthetic-driven media retrieval system that matches movie/book vibes using a custom-built CLIP framework with contrastive learning, cosine similarity search, and a live MLP genre classifier.

## Setup

```bash
pip install -r requirements.txt
```

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