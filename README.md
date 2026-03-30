# VibeMatch

Visual-semantic retrieval system that matches movie/book vibes using CLIP embeddings and FAISS similarity search, with an optional MLP genre classifier.

## Project Structure

```
vibematch/
├── data/
│   ├── raw/               # Original downloaded datasets (posters, covers)
│   └── processed/         # Cleaned images and metadata CSVs after preprocessing
├── src/
│   ├── data_loader.py     # Dataset classes, image transforms, and data splitting
│   ├── encoder.py         # Dual encoder (CLIP) wrapper for image/text embeddings
│   ├── retrieval.py       # FAISS index building and cosine-similarity search
│   ├── classifier.py      # MLP genre classifier architecture and training loop
│   └── app.py             # Streamlit web app connecting query input to results
├── notebooks/
│   └── exploration.ipynb  # EDA: dataset stats, sample visualizations, baselines
├── models/                # Saved model weights and FAISS index files
├── tests/
│   └── test_pipeline.py   # Smoke tests for data loading, encoding, and retrieval
├── requirements.txt
└── README.md
```