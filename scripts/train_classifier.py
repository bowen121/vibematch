"""Trains the MLP genre classifier on frozen embeddings.

Usage (after Member B saves frozen embeddings):
    python scripts/train_classifier.py

Expects:
  - Member B's frozen embeddings: embeddings.pt (or models/embeddings.pt)
    Format: dict with keys "embeddings" (N, 256), "labels" (N, num_genres), "ids" list[str]
  - Processed CSVs at data/processed/{movies,books}.csv (from Member A)

The script:
  1. Loads pre-computed frozen 256‑d embeddings
  2. Reconstructs the same train/val/test split that Member B used when
     extracting (genre_stratified_split with seed=42), partitioning by item ID
     to avoid data leakage
  3. Trains the GenreClassifier MLP using BCEWithLogitsLoss + AdamW
  4. Applies early stopping (patience=5) on validation loss
  5. Saves the best checkpoint to models/genre_classifier.pt
  6. Saves the genre vocabulary to models/genre_vocab.json
"""

import json
import sys
from pathlib import Path

# Ensure repo root is on sys.path for src imports
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import f1_score, precision_score, recall_score

from src.loaders.data_loader import load_processed_frame
from src.loaders.dataset import parse_genres
from src.loaders.split import build_genre_vocab, genre_stratified_split
from src.model.classifier import GenreClassifier

# ── Hyperparameters (from configs/train_config.yaml) ──────────────────
NUM_EPOCHS = 50
BATCH_SIZE = 128
LR = 1e-3
WEIGHT_DECAY = 1e-4
PATIENCE = 5
CHECKPOINT_PATH = Path("models/genre_classifier.pt")
VOCAB_PATH = Path("models/genre_vocab.json")

# Member B's embeddings — check repo root first, then models/
EMBEDDINGS_PATH = Path("embeddings.pt")
if not EMBEDDINGS_PATH.exists():
    EMBEDDINGS_PATH = Path("models/embeddings.pt")

PROCESSED_DIR = Path("data/processed")


def load_embeddings(path: Path) -> tuple[torch.Tensor, torch.Tensor, list]:
    """Load Member B's frozen embeddings file.

    Returns:
        embeddings: (N, 256) float32 L2-normalized image embeddings
        labels:     (N, num_genres) float32 multi-hot genre labels
        ids:        list[str] of item IDs
    """
    data = torch.load(path, map_location="cpu", weights_only=False)
    return data["embeddings"], data["labels"], data["ids"]


def reconstruct_split(
    processed_dir: Path,
    ids: list[str],
    val_frac: float = 0.1,
    test_frac: float = 0.1,
    seed: int = 42,
    min_genre_count: int = 5,
) -> tuple[list[int], list[int], list[int], list[str]]:
    """Reconstruct the same train/val/test partition that extract_embeddings.py
    used, by matching item IDs against genre_stratified_split(seed=42).

    This avoids data leakage: items the CLIP encoder was trained on stay in
    train, and items it never saw stay in val/test.

    Returns:
        train_indices, val_indices, test_indices  — into the embeddings tensor
        genre_vocab — ordered genre list for multi-hot label indexing
    """
    df = load_processed_frame(processed_dir)
    genres_per_item = [parse_genres(g) for g in df["genres"]]
    genre_vocab = build_genre_vocab(genres_per_item, min_count=min_genre_count)
    split = genre_stratified_split(
        genres_per_item, val_frac=val_frac, test_frac=test_frac, seed=seed
    )

    # Build ID → split membership lookup
    train_id_set = set(df.iloc[split.train]["id"].values)
    val_id_set = set(df.iloc[split.val]["id"].values)
    # Anything not in train or val is test (covers edge cases with drop_last)

    train_indices: list[int] = []
    val_indices: list[int] = []
    test_indices: list[int] = []

    for i, item_id in enumerate(ids):
        if item_id in train_id_set:
            train_indices.append(i)
        elif item_id in val_id_set:
            val_indices.append(i)
        else:
            test_indices.append(i)

    return train_indices, val_indices, test_indices, genre_vocab


def train_one_epoch(model, train_loader, criterion, optimizer, device):
    """Run one training epoch. Returns average training loss."""
    model.train()
    total_loss = 0.0
    num_batches = 0

    for embeddings, labels in train_loader:
        embeddings = embeddings.to(device)       # (B, 256)
        labels = labels.to(device)               # (B, K) multi-hot

        logits = model(embeddings)               # (B, K)
        loss = criterion(logits, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        num_batches += 1

    return total_loss / max(num_batches, 1)


@torch.no_grad()
def evaluate(model, val_loader, criterion, device):
    """Run validation. Returns (val_loss, f1, precision, recall)."""
    model.eval()
    total_loss = 0.0
    num_batches = 0

    all_preds = []
    all_labels = []

    for embeddings, labels in val_loader:
        embeddings = embeddings.to(device)
        labels = labels.to(device)

        logits = model(embeddings)
        loss = criterion(logits, labels)

        total_loss += loss.item()
        num_batches += 1

        # Track predictions for metrics calculation
        probs = torch.sigmoid(logits)
        preds = (probs > 0.5).int()
        all_preds.append(preds.cpu())
        all_labels.append(labels.cpu().int())

    avg_loss = total_loss / max(num_batches, 1)

    # Compute metrics across the entire validation set using macro average
    y_true = torch.cat(all_labels).numpy()
    y_pred = torch.cat(all_preds).numpy()

    # zero_division=0 prevents warnings if a model predicts all 0s for a genre
    f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    precision = precision_score(y_true, y_pred, average="macro", zero_division=0)
    recall = recall_score(y_true, y_pred, average="macro", zero_division=0)

    return avg_loss, f1, precision, recall


def train_classifier(model, train_loader, val_loader, device):
    """Full training loop with early stopping (patience=5).

    Saves the best model checkpoint to CHECKPOINT_PATH whenever
    validation loss improves. Stops early if val loss hasn't improved
    for PATIENCE consecutive epochs.
    """
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LR,
        weight_decay=WEIGHT_DECAY,
    )

    best_val_loss = float("inf")
    epochs_without_improvement = 0

    for epoch in range(1, NUM_EPOCHS + 1):
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, f1, precision, recall = evaluate(model, val_loader, criterion, device)

        print(f"Epoch {epoch:3d} | train_loss={train_loss:.4f} | val_loss={val_loss:.4f} | F1={f1:.4f} | Prec={precision:.4f} | Rec={recall:.4f}")

        # ── Early stopping logic ──────────────────────────────────────
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_without_improvement = 0
            CHECKPOINT_PATH.parent.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), CHECKPOINT_PATH)
            print(f"  ✓ Saved best model (val_loss={val_loss:.4f}, F1={f1:.4f})")
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= PATIENCE:
                print(f"  Early stopping at epoch {epoch} (patience={PATIENCE})")
                break

    # Reload the best checkpoint
    model.load_state_dict(torch.load(CHECKPOINT_PATH, weights_only=True))
    print(f"\nTraining complete. Best val_loss={best_val_loss:.4f}")
    return model


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if not EMBEDDINGS_PATH.exists():
        print("ERROR: embeddings.pt not found.")
        print("  Expected at: ./embeddings.pt or ./models/embeddings.pt")
        print("  Run `python scripts/extract_embeddings.py` first (Member B's script).")
        sys.exit(1)

    if not PROCESSED_DIR.exists():
        print("ERROR: data/processed/ not found.")
        print("  Run `python scripts/prepare_data.py` first (Member A's script).")
        sys.exit(1)

    # ── Load embeddings ───────────────────────────────────────────────
    print(f"Loading embeddings from {EMBEDDINGS_PATH}...")
    embeddings, labels, ids = load_embeddings(EMBEDDINGS_PATH)
    num_genres = labels.shape[1]
    print(f"  {embeddings.shape[0]} samples, {num_genres} genres, dim={embeddings.shape[1]}")

    # ── Reconstruct the correct split (matches extract_embeddings.py) ─
    print("Reconstructing train/val/test split from processed CSVs (seed=42)...")
    train_idx, val_idx, test_idx, genre_vocab = reconstruct_split(
        PROCESSED_DIR, ids, val_frac=0.1, test_frac=0.1, seed=42, min_genre_count=5
    )
    print(f"  train={len(train_idx)}, val={len(val_idx)}, test={len(test_idx)}")
    print(f"  genre_vocab: {len(genre_vocab)} genres")

    # ── Save genre vocabulary (needed by app.py for predict_genres) ───
    VOCAB_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(VOCAB_PATH, "w") as f:
        json.dump(genre_vocab, f, indent=2)
    print(f"  ✓ Saved genre vocabulary → {VOCAB_PATH}")

    # ── Build data loaders ────────────────────────────────────────────
    train_emb = embeddings[train_idx]
    train_labels = labels[train_idx]
    val_emb = embeddings[val_idx]
    val_labels = labels[val_idx]
    test_emb = embeddings[test_idx]
    test_labels = labels[test_idx]

    train_loader = DataLoader(
        TensorDataset(train_emb, train_labels),
        batch_size=BATCH_SIZE,
        shuffle=True,
    )
    val_loader = DataLoader(
        TensorDataset(val_emb, val_labels),
        batch_size=BATCH_SIZE,
    )

    # ── Train ─────────────────────────────────────────────────────────
    model = GenreClassifier(input_dim=256, num_genres=num_genres).to(device)
    print(f"\nTraining GenreClassifier (input_dim=256, num_genres={num_genres})...")
    print(f"  device={device}, epochs={NUM_EPOCHS}, batch_size={BATCH_SIZE}, lr={LR}\n")

    trained_model = train_classifier(model, train_loader, val_loader, device)

    # ── Final test evaluation ─────────────────────────────────────────
    test_loader = DataLoader(
        TensorDataset(test_emb, test_labels),
        batch_size=BATCH_SIZE,
    )
    criterion = nn.BCEWithLogitsLoss()
    test_loss, test_f1, test_prec, test_rec = evaluate(trained_model, test_loader, criterion, device)
    print(f"\n═══ Test Results ═══")
    print(f"  test_loss={test_loss:.4f} | F1={test_f1:.4f} | Prec={test_prec:.4f} | Rec={test_rec:.4f}")
