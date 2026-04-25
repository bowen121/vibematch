"""Trains the MLP genre classifier on frozen embeddings.

Usage (after Member B saves frozen embeddings):
    python scripts/train_classifier.py

Expects:
  - Member B's frozen embeddings: models/frozen_embeddings.pt
    Format: dict with keys "train", "val", "test", each a (N, 256) tensor
  - Member A's labels: loaded via DataBundle for genre_vocab + label vectors

The script:
  1. Loads pre-computed frozen 256-d embeddings
  2. Pairs them with multi-hot genre labels from Member A
  3. Trains the GenreClassifier MLP using BCEWithLogitsLoss + AdamW
  4. Applies early stopping (patience=5) on validation loss
  5. Saves the best checkpoint to models/genre_classifier.pt
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from pathlib import Path
from sklearn.metrics import f1_score, precision_score, recall_score

from src.model.classifier import GenreClassifier

# ── Hyperparameters (from configs/train_config.yaml) ──────────────────
NUM_EPOCHS = 50
BATCH_SIZE = 128
LR = 1e-3
WEIGHT_DECAY = 1e-4
PATIENCE = 5
CHECKPOINT_PATH = Path("models/genre_classifier.pt")
EMBEDDINGS_PATH = Path("models/frozen_embeddings.pt")


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

    # TODO: Uncomment once Member B provides frozen_embeddings.pt
    #       and Member A's DataBundle is merged.
    #
    # # Load frozen embeddings from Member B
    # data = torch.load(EMBEDDINGS_PATH, weights_only=True)
    # train_emb = data["train"]   # (N_train, 256)
    # val_emb   = data["val"]     # (N_val, 256)
    # test_emb  = data["test"]    # (N_test, 256)
    #
    # # Load labels from Member A
    # from src.loaders import make_data_bundle
    # from transformers import DistilBertTokenizerFast
    # tokenizer = DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased")
    # bundle = make_data_bundle("data/processed", ".", tokenizer=tokenizer, seed=42)
    #
    # train_labels = ...  # (N_train, K) from bundle
    # val_labels   = ...  # (N_val, K) from bundle
    #
    # num_genres = len(bundle.genre_vocab)
    # train_loader = DataLoader(TensorDataset(train_emb, train_labels), batch_size=BATCH_SIZE, shuffle=True)
    # val_loader   = DataLoader(TensorDataset(val_emb, val_labels), batch_size=BATCH_SIZE)
    #
    # model = GenreClassifier(input_dim=256, num_genres=num_genres).to(device)
    # train_classifier(model, train_loader, val_loader, device)

    # ── Dummy data smoke test (works right now) ───────────────────────
    print("Running smoke test with dummy data...")

    num_genres = 25
    train_emb    = torch.randn(256, 256)
    train_labels = torch.randint(0, 2, (256, num_genres)).float()
    val_emb      = torch.randn(64, 256)
    val_labels   = torch.randint(0, 2, (64, num_genres)).float()

    train_loader = DataLoader(TensorDataset(train_emb, train_labels), batch_size=64, shuffle=True)
    val_loader   = DataLoader(TensorDataset(val_emb, val_labels), batch_size=64)

    model = GenreClassifier(input_dim=256, num_genres=num_genres).to(device)
    train_classifier(model, train_loader, val_loader, device)

