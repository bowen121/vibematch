"""Smoke tests for the MLP genre classifier pipeline.

All tests use synthetic data — no real dataset or trained encoder required.
Run with:  pytest tests/test_pipeline.py -v
"""

import torch
import torch.nn as nn
import pytest
from pathlib import Path

from src.model.classifier import GenreClassifier
from scripts.train_classifier import train_one_epoch, evaluate, train_classifier


# ── Fixtures ──────────────────────────────────────────────────────────

@pytest.fixture
def device():
    return torch.device("cpu")


@pytest.fixture
def num_genres():
    return 25


@pytest.fixture
def model(num_genres, device):
    return GenreClassifier(input_dim=256, num_genres=num_genres).to(device)


@pytest.fixture
def dummy_loaders(num_genres):
    """Create small train and val loaders with random data."""
    train_emb = torch.randn(64, 256)
    train_labels = torch.randint(0, 2, (64, num_genres)).float()
    val_emb = torch.randn(32, 256)
    val_labels = torch.randint(0, 2, (32, num_genres)).float()

    train_loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(train_emb, train_labels),
        batch_size=16, shuffle=True,
    )
    val_loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(val_emb, val_labels),
        batch_size=16,
    )
    return train_loader, val_loader


# ── Test 1: Output shape ─────────────────────────────────────────────

def test_output_shape(model, num_genres, device):
    """Classifier output must be (batch_size, num_genres)."""
    batch = torch.randn(8, 256, device=device)
    logits = model(batch)
    assert logits.shape == (8, num_genres), (
        f"Expected (8, {num_genres}), got {logits.shape}"
    )


# ── Test 2: Single sample ────────────────────────────────────────────

def test_single_sample(model, num_genres, device):
    """Classifier should handle a batch of size 1."""
    single = torch.randn(1, 256, device=device)
    logits = model(single)
    assert logits.shape == (1, num_genres)


# ── Test 3: Output is raw logits (no sigmoid) ────────────────────────

def test_output_is_raw_logits(model, device):
    """Output should contain values outside [0, 1] since no sigmoid is applied."""
    torch.manual_seed(0)
    batch = torch.randn(64, 256, device=device)
    logits = model(batch)
    # With random weights, logits should not all be in [0, 1]
    assert logits.min() < 0.0 or logits.max() > 1.0, (
        "Logits appear to be bounded in [0,1] — model may have an unwanted sigmoid"
    )


# ── Test 4: BCE loss computes without error ──────────────────────────

def test_bce_loss_computes(model, num_genres, device):
    """BCEWithLogitsLoss should run on model output without crashing."""
    criterion = nn.BCEWithLogitsLoss()
    batch = torch.randn(8, 256, device=device)
    labels = torch.randint(0, 2, (8, num_genres), device=device).float()

    logits = model(batch)
    loss = criterion(logits, labels)

    assert loss.shape == (), "Loss should be a scalar"
    assert not torch.isnan(loss), "Loss is NaN"
    assert not torch.isinf(loss), "Loss is Inf"


# ── Test 5: Gradients flow and parameters update ─────────────────────

def test_parameters_update(model, num_genres, device):
    """After one optimizer step, at least some parameters should change."""
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

    batch = torch.randn(8, 256, device=device)
    labels = torch.randint(0, 2, (8, num_genres), device=device).float()

    # Save original parameters
    params_before = [p.clone() for p in model.parameters()]

    logits = model(batch)
    loss = criterion(logits, labels)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # Check that at least one parameter changed
    any_changed = False
    for before, after in zip(params_before, model.parameters()):
        if not torch.equal(before, after.data):
            any_changed = True
            break
    assert any_changed, "No parameters updated after optimizer.step()"


# ── Test 6: train_one_epoch returns a float loss ─────────────────────

def test_train_one_epoch_returns_loss(model, dummy_loaders, device):
    """train_one_epoch should return a finite float."""
    train_loader, _ = dummy_loaders
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

    loss = train_one_epoch(model, train_loader, criterion, optimizer, device)

    assert isinstance(loss, float)
    assert loss > 0.0, "Loss should be positive for random data"
    assert not float("inf") == loss


# ── Test 7: evaluate returns a float loss and metrics ───────────────────

def test_evaluate_returns_metrics(model, dummy_loaders, device):
    """evaluate should return valid training metrics: loss, f1, prec, recall."""
    _, val_loader = dummy_loaders
    criterion = nn.BCEWithLogitsLoss()

    val_loss, f1, precision, recall = evaluate(model, val_loader, criterion, device)

    assert isinstance(val_loss, float), "Val loss should be a float"
    assert val_loss > 0.0, "Val loss should be positive"
    assert 0.0 <= f1 <= 1.0, f"F1 score {f1} out of bounds"
    assert 0.0 <= precision <= 1.0, "Precision out of bounds"
    assert 0.0 <= recall <= 1.0, "Recall out of bounds"


# ── Test 8: model.eval() disables dropout ─────────────────────────────

def test_eval_mode_deterministic(model, device):
    """In eval mode, two forward passes on the same input should be identical."""
    model.eval()
    batch = torch.randn(8, 256, device=device)

    out1 = model(batch)
    out2 = model(batch)

    assert torch.equal(out1, out2), "eval() mode should produce deterministic output"


# ── Test 9: train mode has dropout (non-deterministic) ────────────────

def test_train_mode_has_dropout(model, device):
    """In train mode, dropout should make outputs vary between passes."""
    model.train()
    batch = torch.randn(8, 256, device=device)

    out1 = model(batch)
    out2 = model(batch)

    # With dropout=0.3, outputs should differ (extremely unlikely to match)
    assert not torch.equal(out1, out2), "train() mode should have active dropout"


# ── Test 10: checkpoint saves and loads correctly ─────────────────────

def test_checkpoint_save_load(model, num_genres, device, tmp_path):
    """Model state_dict should save and reload correctly."""
    checkpoint = tmp_path / "test_model.pt"
    torch.save(model.state_dict(), checkpoint)

    loaded_model = GenreClassifier(input_dim=256, num_genres=num_genres).to(device)
    loaded_model.load_state_dict(torch.load(checkpoint, weights_only=True))

    batch = torch.randn(4, 256, device=device)
    model.eval()
    loaded_model.eval()

    assert torch.equal(model(batch), loaded_model(batch)), (
        "Loaded model should produce identical output"
    )


# ── Test 11: predict_genres returns valid genre strings ───────────────

def test_predict_genres(model, device):
    """predict_genres should return a list of genre strings."""
    from src.model.classifier import predict_genres

    genre_vocab = [f"genre_{i}" for i in range(25)]
    embedding = torch.randn(256, device=device)

    predictions = predict_genres(embedding, model, genre_vocab, threshold=0.5)

    assert isinstance(predictions, list)
    for genre in predictions:
        assert genre in genre_vocab, f"'{genre}' not in genre_vocab"
