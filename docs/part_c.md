# Part C — Genre Classification & Testing

Owner: **Edoardo Mongardi**.

This module trains an MLP genre classifier on frozen image embeddings from
Member B's CLIP encoder, and provides a `predict_genres()` function that
Member D's web app calls to show auto-generated genre tags on search results.

---

## TL;DR — train the classifier

```bash
# After Member B has run extract_embeddings.py:
python scripts/train_classifier.py
```

Outputs:
- `models/genre_classifier.pt` — trained MLP weights
- `models/genre_vocab.json` — ordered genre list (maps output indices to names)

---

## Architecture

```
Frozen Embedding (256-d) ──► Linear(256, 512) ──► ReLU ──► Dropout(0.3)
                          ──► Linear(512, 256) ──► ReLU ──► Dropout(0.3)
                          ──► Linear(256, K)   ──► raw logits
```

- **K = 54** genres (determined by `build_genre_vocab(min_count=5)` from the
  processed CSVs).
- Output is raw logits (no sigmoid). `BCEWithLogitsLoss` is used for training
  since it's numerically more stable than applying sigmoid + BCE separately.
- At inference, `torch.sigmoid(logits)` converts to probabilities, and a
  threshold (default 0.5) selects which genre tags to display.

---

## Training details

| setting | value | notes |
|---------|-------|-------|
| `input_dim` | 256 | matches CLIP projection dimension |
| `hidden_dims` | [512, 256] | two hidden layers |
| `dropout` | 0.3 | applied after each hidden layer |
| `optimizer` | AdamW | lr=1e-3, weight_decay=1e-4 |
| `loss` | BCEWithLogitsLoss | multi-label (items can have multiple genres) |
| `epochs` | 50 | with early stopping |
| `patience` | 5 | stop if val loss hasn't improved for 5 consecutive epochs |
| `batch_size` | 128 | |

### Avoiding data leakage

`extract_embeddings.py` (Member B) saves all embeddings as a single flat
tensor. To avoid mixing CLIP-trained items into our test set, this script
**reconstructs the exact same split** that the CLIP encoder used:

1. Load `data/processed/{movies,books}.csv`
2. Run `genre_stratified_split(seed=42)` — same function and seed as
   `make_data_bundle` uses
3. Match each embedding's item ID against the split to partition into
   train / val / test

This guarantees our classifier's test metrics are evaluated on items the
encoder never saw during contrastive training.

### Results (current best checkpoint)

| split | loss | F1 (macro) | Precision (macro) | Recall (macro) |
|-------|------|------------|-------------------|----------------|
| val   | 0.0749 | 0.176 | 0.433 | 0.128 |
| test  | 0.0762 | 0.170 | 0.464 | 0.125 |

Precision is high (46%) — when the model predicts a genre, it's usually
correct. Recall is lower, which is expected for a 54-class multi-label
problem where each item has only ~1.3 genres on average. The model is
conservative at the 0.5 threshold. Lowering the threshold trades precision
for recall.

---

## Integration with the web app

Member D calls `predict_genres()` from `src/model/classifier.py`:

```python
from src.model.classifier import predict_genres, GenreClassifier

model = GenreClassifier(input_dim=256, num_genres=54)
model.load_state_dict(torch.load("models/genre_classifier.pt"))

genre_vocab = json.load(open("models/genre_vocab.json"))

tags = predict_genres(embedding, model, genre_vocab, threshold=0.5)
# → ["Drama", "Romance"]
```

The genre tags appear below each search result card in the Streamlit UI.

---

## Tests

```bash
pytest tests/test_pipeline.py -v
```

11 tests covering:

| # | test | what it checks |
|---|------|----------------|
| 1 | `test_output_shape` | forward pass produces `(B, K)` logits |
| 2 | `test_single_sample` | batch size 1 works |
| 3 | `test_output_is_raw_logits` | no accidental sigmoid in forward |
| 4 | `test_bce_loss_computes` | BCEWithLogitsLoss runs without NaN/Inf |
| 5 | `test_parameters_update` | gradients flow, optimizer updates weights |
| 6 | `test_train_one_epoch_returns_loss` | training loop returns a finite float |
| 7 | `test_evaluate_returns_metrics` | evaluate returns loss, F1, precision, recall in [0,1] |
| 8 | `test_eval_mode_deterministic` | dropout disabled in eval → deterministic output |
| 9 | `test_train_mode_has_dropout` | dropout active in train → non-deterministic output |
| 10 | `test_checkpoint_save_load` | save/load roundtrip produces identical output |
| 11 | `test_predict_genres` | `predict_genres()` returns valid genre strings |

All tests use synthetic data — no real dataset or trained encoder required.

---

## Key files

| file | purpose |
|------|---------|
| `src/model/classifier.py` | `GenreClassifier` MLP + `predict_genres()` inference helper |
| `scripts/train_classifier.py` | training loop, split reconstruction, early stopping |
| `models/genre_classifier.pt` | trained checkpoint |
| `models/genre_vocab.json` | ordered genre list (index → genre name mapping) |
| `tests/test_pipeline.py` | 11 smoke tests |
| `configs/train_config.yaml` | hyperparameters under `classifier:` key |
