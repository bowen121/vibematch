# Part B — CLIP Contrastive Training

Owner: **Bowen Tan**.

This module trains a dual-encoder that maps cover images and text descriptions
into a shared 256-d embedding space via symmetric contrastive loss. The trained
image encoder is what powers retrieval (Part D) and genre classification (Part C).

---

## TL;DR — train and extract embeddings

```bash
# 1. Train projection heads
python scripts/train_clip.py

# 2. Extract embeddings for Member C
python scripts/extract_embeddings.py
```

Outputs:
- `models/clip_projections.pt` — trained projection head weights
- `models/embeddings.pt` — (N, 256) image embeddings + multi-hot labels for MLP training

---

## Architecture

```
Image ──► ResNet-50 (frozen) ──► 2048-d ──► ProjectionHead ──► 256-d L2-norm ──►─┐
                                                                                   ├──► InfoNCE loss
Text  ──► DistilBERT (frozen) ──► 768-d  ──► ProjectionHead ──► 256-d L2-norm ──►─┘
```

**Only the two projection heads are trained.** Backbones stay frozen throughout.

`ProjectionHead`: `Linear → LayerNorm → GELU → Dropout(0.1) → Linear`

Because both backbones are frozen, `.train()` is overridden in `VibeMatchEncoder`
to keep them permanently in `eval()` mode — this prevents BatchNorm running-stat
drift and makes training deterministic with respect to the backbone.

Gradients flow through the projection heads even though the backbone forward
passes run inside `torch.no_grad()`, because the projection inputs (feature
tensors) are detached from the backbone computation graph but the projection
parameters themselves are still tracked.

---

## Loss — Symmetric InfoNCE

`src/model/loss.py` implements the standard CLIP loss:

```
logits  = image_emb @ text_emb.T / τ          # (B, B) cosine-similarity matrix
loss    = (CE(logits, labels) + CE(logits.T, labels)) / 2
```

- Diagonal entries are positive pairs; off-diagonal are negatives.
- Loss is averaged over both image→text and text→image directions.
- Temperature τ = 0.07 (fixed). Set `learnable_temp: true` in
  `configs/train_config.yaml` to let the model tune it.

**Batch size matters more than almost any other hyperparameter** for contrastive
learning — each item in the batch acts as a negative for every other item, so
doubling the batch size roughly doubles the number of negatives seen per step.

---

## Text inputs — VLM captions

A key design decision: the text paired with each image during training is **not**
the title, and it is **not** a genre template like `"a movie poster of a drama film"`.

Instead, `scripts/generate_descriptions.py` runs **Qwen2.5-VL-7B** over every
image in a two-stage pipeline:

| Stage | Input | Output |
|-------|-------|--------|
| 1 — dense | image + prompt | detailed description (objects, lighting, spatial, context) |
| 2 — rewrite | Stage 1 text | 3 concise factual captions, 15–30 words each |

Three captions (`caption_1`, `caption_2`, `caption_3`) are stored in the processed
CSVs. During training, one is sampled randomly per item per epoch.

**Why this matters:** if we train on genre templates but users search with natural
language, the embedding space sees two completely different distributions at train
vs. query time. VLM captions close that gap. The random sampling across 3 captions
also acts as text augmentation — with 66k images × 3 captions the model effectively
sees ~200k unique pairs, which regularizes the projection heads.

Rows without captions fall back to the genre template automatically (`dataset.py`).

---

## Training details

Hyperparameters live in `configs/train_config.yaml` under the `clip` key.

| setting | default | notes |
|---------|---------|-------|
| `projection_dim` | 256 | shared embedding dimension |
| `batch_size` | — | increase for stronger contrastive signal |
| `learning_rate` | — | AdamW |
| `weight_decay` | — | AdamW |
| `warmup_steps` | 500 | linear warmup |
| `epochs` | 30 | cosine decay after warmup |
| `temperature` | 0.07 | fixed; set `learnable_temp: true` to tune |

Scheduler: linear warmup for `warmup_steps` batches, then cosine decay to 0.
Gradient clipping: `max_norm=1.0`.

Source balancing (`balance_train_sources=True`) is always on — a
`WeightedRandomSampler` equalizes the ~8:1 book/movie imbalance so the model
doesn't converge to "everything looks like a book cover."

Best checkpoint (lowest val loss) is saved automatically to `models/clip_projections.pt`.

---

## What Member C receives

`scripts/extract_embeddings.py` encodes every image (train + val + test) with the
trained projection head and saves:

```python
data = torch.load("models/embeddings.pt")
data["embeddings"]  # torch.Tensor (N, 256), float32, L2-normalized
data["labels"]      # torch.Tensor (N, num_genres), float32, multi-hot
data["ids"]         # list[str], item IDs matching the CSV
```

**Important:** run `extract_embeddings.py` only after CLIP training finishes.
Pre-training embeddings carry only ImageNet features and will give misleading
MLP baselines.

---

## What Part D (retrieval) uses

`scripts/build_index.py` calls `model.encode_image()` on all images and builds a
FAISS `IndexFlatIP` (inner product on L2-normalized vectors = cosine similarity).
It reads the same `models/clip_projections.pt` weights. No separate step needed —
`build_index.py` handles it end-to-end.

---

## Key files

| file | purpose |
|------|---------|
| `src/model/encoder.py` | `VibeMatchEncoder`, `ProjectionHead` |
| `src/model/loss.py` | `SymmetricContrastiveLoss` (InfoNCE) |
| `src/loaders/dataset.py` | `MediaDataset` — caption sampling, image transforms |
| `scripts/train_clip.py` | training loop, scheduler, checkpointing |
| `scripts/generate_descriptions.py` | Qwen2.5-VL two-stage caption pipeline |
| `scripts/extract_embeddings.py` | dump (N, 256) embeddings for Member C |
| `configs/train_config.yaml` | all hyperparameters |
