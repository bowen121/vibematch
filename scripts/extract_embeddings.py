"""Extract image embeddings from the trained VibeMatchEncoder and save for MLP training.

Saves models/embeddings.pt with keys:
    embeddings  (N, 256) float32 L2-normalized image embeddings
    labels      (N, num_genres) float32 multi-hot genre labels
    ids         list[str] item IDs

Usage:
    python scripts/extract_embeddings.py
    python scripts/extract_embeddings.py --weights models/clip_projections.pt --output models/embeddings.pt
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
from tqdm import tqdm
from transformers import DistilBertTokenizerFast

from src.loaders.data_loader import make_data_bundle
from src.model.encoder import VibeMatchEncoder


def extract(
    weights_path: str,
    processed_dir: str,
    data_root: str,
    output_path: str,
    batch_size: int,
    device: str | None,
) -> None:
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[extract] device={device}")

    model = VibeMatchEncoder(projection_dim=256).to(device)
    model.load_state_dict(torch.load(weights_path, map_location=device))
    model.eval()
    print(f"[extract] loaded weights from {weights_path}")

    tokenizer = DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased")
    bundle = make_data_bundle(
        processed_dir=processed_dir,
        data_root=data_root,
        tokenizer=tokenizer,
        batch_size=batch_size,
        num_workers=4,
        seed=42,
        pin_memory=(device == "cuda"),
    )

    all_embs: list[torch.Tensor] = []
    all_labels: list[torch.Tensor] = []
    all_ids: list[str] = []

    loaders = [
        ("train", bundle.train_loader),
        ("val", bundle.val_loader),
        ("test", bundle.test_loader),
    ]

    with torch.no_grad():
        for split_name, loader in loaders:
            for batch in tqdm(loader, desc=split_name):
                embs = model.encode_image(batch["image"].to(device))
                all_embs.append(embs.cpu())
                all_labels.append(batch["labels"].float())
                all_ids.extend(batch["item_id"])

    embeddings = torch.cat(all_embs)   # (N, 256)
    labels = torch.cat(all_labels)     # (N, num_genres)

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"embeddings": embeddings, "labels": labels, "ids": all_ids}, out)
    print(f"[extract] saved {embeddings.shape[0]} embeddings → {out}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", default="models/clip_projections.pt")
    parser.add_argument("--processed-dir", default="data/processed")
    parser.add_argument("--data-root", default=".")
    parser.add_argument("--output", default="models/embeddings.pt")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()
    extract(args.weights, args.processed_dir, args.data_root, args.output, args.batch_size, args.device)
