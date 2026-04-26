"""Generates the FAISS index from image embeddings."""

from __future__ import annotations
 
import argparse
import sys
from pathlib import Path
 
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
 
import numpy as np
import torch
import yaml
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from tqdm import tqdm
 
from src.loaders.data_loader import load_processed_frame
from src.model.encoder import VibeMatchEncoder
from src.retrieval.engine import build_index, normalise, save_index
 
 
class PosterDataset(Dataset):
    """Returns (image_tensor, row_index) pairs for inference."""
 
    _transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
 
    def __init__(self, df, data_root: Path) -> None:
        self.df = df.reset_index(drop=True)
        self.data_root = data_root
 
    def __len__(self) -> int:
        return len(self.df)
 
    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        img_path = self.data_root / self.df.loc[idx, "image_path"]
        img = Image.open(img_path).convert("RGB")
        return self._transform(img), idx
 
 
def collate_skip_errors(batch):
    batch = [b for b in batch if b is not None]
    if not batch:
        return None
    tensors, indices = zip(*batch)
    return torch.stack(tensors), list(indices)
 
 
def build(
    train_cfg_path: str = "configs/train_config.yaml",
    app_cfg_path: str = "configs/app_config.yaml",
    processed_dir: str = "data/processed",
    data_root: str = ".",
    batch_size: int = 64,
    device: str | None = None,
) -> None:
    with open(train_cfg_path) as fh:
        train_cfg = yaml.safe_load(fh)
    with open(app_cfg_path) as fh:
        app_cfg = yaml.safe_load(fh)
 
    projection_dim: int = train_cfg["clip"]["projection_dim"]
    clip_weights: str = app_cfg["clip_weights_path"]
    index_path: str = app_cfg["index_path"]
 
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[build_index] device={device}  projection_dim={projection_dim}")
 
    df = load_processed_frame(processed_dir)
    print(f"[build_index] {len(df)} items loaded from {processed_dir}")
 
    encoder = VibeMatchEncoder(projection_dim=projection_dim).to(device)
    encoder.load_state_dict(torch.load(clip_weights, map_location=device))
    encoder.eval()
    print(f"[build_index] encoder weights loaded from {clip_weights}")
 
    loader = DataLoader(
        PosterDataset(df, data_root=Path(data_root)),
        batch_size=batch_size,
        num_workers=4,
        collate_fn=collate_skip_errors,
        pin_memory=(device == "cuda"),
    )
 
    all_embeddings: list[np.ndarray] = []
    all_meta: list[dict] = []
 
    with torch.no_grad():
        for batch in tqdm(loader, desc="Encoding images"):
            if batch is None:
                continue
            images, indices = batch
            img_emb = encoder.encode_image(images.to(device))
            all_embeddings.append(img_emb.cpu().numpy())
            for idx in indices:
                row = df.loc[idx]
                all_meta.append({
                    "image_path": str(row["image_path"]),
                    "genres": str(row["genres"]),   
                    "title": str(row.get("title", "")),
                    "source": str(row.get("source", "")),
                    "id": str(row.get("id", "")),
                })
 
    embeddings = normalise(np.vstack(all_embeddings).astype(np.float32))
    print(f"[build_index] encoded {len(embeddings)} images  shape={embeddings.shape}")
 
    index, metadata = build_index(embeddings, all_meta)
    save_index(index, metadata, index_path)
    print(f"[build_index] saved → {index_path}")
 
 
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build the VibeMatch FAISS index.")
    parser.add_argument("--train-config", default="configs/train_config.yaml")
    parser.add_argument("--app-config", default="configs/app_config.yaml")
    parser.add_argument("--processed-dir", default="data/processed")
    parser.add_argument("--data-root", default=".")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()
    build(args.train_config, args.app_config, args.processed_dir, args.data_root, args.batch_size, args.device)
