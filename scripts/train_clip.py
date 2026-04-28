"""Trains projection layers with contrastive loss."""

from __future__ import annotations

import argparse
import math
import shutil
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
import yaml
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from tqdm import tqdm
from transformers import DistilBertTokenizerFast

from src.loaders.data_loader import make_data_bundle
from src.model.encoder import VibeMatchEncoder
from src.model.loss import SymmetricContrastiveLoss


@torch.no_grad()
def _retrieval_metrics(
    all_img: torch.Tensor, all_txt: torch.Tensor
) -> dict[str, float]:
    """Compute R@1/5/10, median rank, alignment, and uniformity over full val set."""
    # (N, N) cosine similarity — embeddings are already L2-normalized
    sims = all_img @ all_txt.T
    N = sims.shape[0]
    labels = torch.arange(N, device=sims.device)

    def recall_at_k(sim_matrix: torch.Tensor, k: int) -> float:
        topk = sim_matrix.topk(k, dim=1).indices        # (N, k)
        hits = (topk == labels.unsqueeze(1)).any(dim=1)
        return hits.float().mean().item()

    def median_rank(sim_matrix: torch.Tensor) -> float:
        ranks = (sim_matrix > sim_matrix[labels, labels].unsqueeze(1)).sum(dim=1) + 1
        return ranks.float().median().item()

    # Alignment: mean cosine sim of positive pairs (higher = better)
    alignment = sims[labels, labels].mean().item()

    # Uniformity: log mean pairwise Gaussian kernel (lower = more uniform)
    sq_dists = torch.pdist(all_img, p=2).pow(2)
    uniformity = sq_dists.mul(-2).exp().mean().log().item()

    return {
        "R@1":        recall_at_k(sims, 1),
        "R@5":        recall_at_k(sims, 5),
        "R@10":       recall_at_k(sims, 10),
        "med_rank":   median_rank(sims),
        "alignment":  alignment,
        "uniformity": uniformity,
    }


def _make_scheduler(optimizer: AdamW, warmup_steps: int, total_steps: int) -> LambdaLR:
    """Linear warmup then cosine decay, stepped once per batch."""
    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1.0 + math.cos(math.pi * progress))
    return LambdaLR(optimizer, lr_lambda)


def train(
    train_cfg_path: str = "configs/train_config.yaml",
    processed_dir: str = "data/processed",
    data_root: str = ".",
    output_path: str = "models/clip_projections.pt",
    device: str | None = None,
    num_workers: int = 4,
) -> None:
    with open(train_cfg_path) as fh:
        cfg = yaml.safe_load(fh)["clip"]

    # ── run directory ──────────────────────────────────────────────────────────
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = Path("models/runs") / timestamp
    run_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy(train_cfg_path, run_dir / "config.yaml")
    print(f"[train_clip] run dir → {run_dir}")

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[train_clip] device={device}")

    tokenizer = DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased")
    bundle = make_data_bundle(
        processed_dir=processed_dir,
        data_root=data_root,
        tokenizer=tokenizer,
        batch_size=cfg["batch_size"],
        num_workers=num_workers,
        seed=42,
        balance_train_sources=True,
        pin_memory=(device == "cuda"),
    )
    print(
        f"[train_clip] train={len(bundle.train_df)}  val={len(bundle.val_df)}  "
        f"genres={len(bundle.genre_vocab)}"
    )

    model = VibeMatchEncoder(
        projection_dim=cfg["projection_dim"],
        dropout=cfg.get("dropout", 0.1),
    ).to(device)
    criterion = SymmetricContrastiveLoss(
        temperature=cfg["temperature"],
        learnable_temp=cfg.get("learnable_temp", False),
    ).to(device)

    # Only the projection heads are trained
    params = list(model.image_proj.parameters()) + list(model.text_proj.parameters())
    if cfg.get("learnable_temp", False):
        params.append(criterion.log_temp)
    optimizer = AdamW(params, lr=cfg["learning_rate"], weight_decay=cfg["weight_decay"])

    steps_per_epoch = len(bundle.train_loader)
    total_steps = cfg["epochs"] * steps_per_epoch
    warmup_steps = cfg.get("warmup_epochs", 0) * steps_per_epoch or cfg.get("warmup_steps", 0)
    scheduler = _make_scheduler(optimizer, warmup_steps=warmup_steps, total_steps=total_steps)
    print(f"[train_clip] warmup={warmup_steps} steps ({warmup_steps // steps_per_epoch} epochs)")

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    best_val_loss = float("inf")
    global_step = 0
    log_path = run_dir / "train.log"
    log_file = log_path.open("w")

    def log(msg: str) -> None:
        print(msg)
        log_file.write(msg + "\n")
        log_file.flush()

    for epoch in range(1, cfg["epochs"] + 1):
        # ── train ──────────────────────────────────────────────────────────────
        model.train()
        train_loss = 0.0
        for batch in tqdm(bundle.train_loader, desc=f"Epoch {epoch}/{cfg['epochs']} train", leave=False):
            images = batch["image"].to(device)
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)

            img_emb, txt_emb = model(images, input_ids, attention_mask)
            loss = criterion(img_emb, txt_emb)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, max_norm=1.0)
            optimizer.step()
            scheduler.step()

            train_loss += loss.item()
            global_step += 1

        train_loss /= steps_per_epoch

        # ── val ────────────────────────────────────────────────────────────────
        model.eval()
        val_loss = 0.0
        val_steps = len(bundle.val_loader)
        all_img_emb, all_txt_emb = [], []
        if val_steps > 0:
            with torch.no_grad():
                for batch in tqdm(bundle.val_loader, desc=f"Epoch {epoch}/{cfg['epochs']} val", leave=False):
                    images = batch["image"].to(device)
                    input_ids = batch["input_ids"].to(device)
                    attention_mask = batch["attention_mask"].to(device)
                    img_emb, txt_emb = model(images, input_ids, attention_mask)
                    val_loss += criterion(img_emb, txt_emb).item()
                    all_img_emb.append(img_emb)
                    all_txt_emb.append(txt_emb)
            val_loss /= val_steps

        current_lr = scheduler.get_last_lr()[0]
        log(
            f"Epoch {epoch:3d}/{cfg['epochs']} | "
            f"train={train_loss:.4f}  val={val_loss:.4f}  "
            f"lr={current_lr:.2e}  τ={criterion.temperature.item():.4f}"
        )

        if all_img_emb:
            m = _retrieval_metrics(torch.cat(all_img_emb), torch.cat(all_txt_emb))
            log(
                f"             | "
                f"R@1={m['R@1']:.3f}  R@5={m['R@5']:.3f}  R@10={m['R@10']:.3f}  "
                f"med_rank={m['med_rank']:.0f}  align={m['alignment']:.3f}  unif={m['uniformity']:.3f}"
            )

        # ── periodic checkpoint every 10 epochs ────────────────────────────────
        if epoch % 10 == 0:
            ckpt = run_dir / f"epoch_{epoch:03d}.pt"
            torch.save(model.state_dict(), ckpt)
            log(f"  → periodic checkpoint → {ckpt}")

        # ── best checkpoint ────────────────────────────────────────────────────
        if val_steps == 0 or val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), run_dir / "best.pt")
            shutil.copy(run_dir / "best.pt", output_path)
            log(f"  → best checkpoint → {run_dir / 'best.pt'}")

    log(f"[train_clip] done. best_val_loss={best_val_loss:.4f}  run → {run_dir}")
    log_file.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train VibeMatch CLIP projection layers.")
    parser.add_argument("--train-config", default="configs/train_config.yaml")
    parser.add_argument("--processed-dir", default="data/processed")
    parser.add_argument("--data-root", default=".")
    parser.add_argument("--output", default="models/clip_projections.pt")
    parser.add_argument("--device", default=None)
    parser.add_argument("--num-workers", type=int, default=4)
    args = parser.parse_args()
    train(
        args.train_config,
        args.processed_dir,
        args.data_root,
        args.output,
        args.device,
        args.num_workers,
    )
