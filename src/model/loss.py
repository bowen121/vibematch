"""Symmetric contrastive loss (InfoNCE) function."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class SymmetricContrastiveLoss(nn.Module):
    """Symmetric InfoNCE loss for image–text pairs.

    Both image_emb and text_emb are expected to be L2-normalized so that the
    dot product equals cosine similarity. The loss is averaged over both
    directions (image→text and text→image), which is what makes it symmetric.

    Args:
        temperature: scaling factor τ. Typical range 0.01–0.1.
        learnable_temp: if True, τ is a nn.Parameter and updated by the optimizer.
    """

    def __init__(self, temperature: float = 0.07, learnable_temp: bool = False) -> None:
        super().__init__()
        log_t = torch.tensor(temperature).log()
        if learnable_temp:
            self.log_temp = nn.Parameter(log_t)
        else:
            self.register_buffer("log_temp", log_t)

    @property
    def temperature(self) -> torch.Tensor:
        return self.log_temp.exp()

    def forward(self, image_emb: torch.Tensor, text_emb: torch.Tensor) -> torch.Tensor:
        """Compute symmetric InfoNCE loss.

        Args:
            image_emb: (B, D) L2-normalized image embeddings.
            text_emb:  (B, D) L2-normalized text embeddings.

        Returns:
            Scalar loss averaged over both cross-modal directions.
        """
        B = image_emb.shape[0]
        labels = torch.arange(B, device=image_emb.device)

        # (B, B) cosine-similarity matrix; diagonal entries are positive pairs
        logits = (image_emb @ text_emb.T) / self.temperature

        loss_i2t = F.cross_entropy(logits, labels)
        loss_t2i = F.cross_entropy(logits.T, labels)
        return (loss_i2t + loss_t2i) / 2
