"""Frozen ResNet + DistilBERT backbones with custom projection layers."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from transformers import DistilBertModel


class ProjectionHead(nn.Module):
    """Linear → LayerNorm → GELU → Dropout → Linear projection head."""

    def __init__(self, in_dim: int, out_dim: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, in_dim),
            nn.LayerNorm(in_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(in_dim, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class VibeMatchEncoder(nn.Module):
    """Dual-encoder: frozen ResNet-50 image backbone + frozen DistilBERT text backbone,
    each connected to a learned projection head mapping into a shared latent space.

    Only the projection heads are trained; backbone weights are frozen at construction.
    Backbone BatchNorm layers are kept in eval mode even during training (via the
    train() override) so their running statistics do not drift.

    Args:
        projection_dim: shared embedding dimensionality for both modalities.
        dropout: dropout probability inside the projection heads.
    """

    _IMAGE_FEAT_DIM = 2048   # ResNet-50 avgpool output
    _TEXT_FEAT_DIM = 768     # DistilBERT hidden size

    def __init__(self, projection_dim: int = 256, dropout: float = 0.1) -> None:
        super().__init__()

        # Image backbone: ResNet-50 without the final FC layer → (B, 2048, 1, 1)
        resnet = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
        self.image_backbone = nn.Sequential(*list(resnet.children())[:-1])
        for p in self.image_backbone.parameters():
            p.requires_grad_(False)

        # Text backbone: DistilBERT; we use the CLS token → (B, 768)
        self.text_backbone = DistilBertModel.from_pretrained("distilbert-base-uncased")
        for p in self.text_backbone.parameters():
            p.requires_grad_(False)

        # Projection heads — only these are trained
        self.image_proj = ProjectionHead(self._IMAGE_FEAT_DIM, projection_dim, dropout)
        self.text_proj = ProjectionHead(self._TEXT_FEAT_DIM, projection_dim, dropout)

    def train(self, mode: bool = True) -> "VibeMatchEncoder":
        """Keep frozen backbones permanently in eval mode to prevent BN stat drift."""
        super().train(mode)
        self.image_backbone.eval()
        self.text_backbone.eval()
        return self

    def encode_image(self, images: torch.Tensor) -> torch.Tensor:
        """Return L2-normalized image embeddings of shape (B, projection_dim).

        Uses torch.no_grad() for the backbone forward pass to save memory.
        Gradients still flow through image_proj.
        """
        with torch.no_grad():
            feats = self.image_backbone(images)   # (B, 2048, 1, 1)
        feats = feats.flatten(1)                  # (B, 2048)
        proj = self.image_proj(feats)             # (B, projection_dim)
        return F.normalize(proj, dim=-1)

    def encode_text(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor
    ) -> torch.Tensor:
        """Return L2-normalized text embeddings of shape (B, projection_dim).

        Uses the CLS token (index 0) from the last hidden state.
        """
        with torch.no_grad():
            out = self.text_backbone(input_ids=input_ids, attention_mask=attention_mask)
        cls = out.last_hidden_state[:, 0, :]     # (B, 768)
        proj = self.text_proj(cls)               # (B, projection_dim)
        return F.normalize(proj, dim=-1)

    def forward(
        self,
        images: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return (image_emb, text_emb) — both L2-normalized, shape (B, projection_dim)."""
        return self.encode_image(images), self.encode_text(input_ids, attention_mask)
