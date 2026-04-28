"""Model: dual-encoder, contrastive loss, and MLP classifier."""

from src.model.encoder import ProjectionHead, VibeMatchEncoder
from src.model.loss import SymmetricContrastiveLoss

__all__ = ["ProjectionHead", "VibeMatchEncoder", "SymmetricContrastiveLoss"]
