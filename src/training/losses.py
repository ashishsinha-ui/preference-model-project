"""Losses. Targets are floats in [0, 1]: hard labels (0/1) or soft vote shares p_a."""
import torch.nn as nn
import torch.nn.functional as F


class SoftBCELoss(nn.Module):
    """The plan's SoftBCELoss, computed on logits (BCE-with-logits) so it stays stable in bf16."""

    def forward(self, logits, targets):
        return F.binary_cross_entropy_with_logits(logits.squeeze(-1).float(), targets.float())
