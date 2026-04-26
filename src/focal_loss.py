"""
src/focal_loss.py — Per-Endpoint Focal Loss with NaN Masking

FIX v2: Normalized alpha weighting to prevent loss scale imbalance
across endpoints with wildly different class ratios (5× vs 34×).

Previously:
    alpha_t = alpha * target + 1.0 * (1 - target)
    → negative class always gets weight 1.0 regardless of imbalance
    → loss dominated by high-imbalance endpoints (NR-PPAR-gamma at 34×)

Now:
    alpha_pos = alpha / (1 + alpha)   e.g. 34 → 0.971
    alpha_neg = 1.0 / (1 + alpha)     e.g. 34 → 0.029
    → loss scale is consistent across all 12 endpoints
    → model no longer ignores low-imbalance endpoints during training

Reference: Lin et al. (2017) "Focal Loss for Dense Object Detection" ICCV
"""

import torch
import torch.nn as nn
import numpy as np


class PerEndpointFocalLoss(nn.Module):
    """
    Focal Loss applied independently to each of the 12 Tox21 endpoints,
    with normalized per-endpoint alpha weights and NaN masking.

    Args:
        pos_weights: Array of shape (12,) — ratio of negatives to positives
        gamma: Focusing parameter (default 2.0)
        label_smoothing: Smooths hard 0/1 targets to reduce overconfidence (default 0.05)
    """

    def __init__(self, pos_weights: np.ndarray, gamma: float = 2.0,
                 label_smoothing: float = 0.05):
        super().__init__()
        self.gamma = gamma
        self.label_smoothing = label_smoothing

        pos_weights_tensor = torch.tensor(pos_weights, dtype=torch.float32)

        # Normalized alpha: positive class gets alpha/(1+alpha), negative gets 1/(1+alpha)
        # This keeps the total loss contribution balanced regardless of imbalance ratio
        alpha_pos = pos_weights_tensor / (1.0 + pos_weights_tensor)  # shape (12,)
        alpha_neg = 1.0 / (1.0 + pos_weights_tensor)                 # shape (12,)

        self.register_buffer('alpha_pos', alpha_pos)
        self.register_buffer('alpha_neg', alpha_neg)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Compute Focal Loss with NaN masking.

        Args:
            logits:  Raw model outputs, shape (batch, 12). NOT sigmoid-ed.
            targets: True labels, shape (batch, 12). Values: 0, 1, or -1 (missing).

        Returns:
            Scalar loss value (mean over valid entries only)
        """
        # Mask for valid labels (0 or 1); missing = -1
        mask = (targets >= 0).float()

        # Label smoothing: push hard 0/1 toward 0.05/0.95
        targets_smooth = targets.clamp(min=0)
        if self.label_smoothing > 0:
            targets_smooth = targets_smooth * (1 - self.label_smoothing) + \
                             0.5 * self.label_smoothing

        probs = torch.sigmoid(logits)

        # BCE element-wise (no reduction)
        bce = nn.functional.binary_cross_entropy_with_logits(
            logits, targets_smooth, reduction='none'
        )

        # Focal modulation: (1 - p_t)^gamma where p_t = prob of correct class
        targets_hard = targets.clamp(min=0)  # use hard targets for focal weight
        p_t = probs * targets_hard + (1 - probs) * (1 - targets_hard)
        focal_weight = (1 - p_t) ** self.gamma

        # Normalized alpha: select pos or neg alpha per entry
        # alpha_pos/alpha_neg are (12,) — broadcast over batch dimension
        alpha_t = self.alpha_pos * targets_hard + self.alpha_neg * (1 - targets_hard)

        # Final loss: focal_weight * alpha * bce, masked
        loss = focal_weight * alpha_t * bce * mask

        n_valid = mask.sum()
        return loss.sum() / n_valid.clamp(min=1.0)


def compute_pos_weights(df, target_cols: list) -> np.ndarray:
    """
    Compute per-endpoint positive class weights from the dataset.
    Weight = n_negatives / n_positives for each endpoint.
    """
    weights = []
    for col in target_cols:
        n_pos = (df[col] == 1).sum()
        n_neg = (df[col] == 0).sum()
        w = float(n_neg / n_pos) if n_pos > 0 else 1.0
        weights.append(w)

    weights = np.array(weights, dtype=np.float32)

    print("Per-endpoint positive weights (n_neg / n_pos):")
    for col, w in zip(target_cols, weights):
        print(f"  {col:20s}  weight={w:.1f}")

    return weights


# ── Quick test ─────────────────────────────────────────────────────
if __name__ == '__main__':
    import sys
    sys.path.insert(0, '.')
    from src.featurize import load_and_clean_tox21, TARGET_COLS

    df = load_and_clean_tox21()
    pos_weights = compute_pos_weights(df, TARGET_COLS)
    loss_fn = PerEndpointFocalLoss(pos_weights, gamma=2.0)

    batch_size = 32
    dummy_logits = torch.randn(batch_size, 12)
    dummy_targets = torch.randint(0, 2, (batch_size, 12)).float()
    dummy_targets[0, 3] = -1
    dummy_targets[1, 7] = -1

    loss = loss_fn(dummy_logits, dummy_targets)
    print(f"\nFocal Loss (gamma=2.0, normalized alpha): {loss.item():.4f}")
    assert not torch.isnan(loss), "Loss is NaN!"
    assert loss.item() > 0, "Loss should be positive!"
    print("focal_loss.py v2: ALL TESTS PASSED")