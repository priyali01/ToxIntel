"""
src/focal_loss.py — Per-Endpoint Focal Loss with NaN Masking

Standard BCE (Binary Cross Entropy) treats every misclassification equally.
With 97% non-toxic vs 3% toxic, the model can just predict "non-toxic always"
and get 97% accuracy — which is useless.

Focal Loss fixes this by:
    1. Down-weighting easy, correctly classified examples
    2. Up-weighting hard, misclassified examples near the decision boundary
    3. Applying per-endpoint alpha weights based on class prevalence

Formula:
    FL(p) = -alpha * (1 - p)^gamma * log(p)

    Where:
        p = predicted probability of the correct class
        alpha = class weight (higher for rare toxic class)
        gamma = focusing parameter (2.0 is standard, from Lin et al. 2017)

NaN Masking:
    Tox21 has 15-25% missing labels per endpoint.
    We use -1 as a sentinel value. The loss function creates a mask
    that zeroes out the loss contribution for any -1 label, so the
    model is never penalized for unknowns.

Reference: Lin et al. (2017) "Focal Loss for Dense Object Detection" ICCV
"""

import torch
import torch.nn as nn
import numpy as np


class PerEndpointFocalLoss(nn.Module):
    """
    Focal Loss applied independently to each of the 12 Tox21 endpoints,
    with per-endpoint alpha weights and NaN masking.

    Args:
        pos_weights: Array of shape (12,) — ratio of negatives to positives per endpoint
                     Higher weight = more penalty for missing a toxic compound
        gamma: Focusing parameter (default 2.0)
               0 = standard weighted BCE
               2 = strong focus on hard examples (recommended)
    """

    def __init__(self, pos_weights: np.ndarray, gamma: float = 2.0):
        super().__init__()
        self.gamma = gamma
        # Store pos_weights as a buffer (moves to GPU with the model)
        self.register_buffer(
            'alpha',
            torch.tensor(pos_weights, dtype=torch.float32)
        )

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Compute Focal Loss with NaN masking.

        Args:
            logits: Raw model outputs, shape (batch, 12). NOT sigmoid-ed.
            targets: True labels, shape (batch, 12). Values: 0, 1, or -1 (missing).

        Returns:
            Scalar loss value (mean over valid entries only)
        """
        # Create mask: 1 for valid labels (0 or 1), 0 for missing (-1)
        mask = (targets >= 0).float()

        # Clamp targets to [0, 1] for BCE computation
        # (the -1 entries will be masked out, so their value doesn't matter)
        targets_clamped = targets.clamp(min=0)

        # Sigmoid to get probabilities
        probs = torch.sigmoid(logits)

        # Standard BCE (element-wise, no reduction)
        bce = nn.functional.binary_cross_entropy_with_logits(
            logits, targets_clamped, reduction='none'
        )

        # Focal modulation factor: (1 - p_t)^gamma
        # p_t = probability of the correct class
        p_t = probs * targets_clamped + (1 - probs) * (1 - targets_clamped)
        focal_weight = (1 - p_t) ** self.gamma

        # Apply per-endpoint alpha weighting
        # alpha is higher for endpoints with fewer toxic samples
        alpha_t = self.alpha * targets_clamped + (1 - targets_clamped)

        # Combine: focal_weight * alpha * BCE, masked for missing labels
        loss = focal_weight * alpha_t * bce * mask

        # Mean over valid entries only (prevent division by zero)
        n_valid = mask.sum()
        if n_valid > 0:
            return loss.sum() / n_valid
        else:
            return loss.sum()  # Edge case: all missing


def compute_pos_weights(df, target_cols: list) -> np.ndarray:
    """
    Compute per-endpoint positive class weights from the dataset.

    Weight = n_negatives / n_positives for each endpoint.
    This tells Focal Loss to penalize missing toxic compounds more heavily.

    Example:
        NR-PPAR-gamma: 189 toxic / 6387 non-toxic → weight = 33.8
        SR-MMP:        936 toxic / 4978 non-toxic → weight = 5.3

    Args:
        df: Cleaned DataFrame with -1 for missing labels
        target_cols: List of 12 endpoint column names

    Returns:
        Array of shape (12,) with positive class weights
    """
    weights = []
    for col in target_cols:
        n_pos = (df[col] == 1).sum()
        n_neg = (df[col] == 0).sum()
        if n_pos > 0:
            w = n_neg / n_pos
        else:
            w = 1.0
        weights.append(w)

    weights = np.array(weights, dtype=np.float32)

    print("Per-endpoint positive weights (n_neg / n_pos):")
    for col, w in zip(target_cols, weights):
        print(f"  {col:20s}  weight={w:.1f}")

    return weights


# ── Run when called directly (for quick testing) ──────────────────
if __name__ == '__main__':
    import sys
    sys.path.insert(0, '.')
    from src.featurize import load_and_clean_tox21, TARGET_COLS

    df = load_and_clean_tox21()

    print("\n--- Phase 3: Focal Loss Test ---\n")

    # Compute weights from data
    pos_weights = compute_pos_weights(df, TARGET_COLS)

    # Create loss function
    loss_fn = PerEndpointFocalLoss(pos_weights, gamma=2.0)

    # Test with dummy data
    batch_size = 32
    dummy_logits = torch.randn(batch_size, 12)
    dummy_targets = torch.randint(0, 2, (batch_size, 12)).float()

    # Add some -1 (missing) values to test masking
    dummy_targets[0, 3] = -1
    dummy_targets[1, 7] = -1
    dummy_targets[2, 0] = -1

    loss = loss_fn(dummy_logits, dummy_targets)
    print(f"\nFocal Loss (gamma=2.0): {loss.item():.4f}")
    print(f"Loss is scalar: {loss.dim() == 0}")
    print(f"Loss is not NaN: {not torch.isnan(loss).item()}")
    print(f"Loss is positive: {loss.item() > 0}")

    # Compare with gamma=0 (should behave like weighted BCE)
    loss_fn_bce = PerEndpointFocalLoss(pos_weights, gamma=0.0)
    loss_bce = loss_fn_bce(dummy_logits, dummy_targets)
    print(f"\nWeighted BCE (gamma=0):  {loss_bce.item():.4f}")
    print(f"Focal Loss (gamma=2.0): {loss.item():.4f}")
    print(f"Focal < BCE: {loss.item() < loss_bce.item()} (expected: focal down-weights easy examples)")
