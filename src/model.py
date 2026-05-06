"""
src/model.py — ToxNet v2: Multi-Task Neural Network for Tox21

Changes from v1:
    1. Larger task heads: 256→128→64→1 instead of 256→64→1
       Endpoints like NR-PPAR-gamma (0.5% prevalence) need more head
       capacity to learn from very few positive examples.

    2. Residual connections in backbone: helps gradient flow through
       deep layers, especially important early in training when
       imbalance means gradients from positives are rare.

    3. He initialization on all Linear layers: critical for GELU
       activations with BatchNorm — prevents vanishing gradients
       in early epochs before the model sees enough positives.

Architecture:
    Shared Backbone (with residuals):
        Linear(2048 → 1024) → BN → GELU → Dropout
        Linear(1024 → 512)  → BN → GELU → Dropout  [+ residual projection]
        Linear(512  → 256)  → BN → GELU → Dropout

    12 Independent Task Heads:
        Linear(256 → 128) → GELU → Dropout
        Linear(128 → 64)  → GELU → Dropout
        Linear(64  → 1)
"""

import torch
import torch.nn as nn
import os


class ResidualBlock(nn.Module):
    """Single residual block: Linear → BN → GELU → Dropout, with skip connection."""

    def __init__(self, in_dim: int, out_dim: int, dropout: float = 0.3):
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.BatchNorm1d(out_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        # Project residual if dimensions differ
        self.skip = nn.Linear(in_dim, out_dim, bias=False) if in_dim != out_dim else nn.Identity()
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        return self.block(x) + self.skip(x)


class ToxNetLite(nn.Module):
    """
    ToxNetLite: The shared backbone 'expert' that learns chemical embeddings.
    This module is used in the first pass of the two-pass training strategy.
    """
    def __init__(self, input_dim: int = 2048, hidden_dims: list = None, dropout: float = 0.3):
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [1024, 512, 256]
        
        backbone_layers = []
        prev_dim = input_dim
        for dim in hidden_dims:
            backbone_layers.append(ResidualBlock(prev_dim, dim, dropout=dropout))
            prev_dim = dim
            
        self.backbone = nn.Sequential(*backbone_layers)
        self.output_dim = hidden_dims[-1]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)


class ToxNet(nn.Module):
    """
    Multi-task neural network for simultaneous 12-endpoint toxicity prediction.
    Utilizes a ToxNetLite backbone for feature extraction.

    Args:
        backbone:     An instance of ToxNetLite or None (to create a new one)
        head_hidden:  First hidden size in each task head
        head_dropout: Dropout rate for task heads
        n_tasks:      Number of output endpoints
    """

    def __init__(self, backbone: ToxNetLite = None,
                 input_dim: int = 2048,
                 hidden_dims: list = None,
                 head_hidden: int = 128,
                 dropout: float = 0.3,
                 head_dropout: float = 0.1,
                 n_tasks: int = 12):
        super().__init__()

        # Use provided backbone or create a new one
        if backbone is not None:
            self.lite = backbone
        else:
            self.lite = ToxNetLite(input_dim=input_dim, hidden_dims=hidden_dims, dropout=dropout)

        # ── Task Heads — larger than v1 ──
        # embedding_dim → head_hidden → head_hidden//2 → 1
        embedding_dim = self.lite.output_dim
        head_mid = max(head_hidden // 2, 32)
        self.heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(embedding_dim, head_hidden),
                nn.GELU(),
                nn.Dropout(head_dropout),
                nn.Linear(head_hidden, head_mid),
                nn.GELU(),
                nn.Dropout(head_dropout),
                nn.Linear(head_mid, 1),
            )
            for _ in range(n_tasks)
        ])

        # Initialize heads
        for head in self.heads:
            for m in head.modules():
                if isinstance(m, nn.Linear):
                    nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)

        self.n_tasks = n_tasks

    def freeze_backbone(self, freeze: bool = True):
        """Freezes/unfreezes the backbone for the second pass of training."""
        for param in self.lite.parameters():
            param.requires_grad = not freeze

    def get_embeddings(self, x: torch.Tensor) -> torch.Tensor:
        """Extract shared backbone embeddings (before task heads)."""
        return self.lite(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass: fingerprint → 12 endpoint logits."""
        embedding = self.lite(x)
        logits = torch.cat([head(embedding) for head in self.heads], dim=1)
        return logits


def load_toxnet(filepath='models/toxnet_final.pt', input_dim=2048):
    """
    Robustly loads ToxNet by trying all possible hidden_dims configurations
    that Optuna might have selected.
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Model file {filepath} not found.")

    state_dict = torch.load(filepath, map_location='cpu', weights_only=True)

    configs = [
        [1024, 512, 256],       # medium (default)
        [512, 256, 128],        # small
        [1024, 512, 512, 256],  # large
    ]

    for dims in configs:
        for head_hidden in [128, 64]:  # try both v2 and v1 head sizes
            try:
                model = ToxNet(input_dim=input_dim, hidden_dims=dims, head_hidden=head_hidden)
                model.load_state_dict(state_dict)
                return model
            except Exception:
                continue

    raise RuntimeError(
        "Failed to load model. None of the expected configurations matched. "
        "Re-train the model with the current model.py."
    )


# ── Quick test ─────────────────────────────────────────────────────
if __name__ == '__main__':
    print("--- ToxNet v2 Architecture Test ---\n")

    model = ToxNet(input_dim=2048)

    total_params = sum(p.numel() for p in model.parameters())
    backbone_params = sum(p.numel() for p in model.lite.backbone.parameters())
    head_params = total_params - backbone_params

    print(f"Total parameters:    {total_params:,}")
    print(f"Backbone parameters: {backbone_params:,}")
    print(f"Head parameters:     {head_params:,} ({head_params/total_params*100:.1f}%)")

    batch_size = 16
    test_input = torch.randn(batch_size, 2048)
    output = model(test_input)

    assert output.shape == (batch_size, 12), f"Wrong shape: {output.shape}"
    print(f"\nForward pass shape: {output.shape} — PASSED")

    emb = model.get_embeddings(test_input)
    assert emb.shape == (batch_size, 256), f"Wrong embedding shape: {emb.shape}"
    print(f"Embedding shape:    {emb.shape} — PASSED")

    assert output.min() < 0, "Logits should contain negatives (not sigmoid-ed)"
    print("Logits are raw (not sigmoid-ed) — PASSED")

    print("\nAll ToxNet v2 tests PASSED")