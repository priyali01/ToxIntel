"""
src/model.py — ToxNet: Multi-Task Neural Network for Tox21

Architecture:
    Shared Backbone:
        Linear(input_dim → 1024) → BatchNorm → GELU → Dropout
        Linear(1024 → 512)       → BatchNorm → GELU → Dropout
        Linear(512 → 256)        → BatchNorm → GELU → Dropout

    12 Independent Task Heads (one per Tox21 endpoint):
        Linear(256 → 64) → ReLU → Dropout → Linear(64 → 1)

Why multi-task with shared backbone?
    All 12 Tox21 endpoints share common chemical features (ring systems,
    functional groups, etc.). The shared backbone learns these once,
    while each head specializes in its own endpoint.
    This is more parameter-efficient and often more accurate than
    training 12 separate models.

Why GELU over ReLU?
    GELU (Gaussian Error Linear Unit) provides smoother gradients than ReLU,
    which helps with the imbalanced, noisy Tox21 data. Used in modern
    architectures like BERT and GPT.
"""

import torch
import torch.nn as nn


class ToxNet(nn.Module):
    """
    Multi-task neural network for simultaneous 12-endpoint toxicity prediction.

    Args:
        input_dim: Size of input fingerprint (default 2048 for ECFP4)
        hidden_dims: List of hidden layer sizes for shared backbone
        head_hidden: Hidden size in each task head
        dropout: Dropout rate for shared backbone
        head_dropout: Dropout rate for task heads
        n_tasks: Number of output endpoints (default 12 for Tox21)
    """

    def __init__(self, input_dim: int = 2048,
                 hidden_dims: list = None,
                 head_hidden: int = 64,
                 dropout: float = 0.3,
                 head_dropout: float = 0.15,
                 n_tasks: int = 12):
        super().__init__()

        if hidden_dims is None:
            hidden_dims = [1024, 512, 256]

        # ── Shared Backbone ──
        # All 12 endpoints share these layers to learn common chemistry
        layers = []
        prev_dim = input_dim
        for dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, dim),
                nn.BatchNorm1d(dim),
                nn.GELU(),
                nn.Dropout(dropout),
            ])
            prev_dim = dim

        self.backbone = nn.Sequential(*layers)

        # ── Task Heads ──
        # Each endpoint gets its own small classifier
        self.heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dims[-1], head_hidden),
                nn.ReLU(),
                nn.Dropout(head_dropout),
                nn.Linear(head_hidden, 1),
            )
            for _ in range(n_tasks)
        ])

        self.n_tasks = n_tasks

    def get_embeddings(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extract the shared backbone embeddings (before the task heads).

        Used for:
            - SMOTE in embedding space (Phase 3)
            - t-SNE visualization
            - Nearest-neighbor OOD detection (Phase 6)

        Returns:
            Embedding tensor of shape (batch, hidden_dims[-1])
        """
        return self.backbone(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass: fingerprint → 12 endpoint logits.

        Args:
            x: Input tensor of shape (batch, input_dim)

        Returns:
            Logits tensor of shape (batch, n_tasks). NOT sigmoid-ed.
            Apply sigmoid yourself for probabilities.
        """
        # Shared feature extraction
        embedding = self.backbone(x)

        # Each head produces 1 logit for its endpoint
        logits = torch.cat(
            [head(embedding) for head in self.heads],
            dim=1
        )

        return logits

def load_toxnet(filepath='models/toxnet_final.pt', input_dim=2048):
    """
    Robustly loads ToxNet by trying the three possible hidden_dims configurations 
    that OPTUNA might have selected in Phase 4.
    """
    import torch
    import os
    
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Model file {filepath} not found.")
        
    state_dict = torch.load(filepath, map_location='cpu', weights_only=True)
    
    # The possible OPTUNA configurations from train.py
    configs = [
        [1024, 512, 256],          # medium (default)
        [512, 256, 128],           # small
        [1024, 512, 512, 256]      # large
    ]
    
    for dims in configs:
        try:
            model = ToxNet(input_dim=input_dim, hidden_dims=dims)
            model.load_state_dict(state_dict)
            return model
        except Exception:
            continue
            
    raise RuntimeError("Failed to load model state_dict. None of the expected hidden_dims configurations matched.")

# ── Run when called directly (for quick testing) ──────────────────
if __name__ == '__main__':
    print("--- Phase 4: ToxNet Architecture Test ---\n")

    # Create model
    model = ToxNet(input_dim=2048)

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    backbone_params = sum(p.numel() for p in model.backbone.parameters())
    head_params = total_params - backbone_params

    print(f"Total parameters:    {total_params:,}")
    print(f"Backbone parameters: {backbone_params:,}")
    print(f"Head parameters:     {head_params:,} ({head_params/total_params*100:.1f}%)")

    # Test forward pass
    batch_size = 16
    test_input = torch.randn(batch_size, 2048)
    output = model(test_input)
    print(f"\nInput shape:  {test_input.shape}")
    print(f"Output shape: {output.shape}")
    assert output.shape == (batch_size, 12), f"Wrong shape: {output.shape}"
    print("Forward pass: PASSED")

    # Test embeddings
    emb = model.get_embeddings(test_input)
    print(f"Embedding shape: {emb.shape}")
    assert emb.shape == (batch_size, 256), f"Wrong shape: {emb.shape}"
    print("Embeddings: PASSED")

    # Test that logits are raw (not sigmoid-ed)
    assert output.min() < 0, "Logits should contain negative values (not sigmoid-ed)"
    print("Logits are raw: PASSED")

    print("\nAll ToxNet tests PASSED")
