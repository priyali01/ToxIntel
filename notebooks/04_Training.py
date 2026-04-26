# %% [markdown]
# # 04 — ToxNet Training with Focal Loss + OPTUNA
# 
# **Phase 4 of plan.md**
# 
# This notebook:
# 1. Loads featurized data with scaffold split
# 2. Trains ToxNet (multi-task neural network) with Focal Loss
# 3. Uses OPTUNA to find optimal hyperparameters
# 4. Saves the best model to `models/toxnet_final.pt`

# %%
import sys
import os
# Change working directory to project root
if os.path.basename(os.getcwd()) == 'notebooks':
    os.chdir('..')
sys.path.insert(0, '.')

import torch
import numpy as np
import matplotlib.pyplot as plt
from src.featurize import TARGET_COLS
from src.model import ToxNet
from src.train import prepare_data, train_model, run_optuna_search

# %%
# Prepare data (load, featurize, scaffold split)
X_train, Y_train, X_val, Y_val, X_test, Y_test, pos_weights = prepare_data()

# %% [markdown]
# ## 1. ToxNet Architecture
# 
# ```
# Input (2048) → Shared Backbone [1024→512→256] → 12 Task Heads [256→64→1]
# ```
# 
# - Shared backbone learns common chemistry patterns
# - Each head specializes in one toxicity endpoint

# %%
# Show model architecture
model = ToxNet(input_dim=2048)
total_params = sum(p.numel() for p in model.parameters())
print(f"ToxNet Architecture:")
print(f"  Total parameters: {total_params:,}")
print(f"  Input: 2048 (ECFP4 fingerprint)")
print(f"  Output: 12 (one per Tox21 endpoint)")
print(f"\nArchitecture:\n{model}")

# %% [markdown]
# ## 2. Quick Training (Default Hyperparameters)
# 
# First, a quick 20-epoch training to verify everything works.

# %%
# Quick training (20 epochs)
print("Training ToxNet (20 epochs, default params)...")
model, val_auprc, history = train_model(
    X_train, Y_train, X_val, Y_val, pos_weights,
    n_epochs=20, patience=10, verbose=True,
)
print(f"\nQuick training result: val AUPRC = {val_auprc:.4f}")

# %%
# Plot training history
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

axes[0].plot(history['train_loss'], color='#e74c3c')
axes[0].set_xlabel('Epoch')
axes[0].set_ylabel('Focal Loss')
axes[0].set_title('Training Loss')

axes[1].plot(history['val_auprc'], color='#2ecc71')
axes[1].set_xlabel('Epoch')
axes[1].set_ylabel('Macro AUPRC')
axes[1].set_title('Validation AUPRC')
axes[1].axhline(y=max(history['val_auprc']), color='gray', linestyle='--', alpha=0.5)

plt.tight_layout()
plt.savefig('notebooks/04_training_history.png', dpi=150, bbox_inches='tight')
plt.show()
print("Saved: notebooks/04_training_history.png")

# %% [markdown]
# ## 3. OPTUNA Hyperparameter Search
# 
# Tuning: lr, dropout, batch_size, focal_gamma, hidden_dims, weight_decay
# 
# Set n_trials=50 for full search, or n_trials=10 for a quick test.

# %%
# OPTUNA search (set n_trials=50 for full run, 10 for quick test)
N_TRIALS = 10  # Change to 50 for full search

best_params = run_optuna_search(
    X_train, Y_train, X_val, Y_val, pos_weights,
    n_trials=N_TRIALS,
)
print(f"\nBest params: {best_params}")

# %% [markdown]
# ## 4. Final Training with Best Parameters

# %%
# Train with best params
hidden_dims = {
    'small': [512, 256, 128],
    'medium': [1024, 512, 256],
    'large': [1024, 512, 512, 256],
}[best_params['hidden_dims']]

print("Training with OPTUNA-optimized parameters...")
model_final, final_auprc, final_history = train_model(
    X_train, Y_train, X_val, Y_val, pos_weights,
    lr=best_params['lr'],
    dropout=best_params['dropout'],
    batch_size=best_params['batch_size'],
    gamma=best_params['gamma'],
    hidden_dims=hidden_dims,
    weight_decay=best_params['weight_decay'],
    n_epochs=50,
    patience=15,
    verbose=True,
)

print(f"\nFinal val AUPRC: {final_auprc:.4f}")

# %%
# Save final model
import os
os.makedirs('models', exist_ok=True)
torch.save(model_final.state_dict(), 'models/toxnet_final.pt')
print("Model saved to models/toxnet_final.pt")

# Plot final training history
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
axes[0].plot(final_history['train_loss'], color='#e74c3c')
axes[0].set_title('Final Training Loss')
axes[0].set_xlabel('Epoch')
axes[1].plot(final_history['val_auprc'], color='#2ecc71')
axes[1].set_title(f'Final Val AUPRC (best: {final_auprc:.4f})')
axes[1].set_xlabel('Epoch')
plt.tight_layout()
plt.savefig('notebooks/04_final_training.png', dpi=150, bbox_inches='tight')
plt.show()

# %% [markdown]
# ## 5. Summary
# 
# | Metric | Value |
# |--------|-------|
# | Quick Training AUPRC | see above |
# | OPTUNA-Optimized AUPRC | see above |
# | Target (plan.md) | 0.42 - 0.52 |
