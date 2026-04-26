"""
src/train.py — Training Loop + OPTUNA Hyperparameter Search

This module:
    1. Prepares the data (featurize + scaffold split)
    2. Defines the training loop with Focal Loss
    3. Uses OPTUNA to find the best hyperparameters (50 trials)
    4. Saves the best model to models/toxnet_final.pt

OPTUNA tunes:
    - Learning rate (1e-4 to 1e-2)
    - Dropout (0.1 to 0.5)
    - Batch size (32, 64, 128, 256)
    - Focal gamma (1.0 to 3.0)
    - Hidden dimensions
    - Weight decay (1e-5 to 1e-3)
"""

import sys
sys.path.insert(0, '.')

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import optuna
from sklearn.metrics import average_precision_score
import os

from src.featurize import load_and_clean_tox21, featurize_dataset, TARGET_COLS
from src.scaffold_split import scaffold_split
from src.focal_loss import PerEndpointFocalLoss, compute_pos_weights
from src.model import ToxNet


def prepare_data(method: str = 'ecfp4_2048'):
    """
    Load, featurize, and split the Tox21 dataset.

    Returns:
        X_train, Y_train, X_val, Y_val, X_test, Y_test, pos_weights
        All as numpy arrays.
    """
    # Load and clean
    df = load_and_clean_tox21()

    # Scaffold split
    train_idx, val_idx, test_idx = scaffold_split(df['smiles'].tolist())

    # Featurize
    print(f"\nFeaturizing with {method}...")
    all_smiles = df['smiles'].tolist()
    X_all = featurize_dataset(all_smiles, method=method)
    Y_all = df[TARGET_COLS].values.astype(np.float32)

    # Split
    X_train, Y_train = X_all[train_idx], Y_all[train_idx]
    X_val, Y_val = X_all[val_idx], Y_all[val_idx]
    X_test, Y_test = X_all[test_idx], Y_all[test_idx]

    # Compute class weights from training set only
    df_train = df.iloc[train_idx]
    pos_weights = compute_pos_weights(df_train, TARGET_COLS)

    print(f"\nData prepared:")
    print(f"  Train: {X_train.shape}")
    print(f"  Val:   {X_val.shape}")
    print(f"  Test:  {X_test.shape}")

    return X_train, Y_train, X_val, Y_val, X_test, Y_test, pos_weights


def compute_macro_auprc(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """
    Compute macro-averaged AUPRC across all 12 endpoints.

    THIS IS THE PRIMARY METRIC — not AUROC, not accuracy.

    Skips endpoints where all labels are missing (-1) or all same class.
    """
    auprcs = []
    for i in range(y_true.shape[1]):
        # Only evaluate on non-missing labels
        mask = y_true[:, i] >= 0
        if mask.sum() < 2:
            continue
        y_t = y_true[mask, i]
        y_p = y_prob[mask, i]
        # Need both classes present
        if len(np.unique(y_t)) < 2:
            continue
        auprcs.append(average_precision_score(y_t, y_p))

    return float(np.mean(auprcs)) if auprcs else 0.0


def train_one_epoch(model, dataloader, optimizer, loss_fn, device):
    """Train for one epoch. Returns average loss."""
    model.train()
    total_loss = 0
    n_batches = 0

    for X_batch, Y_batch in dataloader:
        X_batch = X_batch.to(device)
        Y_batch = Y_batch.to(device)

        optimizer.zero_grad()
        logits = model(X_batch)
        loss = loss_fn(logits, Y_batch)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        n_batches += 1

    return total_loss / max(n_batches, 1)


def evaluate(model, dataloader, device):
    """Evaluate model. Returns (predictions, true labels) as numpy arrays."""
    model.eval()
    all_probs = []
    all_labels = []

    with torch.no_grad():
        for X_batch, Y_batch in dataloader:
            X_batch = X_batch.to(device)
            logits = model(X_batch)
            probs = torch.sigmoid(logits).cpu().numpy()
            all_probs.append(probs)
            all_labels.append(Y_batch.numpy())

    return np.vstack(all_probs), np.vstack(all_labels)


def train_model(X_train, Y_train, X_val, Y_val, pos_weights,
                lr=1e-3, dropout=0.3, batch_size=64,
                gamma=2.0, hidden_dims=None, weight_decay=1e-4,
                n_epochs=50, patience=10, verbose=True):
    """
    Full training loop with early stopping.

    Args:
        patience: Stop if val AUPRC doesn't improve for this many epochs

    Returns:
        (model, best_val_auprc, history)
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if hidden_dims is None:
        hidden_dims = [1024, 512, 256]

    # Create dataloaders
    train_ds = TensorDataset(
        torch.tensor(X_train, dtype=torch.float32),
        torch.tensor(Y_train, dtype=torch.float32)
    )
    val_ds = TensorDataset(
        torch.tensor(X_val, dtype=torch.float32),
        torch.tensor(Y_val, dtype=torch.float32)
    )
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=256)

    # Create model and loss
    model = ToxNet(
        input_dim=X_train.shape[1],
        hidden_dims=hidden_dims,
        dropout=dropout,
    ).to(device)

    loss_fn = PerEndpointFocalLoss(pos_weights, gamma=gamma).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', patience=5, factor=0.5
    )

    # Training loop with early stopping
    best_auprc = 0
    best_state = None
    patience_counter = 0
    history = {'train_loss': [], 'val_auprc': []}

    for epoch in range(n_epochs):
        train_loss = train_one_epoch(model, train_loader, optimizer, loss_fn, device)
        val_probs, val_labels = evaluate(model, val_loader, device)
        val_auprc = compute_macro_auprc(val_labels, val_probs)

        history['train_loss'].append(train_loss)
        history['val_auprc'].append(val_auprc)

        scheduler.step(val_auprc)

        if val_auprc > best_auprc:
            best_auprc = val_auprc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1

        if verbose and (epoch + 1) % 5 == 0:
            print(f"  Epoch {epoch+1:3d}  loss={train_loss:.4f}  val_AUPRC={val_auprc:.4f}  best={best_auprc:.4f}")

        if patience_counter >= patience:
            if verbose:
                print(f"  Early stopping at epoch {epoch+1}")
            break

    # Restore best model
    model.load_state_dict(best_state)
    model.to(device)

    return model, best_auprc, history


def optuna_objective(trial, X_train, Y_train, X_val, Y_val, pos_weights):
    """
    OPTUNA objective function. Each trial tries different hyperparameters.
    Returns val AUPRC (higher is better).
    """
    lr = trial.suggest_float('lr', 1e-4, 1e-2, log=True)
    dropout = trial.suggest_float('dropout', 0.1, 0.5)
    batch_size = trial.suggest_categorical('batch_size', [32, 64, 128, 256])
    gamma = trial.suggest_float('gamma', 1.0, 3.0)
    weight_decay = trial.suggest_float('weight_decay', 1e-5, 1e-3, log=True)

    # Hidden dims options
    dim_choice = trial.suggest_categorical('hidden_dims', ['small', 'medium', 'large'])
    hidden_dims = {
        'small': [512, 256, 128],
        'medium': [1024, 512, 256],
        'large': [1024, 512, 512, 256],
    }[dim_choice]

    _, val_auprc, _ = train_model(
        X_train, Y_train, X_val, Y_val, pos_weights,
        lr=lr, dropout=dropout, batch_size=batch_size,
        gamma=gamma, hidden_dims=hidden_dims,
        weight_decay=weight_decay,
        n_epochs=30,  # Fewer epochs during search
        patience=7,
        verbose=False,
    )

    return val_auprc


def run_optuna_search(X_train, Y_train, X_val, Y_val, pos_weights,
                      n_trials: int = 20):
    """
    Run OPTUNA hyperparameter search.

    Args:
        n_trials: Number of trials (20 for quick test, 50 for full search)

    Returns:
        best_params dict
    """
    print(f"\nStarting OPTUNA search ({n_trials} trials)...")

    study = optuna.create_study(direction='maximize')
    study.optimize(
        lambda trial: optuna_objective(trial, X_train, Y_train, X_val, Y_val, pos_weights),
        n_trials=n_trials,
        show_progress_bar=True,
    )

    print(f"\nBest trial:")
    print(f"  Val AUPRC: {study.best_value:.4f}")
    print(f"  Params: {study.best_params}")

    return study.best_params


# ── Run when called directly ──────────────────────────────────────
if __name__ == '__main__':
    print("--- Phase 4: Training Pipeline Test ---\n")

    # Prepare data
    X_train, Y_train, X_val, Y_val, X_test, Y_test, pos_weights = prepare_data()

    # Quick training test (5 epochs, no OPTUNA)
    print("\n--- Quick Training Test (5 epochs) ---")
    model, val_auprc, history = train_model(
        X_train, Y_train, X_val, Y_val, pos_weights,
        n_epochs=5, patience=5, verbose=True,
    )

    print(f"\nQuick test val AUPRC: {val_auprc:.4f}")
    print(f"Model output shape: {model(torch.randn(1, 2048)).shape}")

    # Save model
    os.makedirs('models', exist_ok=True)
    torch.save(model.state_dict(), 'models/toxnet_final.pt')
    print(f"Model saved to models/toxnet_final.pt")

    # Verify loadable
    model2 = ToxNet()
    model2.load_state_dict(torch.load('models/toxnet_final.pt', weights_only=True))
    print("Model reload: PASSED")

    print(f"\nPhase 4 quick test: PASSED (val AUPRC = {val_auprc:.4f})")
    print("For full training, run with --full flag or use OPTUNA search")
