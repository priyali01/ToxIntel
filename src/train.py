"""
src/train.py — Training Loop + OPTUNA Hyperparameter Search (v2)

Key fixes from v1:
    1. LR warmup (10 epochs): ramps from lr/100 → lr linearly before
       handing off to cosine annealing. Prevents unstable early gradients
       from BatchNorm + heavy class imbalance.

    2. Longer Optuna trials (50 epochs, patience=12): the model often
       doesn't produce meaningful toxic predictions until epoch 15-25
       with Tox21 imbalance. v1's 30 epochs / patience=7 was terminating
       before models had time to learn.

    3. CosineAnnealingLR instead of ReduceLROnPlateau: more stable
       decay schedule, less sensitive to noisy val AUPRC fluctuations
       on small imbalanced validation sets.

    4. Gradient clipping (max_norm=1.0): prevents occasional exploding
       gradients when a batch happens to contain all positives.

    5. Optuna search space expanded: head_hidden now tunable (64 vs 128),
       label_smoothing included (0.0 vs 0.05).
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
import math

from src.featurize import load_and_clean_tox21, featurize_dataset, TARGET_COLS
from src.scaffold_split import scaffold_split
from src.focal_loss import PerEndpointFocalLoss, compute_pos_weights
from src.model import ToxNet

optuna.logging.set_verbosity(optuna.logging.WARNING)


def prepare_data(method: str = 'ecfp4_2048'):
    """
    Load, featurize, and split the Tox21 dataset.

    Supports all featurize methods including 'ecfp4_desc' (ECFP4 + RDKit
    descriptors). For 'ecfp4_desc', the descriptor scaler is fit on train
    only and applied to val/calib/test to prevent leakage.

    Returns:
        X_train, Y_train, X_val, Y_val, X_calib, Y_calib,
        X_test, Y_test, pos_weights, smiles_calib
    """
    df = load_and_clean_tox21()

    train_idx, val_idx, calib_idx, test_idx = scaffold_split(
        df['smiles'].tolist(),
        train_frac=0.7, val_frac=0.1, calib_frac=0.1, test_frac=0.1
    )

    print(f"\nFeaturizing with {method}...")
    all_smiles  = df['smiles'].tolist()
    Y_all       = df[TARGET_COLS].values.astype(np.float32)

    train_smiles = [all_smiles[i] for i in train_idx]
    val_smiles   = [all_smiles[i] for i in val_idx]
    calib_smiles = [all_smiles[i] for i in calib_idx]
    test_smiles  = [all_smiles[i] for i in test_idx]

    if method == 'ecfp4_desc':
        # Fit scaler on train, apply to all splits — no leakage
        X_train, scaler = featurize_dataset(train_smiles, method=method, fit_scaler=True)
        X_val   = featurize_dataset(val_smiles,   method=method, scaler=scaler)
        X_calib = featurize_dataset(calib_smiles, method=method, scaler=scaler)
        X_test  = featurize_dataset(test_smiles,  method=method, scaler=scaler)
    else:
        X_train = featurize_dataset(train_smiles, method=method)
        X_val   = featurize_dataset(val_smiles,   method=method)
        X_calib = featurize_dataset(calib_smiles, method=method)
        X_test  = featurize_dataset(test_smiles,  method=method)

    Y_train = Y_all[train_idx]
    Y_val   = Y_all[val_idx]
    Y_calib = Y_all[calib_idx]
    Y_test  = Y_all[test_idx]

    smiles_calib = calib_smiles

    df_train    = df.iloc[train_idx]
    pos_weights = compute_pos_weights(df_train, TARGET_COLS)

    print(f"\nData prepared:")
    print(f"  Train: {X_train.shape}")
    print(f"  Val:   {X_val.shape}")
    print(f"  Calib: {X_calib.shape}")
    print(f"  Test:  {X_test.shape}")

    return (X_train, Y_train, X_val, Y_val,
            X_calib, Y_calib, X_test, Y_test,
            pos_weights, smiles_calib)


def compute_macro_auprc(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """
    Compute macro-averaged AUPRC across all 12 endpoints.
    Skips endpoints with missing labels or single class present.
    """
    auprcs = []
    for i in range(y_true.shape[1]):
        mask = y_true[:, i] >= 0
        if mask.sum() < 2:
            continue
        y_t = y_true[mask, i]
        y_p = y_prob[mask, i]
        if len(np.unique(y_t)) < 2:
            continue
        auprcs.append(average_precision_score(y_t, y_p))
    return float(np.mean(auprcs)) if auprcs else 0.0


def get_warmup_cosine_scheduler(optimizer, warmup_epochs: int, total_epochs: int):
    """
    Linear warmup for `warmup_epochs`, then cosine annealing to 0.

    Warmup: lr scales from lr/100 → lr over warmup_epochs.
    This prevents early instability from BatchNorm + heavy imbalance.
    """
    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            return (epoch + 1) / warmup_epochs  # linear ramp: 1/W, 2/W, ..., 1.0
        # Cosine decay after warmup
        progress = (epoch - warmup_epochs) / max(total_epochs - warmup_epochs, 1)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def train_one_epoch(model, dataloader, optimizer, loss_fn, device, max_grad_norm=1.0):
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

        # Gradient clipping: prevents occasional exploding gradients
        # when a batch is skewed toward all positives
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)

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
                head_hidden=128, label_smoothing=0.05,
                n_epochs=100, patience=15, warmup_epochs=10,
                verbose=True):
    """
    Full training loop with LR warmup + cosine annealing + early stopping.

    Key change from v1: patience increased to 15, warmup added,
    scheduler changed from ReduceLROnPlateau to warmup+cosine.

    Returns:
        (model, best_val_auprc, history)
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if hidden_dims is None:
        hidden_dims = [1024, 512, 256]

    train_ds = TensorDataset(
        torch.tensor(X_train, dtype=torch.float32),
        torch.tensor(Y_train, dtype=torch.float32)
    )
    val_ds = TensorDataset(
        torch.tensor(X_val, dtype=torch.float32),
        torch.tensor(Y_val, dtype=torch.float32)
    )
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader   = DataLoader(val_ds,   batch_size=256)

    model = ToxNet(
        input_dim=X_train.shape[1],
        hidden_dims=hidden_dims,
        dropout=dropout,
        head_hidden=head_hidden,
    ).to(device)

    loss_fn = PerEndpointFocalLoss(
        pos_weights, gamma=gamma, label_smoothing=label_smoothing
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=lr, weight_decay=weight_decay
    )
    scheduler = get_warmup_cosine_scheduler(optimizer, warmup_epochs, n_epochs)

    best_auprc = 0.0
    best_state = None
    patience_counter = 0
    history = {'train_loss': [], 'val_auprc': [], 'lr': []}

    for epoch in range(n_epochs):
        current_lr = optimizer.param_groups[0]['lr']
        train_loss = train_one_epoch(model, train_loader, optimizer, loss_fn, device)
        val_probs, val_labels = evaluate(model, val_loader, device)
        val_auprc = compute_macro_auprc(val_labels, val_probs)

        scheduler.step()

        history['train_loss'].append(train_loss)
        history['val_auprc'].append(val_auprc)
        history['lr'].append(current_lr)

        if val_auprc > best_auprc:
            best_auprc = val_auprc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1

        if verbose and (epoch + 1) % 10 == 0:
            print(f"  Epoch {epoch+1:3d}  loss={train_loss:.4f}  "
                  f"val_AUPRC={val_auprc:.4f}  best={best_auprc:.4f}  "
                  f"lr={current_lr:.2e}")

        if patience_counter >= patience:
            if verbose:
                print(f"  Early stopping at epoch {epoch+1} "
                      f"(no improvement for {patience} epochs)")
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    model.to(device)

    return model, best_auprc, history


def optuna_objective(trial, X_train, Y_train, X_val, Y_val, pos_weights):
    """
    Optuna objective. Each trial tries different hyperparameters.
    Returns val AUPRC (higher is better).
    """
    lr             = trial.suggest_float('lr', 1e-4, 5e-3, log=True)
    dropout        = trial.suggest_float('dropout', 0.1, 0.5)
    batch_size     = trial.suggest_categorical('batch_size', [64, 128, 256])
    gamma          = trial.suggest_float('gamma', 1.5, 3.0)
    weight_decay   = trial.suggest_float('weight_decay', 1e-5, 5e-4, log=True)
    head_hidden    = trial.suggest_categorical('head_hidden', [64, 128])
    label_smoothing = trial.suggest_categorical('label_smoothing', [0.0, 0.05])

    dim_choice = trial.suggest_categorical('hidden_dims', ['small', 'medium', 'large'])
    hidden_dims = {
        'small':  [512, 256, 128],
        'medium': [1024, 512, 256],
        'large':  [1024, 512, 512, 256],
    }[dim_choice]

    _, val_auprc, _ = train_model(
        X_train, Y_train, X_val, Y_val, pos_weights,
        lr=lr, dropout=dropout, batch_size=batch_size,
        gamma=gamma, hidden_dims=hidden_dims,
        weight_decay=weight_decay, head_hidden=head_hidden,
        label_smoothing=label_smoothing,
        n_epochs=50,      # was 30 — needs 15-25 epochs just to stabilize
        patience=12,      # was 7 — give model time past warmup
        warmup_epochs=8,
        verbose=False,
    )

    return val_auprc


def run_optuna_search(X_train, Y_train, X_val, Y_val, pos_weights,
                      n_trials: int = 30):
    """
    Run Optuna hyperparameter search.

    Args:
        n_trials: 20 for quick test, 50 for full search.

    Returns:
        best_params dict
    """
    print(f"\nStarting Optuna search ({n_trials} trials)...")

    study = optuna.create_study(
        direction='maximize',
        sampler=optuna.samplers.TPESampler(seed=42),
    )
    study.optimize(
        lambda trial: optuna_objective(
            trial, X_train, Y_train, X_val, Y_val, pos_weights
        ),
        n_trials=n_trials,
        show_progress_bar=True,
    )

    print(f"\nBest trial:")
    print(f"  Val AUPRC: {study.best_value:.4f}")
    print(f"  Params:    {study.best_params}")

    return study.best_params


def run_full_training(best_params, X_train, Y_train, X_val, Y_val,
                      pos_weights, save_path='models/toxnet_final.pt'):
    """
    Train final model with best Optuna params for full epochs.
    Saves model weights to save_path.

    Returns:
        (model, final_val_auprc, history)
    """
    hidden_dims = {
        'small':  [512, 256, 128],
        'medium': [1024, 512, 256],
        'large':  [1024, 512, 512, 256],
    }[best_params.get('hidden_dims', 'medium')]

    print("\nTraining final model with best params...")
    model, val_auprc, history = train_model(
        X_train, Y_train, X_val, Y_val, pos_weights,
        lr=best_params.get('lr', 1e-3),
        dropout=best_params.get('dropout', 0.3),
        batch_size=best_params.get('batch_size', 128),
        gamma=best_params.get('gamma', 2.0),
        hidden_dims=hidden_dims,
        weight_decay=best_params.get('weight_decay', 1e-4),
        head_hidden=best_params.get('head_hidden', 128),
        label_smoothing=best_params.get('label_smoothing', 0.05),
        n_epochs=300,
        patience=35,
        warmup_epochs=25,
        verbose=True,
    )

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save(model.state_dict(), save_path)
    print(f"\nModel saved to {save_path}")
    print(f"Final val AUPRC: {val_auprc:.4f}")

    return model, val_auprc, history


# ── Run when called directly ───────────────────────────────────────
if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--full', action='store_true',
                        help='Run full Optuna search (50 trials) + final training')
    parser.add_argument('--trials', type=int, default=20,
                        help='Number of Optuna trials for --full mode')
    args = parser.parse_args()

    print("--- Training Pipeline v2 ---\n")

    (X_train, Y_train, X_val, Y_val,
     X_calib, Y_calib, X_test, Y_test,
     pos_weights, smiles_calib) = prepare_data()

    if args.full:
        best_params = run_optuna_search(
            X_train, Y_train, X_val, Y_val, pos_weights,
            n_trials=args.trials
        )
        model, val_auprc, history = run_full_training(
            best_params, X_train, Y_train, X_val, Y_val, pos_weights
        )
    else:
        # Quick sanity check: 10 epochs, no Optuna
        print("\n--- Quick Training Test (10 epochs, no Optuna) ---")
        model, val_auprc, history = train_model(
            X_train, Y_train, X_val, Y_val, pos_weights,
            n_epochs=10, patience=10, warmup_epochs=3, verbose=True,
        )
        print(f"\nQuick test val AUPRC: {val_auprc:.4f}")

        os.makedirs('models', exist_ok=True)
        torch.save(model.state_dict(), 'models/toxnet_final.pt')
        print("Model saved to models/toxnet_final.pt")

    print("\nDone.")