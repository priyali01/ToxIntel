"""
retrain_best.py — Retrain final model with best Optuna params + ecfp4_desc features

Place this file in your project root (same level as src/).

Changes from previous run:
    - method='ecfp4_desc': ECFP4 (2048 bits) + 10 RDKit descriptors = 2058 features
    - lr=0.0004: lower LR needed for descriptor feature space (was 0.00183)
    - n_epochs=300, patience=35, warmup_epochs=25: longer training for large arch

Run with:
    python retrain_best.py
"""

import sys
sys.path.insert(0, '.')

from src.train import prepare_data, run_full_training

# Best params from Optuna (30 trials on ecfp4_2048)
# LR reduced from 0.00183 → 0.0004 for ecfp4_desc stability
best_params = {
    'lr':              0.0004,
    'dropout':         0.3957417277563036,
    'batch_size':      128,
    'gamma':           1.9644838177524269,
    'weight_decay':    4.128838345584083e-05,
    'head_hidden':     128,
    'label_smoothing': 0.0,
    'hidden_dims':     'large',
}

print("=" * 55)
print("  Retraining ToxNet — ecfp4_desc features")
print("=" * 55)

# Prepare data with combined ECFP4 + descriptor features
# Scaler is fit on train only — no leakage
(X_train, Y_train, X_val, Y_val,
 X_calib, Y_calib, X_test, Y_test,
 pos_weights, smiles_calib) = prepare_data(method='ecfp4_desc')

print(f"\nInput dim: {X_train.shape[1]}  (2048 ECFP4 bits + 10 descriptors)")
print(f"Best params: {best_params}\n")

# Train — n_epochs/patience/warmup are set inside run_full_training (train.py)
# Make sure train.py has: n_epochs=300, patience=35, warmup_epochs=25
model, val_auprc, history = run_full_training(
    best_params,
    X_train, Y_train,
    X_val, Y_val,
    pos_weights,
    save_path='models/toxnet_final.pt',
)

print(f"\n{'=' * 55}")
print(f"  Final val AUPRC: {val_auprc:.4f}")
print(f"{'=' * 55}")

if val_auprc >= 0.40:
    print("  [PASS] Meets plan.md target (>=0.40)")
elif val_auprc >= 0.37:
    print("  [CLOSE] Run targeted Optuna on ecfp4_desc next")
else:
    print("  [RETRY] LR may still need tuning — try lr=0.0002")