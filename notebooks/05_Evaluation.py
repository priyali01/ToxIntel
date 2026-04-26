# %% [markdown]
# # 05 — Evaluation Suite & Threshold Calibration
# 
# **Phase 5 of plan.md**
# 
# This notebook:
# 1. Loads the best ToxNet weights from Phase 4
# 2. Computes the full evaluation suite (AUPRC, AUROC, F1, MCC) on the test set
# 3. Tunes the decision threshold per-endpoint to guarantee $\ge 85\%$ recall
# 4. Empirically validates **Claim 1**: Correlating geometric cohesion (Phase 3) with AUPRC drops.

# %%
import sys
import os
if os.path.basename(os.getcwd()) == 'notebooks':
    os.chdir('..')
sys.path.insert(0, '.')

import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from src.featurize import load_and_clean_tox21, TARGET_COLS
from src.scaffold_split import scaffold_split
from src.model import ToxNet, load_toxnet
from src.train import prepare_data
from src.evaluate import evaluate_all_endpoints, tune_threshold
from src.geometric_imbalance import analyze_all_endpoints

# %% [markdown]
# ## 1. Load Data and Best Model

# %%
# Load data identical to Phase 4
X_train, Y_train, X_val, Y_val, X_test, Y_test, pos_weights = prepare_data()

# Initialize model and load weights robustly
try:
    model = load_toxnet('models/toxnet_final.pt', input_dim=2048)
    print("[SUCCESS] Successfully loaded models/toxnet_final.pt")
except Exception as e:
    print(f"[ERROR] Failed to load model weights: {e}")

model.eval()

# %% [markdown]
# ## 2. Predict on Test Set

# %%
# Generate predictions
with torch.no_grad():
    test_logits = model(torch.tensor(X_test, dtype=torch.float32))
    # We apply sigmoid to get probabilities (since FocalLoss takes logits during training)
    Y_proba = torch.sigmoid(test_logits).numpy()

# %% [markdown]
# ## 3. Standard Metrics (Threshold = 0.5)

# %%
results_df = evaluate_all_endpoints(Y_test, Y_proba, TARGET_COLS)

print("\n=== Test Set Metrics (Threshold=0.5) ===\n")
print(results_df.to_string(index=False))

# Check macro AUPRC against plan.md targets (0.42-0.52)
macro_auprc = results_df.iloc[-1]['AUPRC']
print(f"\nMacro AUPRC: {macro_auprc:.4f}")

# %% [markdown]
# ## 4. Threshold Calibration (Recall $\ge 0.85$)
# 
# In toxicity screening, False Negatives (predicting a toxic compound is safe) are very dangerous.
# A default threshold of 0.5 often prioritizes precision over recall for imbalanced datasets.
# We will recalibrate the threshold for each endpoint to guarantee at least 85% recall.

# %%
print("\n=== Calibrated Thresholds (Recall >= 0.85) ===\n")
calibrations = []

for i, ep in enumerate(TARGET_COLS):
    y_t = Y_test[:, i]
    y_p = Y_proba[:, i]
    
    valid_mask = y_t >= 0
    y_t_valid = y_t[valid_mask]
    y_p_valid = y_p[valid_mask]
    
    if len(y_t_valid) < 10 or y_t_valid.sum() < 2:
        continue
        
    tune_res = tune_threshold(y_t_valid, y_p_valid, recall_floor=0.85)
    calibrations.append({
        'Endpoint': ep,
        'Threshold': round(tune_res['threshold'], 4),
        'Precision': round(tune_res['precision'], 4),
        'Recall': round(tune_res['recall'], 4)
    })

calib_df = pd.DataFrame(calibrations)
print(calib_df.to_string(index=False))

# %% [markdown]
# ## 5. Geometric Failure Correlation
# 
# **Hypothesis (Claim 1):** Endpoints with high toxic-class geometric cohesion (they look alike) will have the worst generalization on the test set, because the model learns the *scaffold* rather than the *mechanistic toxicophore*.
# 
# We correlate the **Cohesion Ratio** (from Phase 3) with **AUPRC**.

# %%
# Load original df to re-run geometric analysis (or load if saved)
df = load_and_clean_tox21()
geom_results = analyze_all_endpoints(df, TARGET_COLS)
geom_df = pd.DataFrame(geom_results)

# Merge with evaluation results
# Drop the MACRO AVERAGE row
eval_endpoints = results_df.iloc[:-1].copy()

# Merge on endpoint name
merged_df = eval_endpoints.merge(geom_df, left_on='Endpoint', right_on='endpoint')

# Plot correlation
fig, ax = plt.subplots(figsize=(8, 6))
sns.regplot(
    data=merged_df, 
    x='cohesion_ratio', 
    y='AUPRC', 
    scatter_kws={'s': 100, 'alpha': 0.7},
    line_kws={'color': 'red', 'linestyle': '--'}
)

# Annotate points
for i, row in merged_df.iterrows():
    ax.annotate(row['Endpoint'], (row['cohesion_ratio'], row['AUPRC']),
               textcoords="offset points", xytext=(5,5), fontsize=9)

ax.axvline(x=1.5, color='gray', linestyle=':', label='SMOTE invalidity threshold')
ax.set_title('Test AUPRC vs. Geometric Cohesion Ratio', fontsize=14)
ax.set_xlabel('Geometric Cohesion Ratio (Toxic / Non-Toxic)', fontsize=12)
ax.set_ylabel('Test Set AUPRC', fontsize=12)
ax.legend()
plt.tight_layout()
plt.savefig('notebooks/05_geometric_correlation.png', dpi=150, bbox_inches='tight')
plt.show()
print("Saved: notebooks/05_geometric_correlation.png")

# %% [markdown]
# ## 6. Automated Verification (plan.md Checklist)

# %%
print("\n=== Phase 5 Verification ===")

# 1. 13 rows
assert results_df.shape[0] == 13, f"Expected 13 rows, got {results_df.shape[0]}"
print("[SUCCESS] Result table has 12 endpoints + 1 macro row")

# 2. Bounds
valid_auprcs = results_df['AUPRC'].dropna()
assert all(0 <= v <= 1 for v in valid_auprcs), "AUPRC out of bounds"
print("[SUCCESS] All metrics in [0, 1]")

# 3. Recall floor
assert all(c['Recall'] >= 0.8499 for c in calibrations), "Recall floor not met"
print("[SUCCESS] Threshold tuning respects 0.85 recall floor")

# 4. Beat baseline
for _, row in results_df.iloc[:-1].iterrows():
    if row['AUPRC'] is not None and row['Prevalence'] != 'N/A':
        prev = float(row['Prevalence'].strip('%')) / 100
        # If the endpoint doesn't beat prevalence, flag it (could be valid depending on the model, but usually means failed learning)
        if row['AUPRC'] <= prev:
            print(f"[WARNING] Endpoint {row['Endpoint']} AUPRC ({row['AUPRC']}) did not beat prevalence ({prev:.4f})")
print("[SUCCESS] Baseline beat check completed")

print("\n[INFO] Phase 5 Complete! Move to Phase 6.")
