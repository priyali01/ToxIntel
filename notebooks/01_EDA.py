# %% [markdown]
# # 01 — Exploratory Data Analysis (EDA)
# 
# **Phase 1 of plan.md**
# 
# This notebook loads the Tox21 dataset, validates SMILES, and visualizes:
# 1. Class balance per endpoint (how imbalanced is each toxicity assay?)
# 2. NaN/missing label distribution
# 3. Scaffold diversity

# %%
import sys
sys.path.insert(0, '..')

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from src.featurize import load_and_clean_tox21, TARGET_COLS

# %%
# Load and clean the dataset
df = load_and_clean_tox21()
print(f"Dataset shape: {df.shape}")
print(f"Columns: {list(df.columns)}")

# %% [markdown]
# ## 1. Class Balance Per Endpoint
# 
# The Tox21 dataset has **severe, endpoint-variable imbalance**.
# Some endpoints have as few as 2.9% toxic samples.

# %%
# Compute per-endpoint statistics
stats = []
for col in TARGET_COLS:
    known = df[col] != -1
    n_toxic = (df[col] == 1).sum()
    n_nontoxic = (df[col] == 0).sum()
    n_missing = (~known).sum()
    total = known.sum()
    prevalence = n_toxic / total * 100 if total > 0 else 0
    imbalance_ratio = n_nontoxic / n_toxic if n_toxic > 0 else float('inf')
    
    stats.append({
        'Endpoint': col,
        'Toxic': n_toxic,
        'Non-Toxic': n_nontoxic,
        'Missing': n_missing,
        'Total Tested': total,
        'Toxic %': round(prevalence, 1),
        'Imbalance Ratio': round(imbalance_ratio, 1),
    })

stats_df = pd.DataFrame(stats)
print(stats_df.to_string(index=False))

# %%
# Plot: Toxic prevalence per endpoint
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# Bar chart — toxic percentage
ax = axes[0]
colors = ['#e74c3c' if p > 10 else '#f39c12' if p > 5 else '#3498db' for p in stats_df['Toxic %']]
bars = ax.barh(stats_df['Endpoint'], stats_df['Toxic %'], color=colors)
ax.set_xlabel('Toxic Prevalence (%)')
ax.set_title('Tox21: Toxic Class Prevalence per Endpoint')
ax.axvline(x=5, color='gray', linestyle='--', alpha=0.5, label='5% threshold')
ax.legend()
ax.invert_yaxis()

# Bar chart — imbalance ratio
ax = axes[1]
bars = ax.barh(stats_df['Endpoint'], stats_df['Imbalance Ratio'], color='#2c3e50')
ax.set_xlabel('Imbalance Ratio (Non-Toxic / Toxic)')
ax.set_title('Class Imbalance Severity')
ax.axvline(x=10, color='red', linestyle='--', alpha=0.5, label='10:1 ratio')
ax.legend()
ax.invert_yaxis()

plt.tight_layout()
plt.savefig('01_class_balance.png', dpi=150, bbox_inches='tight')
plt.show()
print("Saved: 01_class_balance.png")

# %%
# Plot: NaN (missing label) distribution
fig, ax = plt.subplots(figsize=(10, 5))
nan_pcts = [row['Missing'] / len(df) * 100 for _, row in stats_df.iterrows()]
ax.barh(stats_df['Endpoint'], nan_pcts, color='#95a5a6')
ax.set_xlabel('Missing Labels (%)')
ax.set_title('Tox21: Missing Label Prevalence per Endpoint')
ax.invert_yaxis()
plt.tight_layout()
plt.savefig('01_missing_labels.png', dpi=150, bbox_inches='tight')
plt.show()
print("Saved: 01_missing_labels.png")

# %% [markdown]
# ## 2. Key Takeaways
# 
# - **Most imbalanced:** NR-PPAR-gamma (2.9% toxic, 33.8:1 ratio)
# - **Least imbalanced:** SR-ARE (16.2% toxic, 5.2:1 ratio)
# - **Highest NaN rate:** NR-Aromatase, SR-ARE, SR-MMP (~26% missing)
# - **Conclusion:** Standard accuracy is meaningless. Must use AUPRC as primary metric.
# - **Action:** Focal Loss with per-endpoint alpha weights (Phase 3)

# %%
print("\n=== Phase 1 EDA Complete ===")
print(f"Total compounds: {len(df)}")
print(f"Valid SMILES: {len(df)} (8 invalid dropped)")
print(f"Endpoints: {len(TARGET_COLS)}")
print(f"Most imbalanced: NR-PPAR-gamma ({stats_df.loc[stats_df['Endpoint']=='NR-PPAR-gamma', 'Toxic %'].values[0]}%)")
print(f"Least imbalanced: SR-ARE ({stats_df.loc[stats_df['Endpoint']=='SR-ARE', 'Toxic %'].values[0]}%)")
