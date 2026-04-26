# %% [markdown]
# # 02 — Geometric Imbalance Analysis (NOVEL)
# 
# **Phase 3 of plan.md — This is the primary research contribution.**
# 
# Standard imbalance analysis only counts samples (e.g., "3% toxic").
# We go further: measuring **geometric imbalance** — how tightly the toxic 
# molecules cluster in chemical space vs. non-toxic ones.
#
# **Key insight:** If toxic molecules all share the same scaffold (high cohesion),
# SMOTE reinforces the cluster instead of adding diversity, and SHAP may 
# attribute toxicity to a scaffold-specific feature rather than a mechanistic one.
#
# **This per-endpoint cohesion table has never been published for Tox21.**

# %%
import sys
import os
# Change working directory to project root
if os.path.basename(os.getcwd()) == 'notebooks':
    os.chdir('..')
sys.path.insert(0, '.')

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from src.featurize import load_and_clean_tox21, TARGET_COLS
from src.geometric_imbalance import analyze_all_endpoints

# %%
# Load data
df = load_and_clean_tox21()

# %% [markdown]
# ## 1. Per-Endpoint Geometric Cohesion Analysis
#
# For each endpoint, we compute:
# - **Toxic cohesion:** Mean pairwise Tanimoto similarity within the toxic class
# - **Non-toxic cohesion:** Mean pairwise Tanimoto within the non-toxic class
# - **Cohesion ratio:** toxic / non-toxic
# - **SMOTE valid?** If ratio > 1.5, SMOTE is likely counterproductive

# %%
# Run the analysis (this takes ~1 minute)
results = analyze_all_endpoints(df, TARGET_COLS)

# Convert to DataFrame for plotting
results_df = pd.DataFrame(results)
print("\n=== Geometric Imbalance Table (PUBLISHABLE) ===\n")
print(results_df.to_string(index=False))

# %%
# Plot: Cohesion comparison (toxic vs non-toxic per endpoint)
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# Grouped bar chart
ax = axes[0]
x = np.arange(len(TARGET_COLS))
width = 0.35
bars1 = ax.bar(x - width/2, results_df['toxic_cohesion'], width, label='Toxic', color='#e74c3c')
bars2 = ax.bar(x + width/2, results_df['nontoxic_cohesion'], width, label='Non-Toxic', color='#3498db')
ax.set_xlabel('Endpoint')
ax.set_ylabel('Mean Pairwise Tanimoto')
ax.set_title('Intraclass Tanimoto Cohesion per Endpoint')
ax.set_xticks(x)
ax.set_xticklabels(TARGET_COLS, rotation=45, ha='right')
ax.legend()

# Cohesion ratio with threshold line
ax = axes[1]
colors = ['#e74c3c' if r > 1.5 else '#2ecc71' for r in results_df['cohesion_ratio']]
ax.bar(TARGET_COLS, results_df['cohesion_ratio'], color=colors)
ax.axhline(y=1.5, color='red', linestyle='--', label='SMOTE invalidity threshold (1.5)')
ax.axhline(y=1.0, color='gray', linestyle=':', alpha=0.5)
ax.set_ylabel('Cohesion Ratio (Toxic / Non-Toxic)')
ax.set_title('Geometric Imbalance Ratio')
ax.set_xticklabels(TARGET_COLS, rotation=45, ha='right')
ax.legend()

plt.tight_layout()
plt.savefig('notebooks/02_geometric_imbalance.png', dpi=150, bbox_inches='tight')
plt.show()
print("Saved: notebooks/02_geometric_imbalance.png")

# %% [markdown]
# ## 2. SMOTE Validity Map
#
# Endpoints with cohesion ratio > 1.5 should NOT use SMOTE.
# For these endpoints, Focal Loss alone is the correct strategy.

# %%
# SMOTE validity summary
print("\n=== SMOTE Validity Map ===\n")
for _, row in results_df.iterrows():
    status = "YES - SMOTE may help" if row['smote_valid'] else "NO - Use Focal Loss only"
    marker = "  " if row['smote_valid'] else ">>"
    print(f"  {marker} {row['endpoint']:20s}  ratio={row['cohesion_ratio']:.4f}  -> {status}")

valid_count = results_df['smote_valid'].sum()
print(f"\nSMOTE valid for {valid_count}/{len(results_df)} endpoints")

# %% [markdown]
# ## 3. Key Findings
#
# - **NR-AR** (ratio 1.51) and **NR-AR-LBD** (ratio 1.57) show geometric imbalance
# - These are the endpoints where SMOTE is most likely to HURT performance
# - **SR-HSE** (ratio 0.95) is unusual: toxic class is MORE spread out than non-toxic
# - This table supports **Claim 1** of the research question
