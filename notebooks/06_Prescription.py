# %% [markdown]
# # 06 — The Prediction-to-Prescription Pipeline
# 
# **Phase 6 of plan.md**
# 
# This is the culmination of the ToxIntel project. We demonstrate the **4-Step Corrected Flow**:
# 1. Predict toxicity and extract SHAP attributions.
# 2. Validate SHAP bits against known structural alerts (OCHEM/PAINS/Brenk).
# 3. Query the (mock) ChEMBL offline cache for bioisostere replacements and filter by Synthesizability (SAScore) & ADME preservation.
# 4. Re-predict toxicity, detect Out-Of-Distribution (OOD) shifts, and evaluate Pareto dominance.

# %%
import sys
import os
if os.path.basename(os.getcwd()) == 'notebooks':
    os.chdir('..')
sys.path.insert(0, '.')

import torch
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Draw
from IPython.display import display

import pickle
from src.model import ToxNet, load_toxnet
from src.featurize import load_and_clean_tox21
from src.prescription_pipeline import run_prescription_pipeline

# %% [markdown]
# ## 1. Load Model and Training Data
# We load the training data to compute the OOD Tanimoto distances later.

# %%
# Load the dataset (we only need the SMILES for the OOD check)
df = load_and_clean_tox21()
# Take a random sample of 500 training smiles for fast OOD checking
train_smiles_sample = df['smiles'].sample(500, random_state=42).tolist()

# Load the model artifact bundle
try:
    with open('models/model_artifact.pkl', 'rb') as f:
        model_artifact = pickle.load(f)
    print("[SUCCESS] Successfully loaded models/model_artifact.pkl")
except Exception as e:
    print(f"[ERROR] Failed to load model artifact: {e}")
    # Fallback for dev if needed
    model = load_toxnet('models/toxnet_final.pt', input_dim=2048)
    model.eval()
    model_artifact = {'mondrian_predictor': type('dummy', (), {'base_model': model})()}

# %% [markdown]
# ## 2. Execute the Pipeline on a Toxic Target
# Let's run a test molecule that contains a known toxicophore (aniline: `c1ccccc1N`). Aniline derivatives frequently flag positive for toxicity endpoints.
# 
# Target: **Aniline (c1ccccc1N)**

# %%
target_smiles = 'c1ccccc1N'

# Show the molecule
mol = Chem.MolFromSmiles(target_smiles)
display(Draw.MolToImage(mol, size=(200, 200)))

# Run the 4-step pipeline
results = run_prescription_pipeline(target_smiles, model_artifact)

# %% [markdown]
# ## 3. Review the Output Pareto Front

# %%
if results.get('status') == 'SUCCESS' and results.get('suggestions'):
    pareto_df = pd.DataFrame(results['suggestions'])
    print("\n=== Bioisostere Pareto Front ===")
    display(pareto_df)
    
    # Draw the viable replacements
    mols = [Chem.MolFromSmiles(s) for s in pareto_df['smiles']]
    labels = [f"{st}\nSA:{sa}\nΔLogP:{dlp}" for st, sa, dlp in zip(pareto_df['pareto_status'], pareto_df['sa_score'], pareto_df['delta_logp'])]
    img = Draw.MolsToGridImage(mols, molsPerRow=3, subImgSize=(250, 250), legends=labels)
    display(img)
else:
    print("\nNo viable replacements generated or pipeline aborted.")
    if 'reason' in results:
        print(f"Reason: {results['reason']}")

# %% [markdown]
# ## 4. Summary of the Corrected Flow
# 
# Notice what just happened:
# - The model predicted toxicity.
# - The SHAP validator matched the SHAP attention to the **aniline** substructure.
# - The cache suggested replacing it with **pyridine** or an **aliphatic amine**.
# - The ADME/Synthesizability filters ensured the molecules were realistic.
# - The Pareto evaluator checked if the new molecules were strictly safer or just traded one toxicity for another.
# 
# **This is the Prediction-to-Prescription Pipeline!**
