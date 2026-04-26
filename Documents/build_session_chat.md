# ToxIntel — Build Session Chat Log
## Session 2: Execution Phase (Phase 1–4)
**Date:** 2026-04-26 (00:50 IST – 13:00 IST)

---

## What Happened in This Session

This session moved from planning to **code execution**. We built the core AI pipeline (Phases 1–4) following `plan.md` strictly, using a test-driven approach where each module was verified before proceeding.

---

## Key Decisions Made

### 1. venv Instead of Conda
- Conda was not installed on the system
- RDKit is now pip-installable (`pip install rdkit`)
- venv is simpler — no new tool to install
- `plan.md` updated to reflect this

### 2. Test-Driven Development (Build → Verify → Proceed)
- Added verification checkpoints to every phase in plan.md
- Each module is tested with assert statements before moving on
- No phase starts until the previous one passes all checks

### 3. Notebooks Alongside Code (Option B)
- Creating .py notebooks (VS Code cell format) alongside each phase
- Notebooks import from src/ — they're visualization wrappers, not logic containers
- Better for research paper: visual proof of results at each stage

---

## Modules Built & Verified

### Phase 1 — Environment + Data ✅
**Files created:**
- `requirements.txt` — all pip dependencies
- `src/__init__.py` — makes src/ importable
- `src/featurize.py` (Phase 1 part) — SMILES validation gate + data loader
- `data/tox21.csv` — downloaded from DeepChem GitHub (8,014 compounds)

**Results:**
- 8,014 raw compounds → 8,006 valid (8 aluminum compounds dropped)
- 12 endpoints with 7–26% NaN prevalence
- Toxic prevalence: 2.9% (NR-PPAR-gamma) to 16.2% (SR-ARE)

---

### Phase 2 — Featurization + Scaffold Split ✅
**Files created/modified:**
- `src/featurize.py` — added 5 fingerprint representations:
  - `smiles_to_morgan()` — ECFP4/ECFP6 fingerprints
  - `smiles_to_morgan_with_info()` — same + bitInfo for SHAP atom mapping
  - `smiles_to_maccs()` — 166-bit structural keys
  - `smiles_to_rdkit_descriptors()` — 10 physicochemical descriptors
  - `featurize_dataset()` — batch processor for any representation
- `src/scaffold_split.py` — Murcko scaffold-stratified splitting

**Results:**
- All 5 representations generate correct shapes
- bitInfo captures 36 active bits (needed for Phase 6 SHAP)
- Scaffold split: Train 6,404 / Val 800 / Test 802 — zero overlap
- 2,325 unique scaffolds in the dataset

---

### Phase 3 — Geometric Imbalance + Focal Loss ✅
**Files created:**
- `src/geometric_imbalance.py` — per-endpoint Tanimoto cohesion analysis (NOVEL)
- `src/focal_loss.py` — PerEndpointFocalLoss with NaN masking

**Geometric Imbalance Results (PUBLISHABLE TABLE):**
```
Endpoint              Toxic Coh  NonTox Coh    Ratio  SMOTE?
NR-AR                    0.1226      0.0813   1.5081     NO
NR-AR-LBD                0.1254      0.0798   1.5717     NO
NR-AhR                   0.1080      0.0802   1.3464    YES
NR-Aromatase             0.0905      0.0803   1.1259    YES
NR-ER                    0.0982      0.0807   1.2172    YES
NR-ER-LBD                0.1013      0.0794   1.2747    YES
NR-PPAR-gamma            0.0984      0.0801   1.2286    YES
SR-ARE                   0.0841      0.0838   1.0038    YES
SR-ATAD5                 0.0908      0.0823   1.1027    YES
SR-HSE                   0.0780      0.0821   0.9500    YES
SR-MMP                   0.0962      0.0816   1.1783    YES
SR-p53                   0.0877      0.0803   1.0927    YES
```
- NR-AR and NR-AR-LBD: SMOTE invalid (ratio > 1.5)
- SMOTE valid for 10/12 endpoints

**Focal Loss Results:**
- Per-endpoint weights computed (NR-PPAR-gamma highest at 33.8x)
- Focal Loss < BCE (3.10 vs 7.33) — confirms focal correctly down-weights easy examples
- NaN masking works correctly

---

### Phase 4 — ToxNet + Training ✅
**Files created:**
- `src/model.py` — ToxNet architecture
  - Shared backbone: [2048→1024→512→256] with BatchNorm + GELU + Dropout
  - 12 independent task heads: [256→64→1]
  - Total parameters: 2,956,044
- `src/train.py` — training loop + OPTUNA hyperparameter search
  - Focal Loss with per-endpoint alpha weights
  - Early stopping on validation AUPRC
  - ReduceLROnPlateau scheduler

**Results:**
- Quick test (5 epochs): val AUPRC = 0.295 (expected to reach 0.42-0.52 with full training + OPTUNA)
- Model save/reload verified
- Model output shape: (batch, 12) ✅

---

### Notebooks Created ✅
- `notebooks/01_EDA.py` — class balance charts, NaN distribution
- `notebooks/02_Geometric_Imbalance.py` — cohesion heatmap, SMOTE validity
- `notebooks/03_Featurization_Ablation.py` — 5-representation AUPRC comparison
- `notebooks/04_Training.py` — ToxNet training with OPTUNA

Notebooks use `# %%` cell format (VS Code compatible).

---

## Current Project State

```
PDS_exp10/
├── data/tox21.csv                      ✅ 8,006 valid compounds
├── notebooks/
│   ├── 01_EDA.py                       ✅ 
│   ├── 02_Geometric_Imbalance.py       ✅
│   ├── 03_Featurization_Ablation.py    ✅
│   └── 04_Training.py                  ✅
├── src/
│   ├── __init__.py                     ✅
│   ├── featurize.py                    ✅ Phase 1+2
│   ├── scaffold_split.py              ✅ Phase 2
│   ├── geometric_imbalance.py         ✅ Phase 3
│   ├── focal_loss.py                  ✅ Phase 3
│   ├── model.py                       ✅ Phase 4
│   └── train.py                       ✅ Phase 4
├── models/toxnet_final.pt             ✅ (quick-trained)
├── venv/                              ✅ (gitignored)
├── .gitignore                         ✅
├── requirements.txt                   ✅
├── README.md                          ✅ (updated)
├── plan.md                            ✅ (updated for venv + TDD)
└── implementation_plan.md             ✅ (full-stack upgrade path)
```

---

## What's Next

**Phase 5: Evaluation Suite** — Create `src/evaluate.py` with:
- AUPRC, AUROC, F1, MCC per endpoint
- Per-endpoint threshold tuning (recall floor ≥ 0.85)
- Geometric failure correlation analysis

**Phase 6: Prescription Pipeline** — The core novel contribution
**Phase 7: Streamlit Dashboard** — Visual UI

---

## Git History (This Session)

```
dfed3d8  Done with Data pipeline + featurization + scaffold split
9488113  implement geometric imbalance analysis and focal loss
[next]   Phase 4 + notebooks + README updates
```

## Remote Repositories
- **origin:** https://github.com/priyali01/ToxIntel.git
- **friend-origin:** https://github.com/Abhi-786-coder/PDS_exp10.git
