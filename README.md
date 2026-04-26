<p align="center">
  <h1 align="center">🧪 ToxIntel</h1>
  <p align="center">
    <b>A Prediction-to-Prescription Decision-Support System for Multi-Endpoint Molecular Toxicity</b>
  </p>
  <p align="center">
    <a href="#research-question">Research Question</a> •
    <a href="#novel-contributions">Novel Contributions</a> •
    <a href="#architecture">Architecture</a> •
    <a href="#getting-started">Getting Started</a> •
    <a href="#pipeline-overview">Pipeline</a> •
    <a href="#references">References</a>
  </p>
</p>

---

## Abstract

ToxIntel is not a standard toxicity classifier. It is a **Prediction-to-Prescription decision-support system** that transforms molecular toxicity prediction from a classification endpoint into a **structural optimization pipeline**. Given a molecule as a SMILES string, ToxIntel:

1. **Predicts** toxicity across all **12 Tox21 biological endpoints** simultaneously using **ToxNet** — a PyTorch multi-task neural network with a shared backbone and 12 independent task heads.
2. **Explains** which molecular fragment is responsible for the toxicity flag using **SHAP attribution**, cross-validated against established structural alerts (Brenk, PAINS, OCHEM).
3. **Prescribes** synthesizable bioisostere replacements sourced from **ChEMBL**, filtered by **SAScore** (synthesizability) and **ADME preservation** (ΔLogP, ΔMW).
4. **Re-predicts** all 12 endpoints for each candidate and ranks alternatives on a **Pareto front**, with **calibrated uncertainty intervals** and **out-of-distribution (OOD) flags**.

> **Closest prior art:** Cyto-Safe (Feitosa et al.) terminates at step 2 — visualizing the toxic fragment. ToxIntel extends through step 4, producing actionable, safety-validated structural alternatives.

---

## Research Question

> *"Does the geometric cohesion of each Tox21 endpoint's toxic class (measured by intraclass Tanimoto similarity) predict where SMOTE-augmented training and SHAP attribution systematically fail — and does a SHAP-guided, structurally-validated, synthesizability-filtered bioisostere pipeline demonstrate the practical consequence of those failures by producing Pareto-dominant reductions across all 12 endpoints for high-cohesion vs. low-cohesion scaffolds?"*

### Three Testable Claims

| Priority | Claim | Evaluation Method |
|----------|-------|-------------------|
| **Primary** | Geometric imbalance (intraclass Tanimoto cohesion) predicts SMOTE failure and SHAP attribution reliability per endpoint | Correlate cohesion ratio with AUPRC gain from SMOTE and structural alert overlap rate across 12 endpoints |
| Supporting | SHAP × alert cross-validation is the first evaluation criterion for attribution quality in molecular toxicity | Overlap rate: SHAP high-confidence fragments vs. known toxic scaffolds |
| Demonstration | Pareto optimization reveals risk-shifting effects that single-endpoint models miss | Show NR-AR↓ while SR-MMP↑ on real compounds |

---

## Novel Contributions

| Contribution | Standard Approach | ToxIntel |
|-------------|------------------|----------|
| Endpoint coverage | Single toxicity score | All 12 Tox21 endpoints simultaneously |
| Output format | Point estimate (e.g., "73% toxic") | Pareto front across 12 dimensions |
| Uncertainty | None | Mondrian Conformal Prediction (90% coverage) + OOD flag |
| Synthesizability | Not considered | SAScore < 4.0 gate on all bioisostere suggestions |
| Attribution validation | SHAP only | SHAP × structural alert overlay (Brenk + PAINS) |
| Scope declaration | Implicit | Explicitly declared (in-vitro only, no efficacy model) |
| Failure analysis | None | Stratified coverage report by pharmaceutical space proximity |
| Actionability | "It's toxic" | "Swap fragment X → Y, but watch SR-MMP" |
| Geometric imbalance | Not studied for Tox21 | Per-endpoint intraclass Tanimoto cohesion table ← **Novel finding** |

### The Publishable Claim

> *"To our knowledge, this is the first open-source pipeline to combine SHAP-guided bioisostere substitution, multi-endpoint Pareto optimization, synthesizability filtering, and out-of-distribution uncertainty quantification for the Tox21 benchmark — along with a systematic characterization of pipeline failure modes stratified by pharmaceutical chemical space coverage."*

---

## Architecture

```
Input: SMILES String
    │
    ▼
┌────────────────────────────────────────────────────────────┐
│  Phase 1: Environment + Data Acquisition                   │
│  • SMILES validation gate (RDKit)                          │
│  • Tox21 dataset: 7,831 compounds, 12 binary endpoints     │
└────────────────────────────────────────────────────────────┘
    │
    ▼
┌────────────────────────────────────────────────────────────┐
│  Phase 2: Multi-Representation Featurization                │
│  • Ablation: ECFP4 / ECFP6 / MACCS / RDKit / ECFP+Desc    │
│  • Murcko scaffold-stratified split (prevents leakage)     │
└────────────────────────────────────────────────────────────┘
    │
    ▼
┌────────────────────────────────────────────────────────────┐
│  Phase 3: Geometric Imbalance Analysis ← NOVEL             │
│  • Intraclass Tanimoto cohesion per endpoint               │
│  • Per-endpoint Focal Loss (γ=2, masked for NaN labels)    │
│  • SMOTE validity map (only in embedding space)            │
└────────────────────────────────────────────────────────────┘
    │
    ▼
┌────────────────────────────────────────────────────────────┐
│  Phase 4: ToxNet Training                                   │
│  • Shared backbone [1024→512→256] + 12 task heads          │
│  • OPTUNA hyperparameter search (50 trials, maximize AUPRC)│
│  • Masked Focal Loss for NaN labels                        │
└────────────────────────────────────────────────────────────┘
    │
    ▼
┌────────────────────────────────────────────────────────────┐
│  Phase 5: Evaluation Suite                                  │
│  • AUPRC (primary), AUROC, F1, MCC (secondary)            │
│  • Per-endpoint threshold tuning (recall floor ≥ 0.85)     │
│  • Geometric failure correlation analysis                  │
└────────────────────────────────────────────────────────────┘
    │
    ▼
┌────────────────────────────────────────────────────────────┐
│  Phase 6: Prescription Pipeline ← CORE CONTRIBUTION        │
│  Step 1:  Predict 12 endpoints → SHAP per endpoint         │
│  Step 1b: Cross-validate SHAP vs Brenk/PAINS/OCHEM alerts  │
│  Step 2:  Map top SHAP bit → molecular fragment             │
│  Step 3:  Query ChEMBL bioisosteres → SAScore + ADME filter │
│  Step 4:  Re-predict 12 endpoints → Pareto dominance ranking│
└────────────────────────────────────────────────────────────┘
    │
    ▼
┌────────────────────────────────────────────────────────────┐
│  Phase 7: Streamlit Dashboard (ToxIntel™)                   │
│  • Molecular identity + descriptor panel                   │
│  • 12-axis radar chart + safety threshold overlay          │
│  • Per-endpoint breakdown table (status + uncertainty)     │
│  • Bioisostere prescription engine with Pareto ranking     │
│  • Side-by-side comparison view (original vs candidate)    │
│  • Mondrian OOD warnings + reliability signals             │
└────────────────────────────────────────────────────────────┘
```

---

## Dataset

**Tox21 Challenge Dataset** — a benchmark for toxicity prediction in computational chemistry.

| Property | Value |
|----------|-------|
| Compounds | 7,831 |
| Endpoints | 12 binary toxicity assays |
| NaN Prevalence | 15–25% per endpoint |
| Toxic Prevalence | 2.8% (NR-AR-LBD) to 16.9% (SR-MMP) |
| Split | Murcko scaffold-stratified (mandatory) |

### The 12 Tox21 Endpoints

| Endpoint | Full Name | Category |
|----------|-----------|----------|
| NR-AR | Androgen Receptor | Nuclear Receptor |
| NR-AR-LBD | Androgen Receptor Ligand Binding Domain | Nuclear Receptor |
| NR-AhR | Aryl Hydrocarbon Receptor | Nuclear Receptor |
| NR-Aromatase | Aromatase | Nuclear Receptor |
| NR-ER | Estrogen Receptor alpha | Nuclear Receptor |
| NR-ER-LBD | Estrogen Receptor Ligand Binding Domain | Nuclear Receptor |
| NR-PPAR-gamma | Peroxisome Proliferator-Activated Receptor gamma | Nuclear Receptor |
| SR-ARE | Antioxidant Response Element | Stress Response |
| SR-ATAD5 | ATPase Family AAA Domain Containing 5 | Stress Response |
| SR-HSE | Heat Shock Element | Stress Response |
| SR-MMP | Mitochondrial Membrane Potential | Stress Response |
| SR-p53 | p53 Tumor Suppressor | Stress Response |

---

## ML Pipeline — Technical Details

### Featurization (5-Representation Ablation)

We empirically compare five molecular representations — a systematic comparison not previously published for Tox21 with AUPRC as the primary metric.

| Representation | Bits | Description |
|---------------|------|-------------|
| ECFP4_1024 | 1,024 | Morgan circular fingerprint (radius=2) |
| **ECFP4_2048** | **2,048** | **Selected representation** — compatible with SHAP attribution |
| ECFP6_2048 | 2,048 | Morgan circular fingerprint (radius=3) |
| MACCS | 166 | Structural key fingerprint |
| ECFP4+Desc | 2,048 + n | ECFP4 concatenated with RDKit descriptors (requires StandardScaler) |

> **Critical:** `bitInfo={}` must be declared outside the function call to preserve atom-to-bit mapping for SHAP interpretation in Phase 6.

### Scaffold Splitting (Mandatory)

Random splitting on molecular data causes **structural leakage** — similar Murcko scaffolds appear in both train and test sets, inflating AUC metrics falsely. We enforce **scaffold-stratified splitting** using `rdkit.Chem.Scaffolds.MurckoScaffold`, as mandated by Zhang et al. (2025) and Barua et al.

### Handling Severe Class Imbalance — 5-Layer Strategy

The Tox21 dataset exhibits severe, endpoint-variable imbalance (2.8%–16.9% toxic prevalence). We address this across five layers:

```
Layer 5: Architecture    → Per-endpoint heads + shared backbone (ToxNet)
Layer 4: Threshold       → Recall floor ≥ 0.85 per endpoint
Layer 3: Evaluation      → AUPRC primary (not AUROC, never accuracy)
Layer 2: Loss Function   → Focal Loss + per-endpoint α weights  ← Biggest gain
Layer 1: Data            → SMOTE in embedding space + Tomek Links (NOT raw fingerprints)
```

**Why Focal Loss over Weighted BCE:** Standard weighted BCE weights every toxic sample equally. Focal Loss (`FL(p) = -α(1-p)^γ · log(p)`) dynamically down-weights easy examples and up-weights hard, ambiguous examples near the decision boundary — where all real learning happens in imbalanced data.

**Why SMOTE in Embedding Space, not Fingerprint Space:** Morgan fingerprints are binary bit vectors. Interpolating two valid fingerprints (`[1,0,1,0,...] + [0,1,0,1,...] = [0.5, 0.5, ...]`) produces physically meaningless feature vectors that do not correspond to any real molecule. We first extract continuous embeddings from ToxNet's shared backbone, then apply ADASYN + TomekLinks in this continuous space.

### Geometric Imbalance — The Novel Finding

Beyond count-based imbalance, toxic compounds in Tox21 exhibit **geometric imbalance**: they cluster in dense, specific regions of chemical space while non-toxic compounds are diffusely spread.

We quantify this using **intraclass Tanimoto cohesion** — the mean pairwise Tanimoto similarity within the toxic class vs. within the non-toxic class. A cohesion ratio > 1.5 indicates geometric imbalance, where:
- SMOTE reinforces the existing cluster instead of generating novel examples
- SHAP may attribute toxicity to a cluster-specific feature rather than a mechanistically relevant substructure

> **This per-endpoint cohesion table is itself a publishable research finding.** No existing Tox21 paper has quantified geometric imbalance across all 12 endpoints.

### ToxNet Architecture

```
Input (batch, 2048)
    │
    ▼
Shared Backbone:
    Linear(2048→1024) → BatchNorm → GELU → Dropout(0.3)
    Linear(1024→512)  → BatchNorm → GELU → Dropout(0.3)
    Linear(512→256)   → BatchNorm → GELU → Dropout(0.3)
    │
    ├── Head 1: Linear(256→64) → ReLU → Dropout(0.15) → Linear(64→1)  → NR-AR
    ├── Head 2: Linear(256→64) → ReLU → Dropout(0.15) → Linear(64→1)  → NR-AR-LBD
    ├── ...
    └── Head 12: Linear(256→64) → ReLU → Dropout(0.15) → Linear(64→1) → SR-p53

Output: (batch, 12) logits
```

**Hyperparameter optimization** is performed with **OPTUNA** (50 trials), tuning learning rate, dropout, batch size, focal γ, shared layer dimensions, and weight decay — maximizing macro AUPRC on the validation set.

### Evaluation Protocol

| Metric | Role | Why |
|--------|------|-----|
| **AUPRC** | Primary | Focuses on minority class detection; random baseline = prevalence |
| AUROC | Secondary | Threshold-independent ranking quality |
| F1 | Secondary | Single-number summary at deployment threshold |
| MCC | Secondary | Accounts for all 4 confusion matrix cells; robust to imbalance |
| Accuracy | **Never used** | Meaningless under class imbalance |

**Per-endpoint threshold tuning:** We tune classification thresholds on the validation set with a **recall floor of 0.85** — a toxicity screening pipeline must prioritize catching toxic compounds (high recall) over reducing false positives (high precision).

### Benchmark Targets

| Model | Expected Macro AUPRC | Expected Macro AUROC |
|-------|:---:|:---:|
| Logistic Regression (baseline) | 0.20–0.28 | 0.72–0.76 |
| XGBoost + ECFP4 | 0.35–0.45 | 0.82–0.86 |
| **ToxNet + Focal Loss + OPTUNA** | **0.42–0.52** | **0.84–0.88** |

---

## The Prescription Pipeline (4-Step Corrected Flow)

This is the core contribution — transforming prediction into prescription.

```
Step 1:  Predict all 12 endpoints → SHAP per endpoint
Step 1b: Cross-validate SHAP against Brenk + PAINS + OCHEM alerts
         → Flag low-confidence attributions
         → Only proceed if alert overlap confirmed

Step 2:  Map top SHAP bit → molecular fragment (RDKit bitInfo atom mapping)

Step 3:  Query ChEMBL cache for bioisostere replacements
         → Filter A: SAScore < 4.0 (synthesizability)
         → Filter B: |ΔLogP| < 0.5, |ΔMW| < 25 Da (ADME preservation)
         → Flag: Tanimoto to DrugBank < 0.4 (out-of-pharma-space)

Step 4:  Re-predict all 12 endpoints for each candidate
         → Temperature-scaled calibrated probabilities
         → OOD detection (Tanimoto to training set)
         → Pareto dominance: DOMINATES / TRADE-OFF / DOMINATED / NO_CHANGE
         → Output: sorted Pareto front with uncertainty intervals
```

### Example Output

```
═══════════════════════════════════════════════════════════
 QUERY: CC(Cl)Nc1ccccc1  [Chloroacetanilide derivative]
 SHAP Attribution: Confidence = HIGH (confirmed: aniline alert)
 Coverage: Medium (Tanimoto to DrugBank = 0.51)
═══════════════════════════════════════════════════════════

  CANDIDATE 1: Replace -Cl with -F  [SAScore: 1.8 — EASY]
  Status: TRADE-OFF  (OOD: NO, confidence: HIGH)

  Endpoint         Original   Modified   Delta    Interval
  NR-AR            0.82       0.31       -0.51 ✅  [0.19, 0.43]
  SR-MMP           0.31       0.58       +0.27 ⚠️  [0.43, 0.73]
  ...

  CANDIDATE 2: Replace -Cl with -OH  [SAScore: 1.3 — EASY]
  Status: DOMINATES  (all 12 endpoints reduced or unchanged ✅)
═══════════════════════════════════════════════════════════
```

---

## Project Structure

```
PDS_exp10/
├── data/
│   └── tox21.csv                    # Raw dataset (8,014 compounds → 8,006 valid, 12 endpoints)
├── notebooks/                       # Each notebook has .ipynb + .py (jupytext source)
│   ├── 01_EDA.ipynb                 # ✅ Class balance, NaN distribution, imbalance ratios
│   ├── 02_Geometric_Imbalance.ipynb # ✅ Intraclass Tanimoto cohesion analysis ← Novel
│   ├── 03_Featurization_Ablation.ipynb # ✅ ECFP4 vs ECFP6 vs MACCS vs RDKit comparison
│   ├── 04_Training.ipynb            # ✅ ToxNet + Focal Loss + OPTUNA tuning
│   ├── 05_Evaluation.ipynb          # ✅ AUPRC / AUROC / MCC / threshold tuning
│   └── 06_Prescription.ipynb        # ✅ Full pipeline end-to-end demo
├── src/
│   ├── __init__.py                  # Package init
│   ├── featurize.py                 # SMILES validation + 5 fingerprint representations
│   ├── scaffold_split.py            # Murcko scaffold stratified split (no structural leakage)
│   ├── geometric_imbalance.py       # Intraclass Tanimoto cohesion analysis ← Novel
│   ├── focal_loss.py                # PerEndpointFocalLoss with NaN masking (PyTorch)
│   ├── model.py                     # ToxNet: shared backbone [2048→1024→512→256] + 12 heads
│   ├── train.py                     # Training loop + OPTUNA hyperparameter search
│   ├── evaluate.py                  # Full metric suite + threshold calibration
│   ├── shap_validator.py            # SHAP × structural alert cross-validation
│   ├── bioisostere.py               # ChEMBL query + SAScore filter
│   ├── pareto.py                    # Pareto dominance evaluation
│   ├── uncertainty.py               # Mondrian Conformal Prediction + OOD detection
│   └── prescription_pipeline.py     # Full corrected 4-step pipeline
├── models/
│   ├── toxnet_final.pt              # Trained ToxNet weights (2.96M parameters)
│   └── model_artifact.pkl           # Bundled artifact (weights + thresholds + Mondrian predictor)
├── Documents/                       # Research specs, literature, build session logs
├── app.py                           # ToxIntel™ Streamlit dashboard (Phase 7)
├── requirements.txt                 # pip dependencies
├── .gitignore
└── plan.md                          # Master implementation plan
```

---

## Current Progress

| Phase | Status | Key Files |
|-------|--------|----------|
| Phase 1: Environment + Data | ✅ Done | `featurize.py`, `data/tox21.csv` |
| Phase 2: Featurization + Split | ✅ Done | `featurize.py`, `scaffold_split.py` |
| Phase 3: Geometric Imbalance | ✅ Done | `geometric_imbalance.py`, `focal_loss.py` |
| Phase 4: ToxNet Training | ✅ Done | `model.py`, `train.py` |
| Phase 5: Evaluation | ✅ Done | `evaluate.py`, `05_Evaluation.ipynb` |
| Phase 6: Prescription Pipeline | ✅ Done | `prescription_pipeline.py`, `06_Prescription.ipynb` |
| Phase 7: Dashboard | ✅ Done | `app.py` |

---

## Getting Started

### Prerequisites

- Python 3.10+ (tested on 3.11.9)
- Git

### Installation

```bash
# Clone the repository
git clone https://github.com/priyali01/ToxIntel.git
cd ToxIntel

# Create and activate virtual environment
python -m venv venv
venv\Scripts\activate          # Windows
# source venv/bin/activate     # Mac/Linux

# Install dependencies
pip install -r requirements.txt

# Verify installation
python -c "from rdkit import Chem; print('RDKit OK')"
python -c "import torch; print(f'PyTorch {torch.__version__}')"
python -c "import streamlit; print('Streamlit OK')"
```

### Running the Dashboard

```bash
streamlit run app.py
```

The **ToxIntel™ Dashboard** includes:
- **Molecular Identity Panel** — Structure visualization + MW, LogP, HBD, HBA, TPSA, RotBonds
- **Reliability & OOD Signals** — Mondrian Conformal group badge (Known / Similar / Novel) + uncertainty count
- **12-Axis Toxicity Radar** — With per-endpoint safety threshold overlay (dashed line)
- **Per-Endpoint Breakdown Table** — Probability, threshold, TOXIC/CAUTION/SAFE status, uncertainty flag
- **Bioisostere Prescription Engine** — Pareto-ranked candidates with Dominant / Trade-off / Dominated counts
- **Side-by-Side Comparison View** — Original vs. candidate molecule images + overlaid radar + delta table
- **Net Safety Gain Metrics** — Aggregate improvement score + relative % + candidate Mondrian group
- **System Insights Footer** — Model architecture, loss function, calibration method

### Running the Notebooks

All notebooks are available as `.ipynb` files. Open them in Jupyter or VS Code:

```bash
# Option 1: Jupyter Notebook
jupyter notebook notebooks/01_EDA.ipynb

# Option 2: VS Code — just open the .ipynb file, click "Run All"

# Option 3: Execute from command line
jupyter nbconvert --to notebook --execute notebooks/01_EDA.ipynb --inplace
```

> **Note:** `.py` source files are also included for each notebook (jupytext format).
> These are plain-text readable versions of the same code.

---

## Notebook Guide — Step-by-Step Execution & Results

> **Run order:** Notebooks must be run sequentially (01 → 06) because each phase depends on outputs from the previous one. The dashboard (`app.py`) can be launched after Notebook 05 has generated `models/model_artifact.pkl`.

---

### 📓 01_EDA.ipynb — Exploratory Data Analysis

**Phase:** 1 · **Runtime:** ~30 seconds

**What it does:**
- Loads the Tox21 dataset (7,831 compounds, 12 binary endpoints)
- Validates SMILES using RDKit
- Visualizes per-endpoint class balance and missing label distributions

**How to run:**
```bash
jupyter notebook notebooks/01_EDA.ipynb   # then click "Run All"
```

**Key Results:**

| Metric | Value |
|--------|-------|
| Most imbalanced endpoint | NR-PPAR-gamma (2.9% toxic, 33.8:1 ratio) |
| Least imbalanced endpoint | SR-ARE (16.2% toxic, 5.2:1 ratio) |
| Highest NaN rate | NR-Aromatase, SR-ARE, SR-MMP (~26% missing) |
| Valid compounds after cleaning | 7,831 (8 invalid SMILES dropped) |

**Output files:**
- `notebooks/01_class_balance.png` — Toxic prevalence + imbalance ratio charts
- `notebooks/01_missing_labels.png` — NaN distribution per endpoint

**Conclusion:** Standard accuracy is meaningless under 2.9%–16.9% toxic prevalence. AUPRC must be the primary evaluation metric. Focal Loss with per-endpoint α weights is required.

---

### 📓 02_Geometric_Imbalance.ipynb — Novel Geometric Analysis

**Phase:** 3 · **Runtime:** ~1 minute

**What it does:**
- Computes **intraclass Tanimoto cohesion** — mean pairwise Tanimoto similarity within the toxic vs. non-toxic class for each endpoint
- Generates a **SMOTE validity map** to determine where SMOTE helps vs. hurts
- Produces the **publishable cohesion table** (novel finding — no prior Tox21 paper has done this)

**How to run:**
```bash
jupyter notebook notebooks/02_Geometric_Imbalance.ipynb
```

**Key Results:**

| Endpoint | Cohesion Ratio | SMOTE Valid? |
|----------|:--------------:|:------------:|
| NR-AR | 1.51 | ❌ No |
| NR-AR-LBD | 1.57 | ❌ No |
| SR-HSE | 0.95 | ✅ Yes |
| Other endpoints | ~1.0–1.3 | ✅ Yes |

**Output files:**
- `notebooks/02_geometric_imbalance.png` — Cohesion bar charts + ratio plot

**Conclusion:** NR-AR and NR-AR-LBD show geometric imbalance (ratio > 1.5), meaning SMOTE reinforces the existing cluster instead of generating diversity. For these endpoints, Focal Loss alone is the correct strategy. **This finding supports Claim 1 of the research question.**

---

### 📓 03_Featurization_Ablation.ipynb — Representation Comparison

**Phase:** 2 · **Runtime:** ~3–5 minutes

**What it does:**
- Compares 5 molecular representations (ECFP4_1024, ECFP4_2048, ECFP6_2048, MACCS, RDKit_Desc) using the **same scaffold split**
- Uses RandomForest as a consistent baseline classifier
- Produces per-endpoint AUPRC heatmaps

**How to run:**
```bash
jupyter notebook notebooks/03_Featurization_Ablation.ipynb
```

**Key Results:**

| Representation | Dims | Macro AUPRC | Macro AUROC |
|---------------|:----:|:-----------:|:-----------:|
| **ECFP4_2048** | **2048** | **Best** | **Best** |
| ECFP6_2048 | 2048 | Close second | — |
| ECFP4_1024 | 1024 | Slightly lower | — |
| MACCS | 167 | Lower | — |
| RDKit_Desc | 10 | Lowest | — |

**Output files:**
- `notebooks/03_featurization_ablation.png` — Macro AUPRC bar chart + dims-vs-performance scatter
- `notebooks/03_endpoint_heatmap.png` — Per-endpoint × representation AUPRC heatmap

**Conclusion:** ECFP4_2048 is selected as the primary representation because it achieves the best AUPRC while maintaining SHAP compatibility (bit → atom mapping via `bitInfo`). MACCS and RDKit descriptors lack the structural resolution needed for SHAP attribution in the Prescription Pipeline.

---

### 📓 04_Training.ipynb — ToxNet + Focal Loss + OPTUNA

**Phase:** 4 · **Runtime:** ~10–30 minutes (depends on `N_TRIALS`)

**What it does:**
1. Quick 20-epoch training with default hyperparameters (sanity check)
2. OPTUNA hyperparameter search (tunes lr, dropout, batch_size, focal_gamma, hidden_dims, weight_decay)
3. Final training with optimized parameters (50 epochs)
4. Saves `models/toxnet_final.pt`

**How to run:**
```bash
jupyter notebook notebooks/04_Training.ipynb
# Set N_TRIALS = 10 for quick test, 50 for full search
```

**Key Results:**

| Stage | Macro AUPRC |
|-------|:-----------:|
| Quick training (20 epochs) | ~0.20–0.25 |
| OPTUNA-optimized (50 epochs) | ~0.29–0.45 |
| Plan.md target | 0.42–0.52 |

**Output files:**
- `notebooks/04_training_history.png` — Loss + AUPRC curves (quick training)
- `notebooks/04_final_training.png` — Loss + AUPRC curves (final optimized model)
- `models/toxnet_final.pt` — Saved model weights

**Conclusion:** OPTUNA search significantly improves over default hyperparameters. Increasing `N_TRIALS` to 50 is recommended for production-grade results.

---

### 📓 05_Evaluation.ipynb — Metric Suite + Calibration + Artifact Bundle

**Phase:** 5 · **Runtime:** ~2–3 minutes

**What it does:**
1. Loads best ToxNet weights and computes AUPRC, AUROC, F1, MCC on the **test set**
2. Runs low-AUPRC diagnostics for struggling endpoints
3. Calibrates per-endpoint decision thresholds (recall floor ≥ 0.85)
4. Correlates geometric cohesion with AUPRC (tests **Claim 1** of the research question)
5. Fits **Mondrian Conformal Predictor** on the calibration split
6. Saves the production artifact bundle: `models/model_artifact.pkl`

**How to run:**
```bash
jupyter notebook notebooks/05_Evaluation.ipynb
```

**Key Results:**

| Metric | Value |
|--------|-------|
| Macro AUPRC (test set) | ~0.29 (leakage-free, honest baseline) |
| Recall floor | ≥ 0.85 on all calibrated endpoints |
| Pearson r (cohesion vs AUPRC) | Negative correlation (supports Claim 1) |
| Verification checks passed | 4/4 |

**Output files:**
- `notebooks/05_geometric_correlation.png` — Scatter: cohesion ratio vs. AUPRC
- `models/model_artifact.pkl` — Bundled artifact (weights + thresholds + Mondrian predictor)

**Conclusion:** The leakage-free scaffold split gives an honest AUPRC baseline. Geometric cohesion negatively correlates with endpoint performance, supporting the novel claim. The Mondrian Conformal Predictor provides 90% coverage guarantees for uncertainty quantification.

---

### 📓 06_Prescription.ipynb — End-to-End Pipeline Demo

**Phase:** 6 · **Runtime:** ~30 seconds

**What it does:**
- Loads the model artifact bundle
- Runs the full **4-Step Corrected Flow** on aniline (`c1ccccc1N`) — a known toxicophore
- Displays the Pareto front of bioisostere replacements with molecule grid images

**How to run:**
```bash
jupyter notebook notebooks/06_Prescription.ipynb
```

**Key Results:**
- SHAP validator identifies the **aniline substructure** as the primary toxic driver
- Bioisostere cache suggests replacements (e.g., pyridine, aliphatic amine)
- SAScore + ADME filters ensure realistic candidates
- Pareto evaluator classifies candidates as **DOMINATES**, **TRADE-OFF**, or **DOMINATED**

**Output:** Interactive Pareto front table + molecule grid visualization

**Conclusion:** The Prediction-to-Prescription pipeline successfully transforms a toxicity prediction into actionable structural modifications. Pareto-dominant candidates improve safety across all 12 endpoints without worsening any single endpoint.

---

### 🖥 app.py — ToxIntel™ Streamlit Dashboard

**Phase:** 7 · **Runtime:** Interactive (runs continuously)

**How to run:**
```bash
streamlit run app.py
```

**What it provides:**
1. Enter any SMILES → see the full 12-endpoint toxicity radar with safety thresholds
2. Per-endpoint breakdown table (TOXIC / CAUTION / SAFE + uncertainty flags)
3. Mondrian Conformal Prediction reliability badges (KNOWN / SIMILAR / NOVEL)
4. Bioisostere prescription table with Pareto ranking
5. Side-by-side comparison view with delta improvement metrics
6. Net Safety Gain score for the selected candidate

---

## Tech Stack

| Layer | Technology | Purpose |
|-------|-----------|---------|
| Environment | venv + pip | Lightweight virtual environment (no Conda needed) |
| Data Processing | Pandas, NumPy | CSV loading, NaN→-1 sentinel, array operations |
| Chemistry | RDKit | SMILES validation, fingerprints, scaffolds, SAScore, structural alerts |
| Featurization | RDKit (AllChem) | ECFP4/ECFP6/MACCS + bitInfo capture for SHAP atom mapping |
| Imbalance | imbalanced-learn | ADASYN + TomekLinks in embedding space |
| Loss Function | PyTorch (custom) | PerEndpointFocalLoss with NaN masking + per-endpoint α weights |
| Model | PyTorch | ToxNet — shared backbone [2048→1024→512→256] + 12 task heads |
| Hyperparameter Search | Optuna | Tuning lr, dropout, γ, batch_size, hidden_dims, weight_decay |
| Evaluation | scikit-learn | AUPRC (primary), AUROC, F1, MCC, precision-recall curves |
| Explainability | SHAP | DeepExplainer → bit-level attribution → atom-level mapping |
| Bioisostere DB | chembl-webresource-client | Offline cache of fragment replacements |
| Uncertainty | MAPIE | Mondrian Conformal Prediction (90% coverage guarantee) |
| Dashboard | Streamlit + Plotly | Interactive UI, radar charts, Pareto tables |
| Visualization | Matplotlib, Seaborn, Plotly | Heatmaps, PR curves, cohesion plots |
| Notebook Tooling | jupytext, nbconvert | .py ↔ .ipynb conversion and headless execution |

---

## Scope & Limitations

> ⚠️ **Honest scoping is a mark of scientific maturity.**

This pipeline optimizes for **Tox21 in-vitro toxicity only**. It has **no knowledge of**:
- Binding affinity or pharmacological activity
- In-vivo pharmacokinetics or ADME properties
- Target selectivity

**Designed for:**
- ✅ Early-stage hit filtering and scaffold prioritization
- ✅ Academic toxicity research and education
- ✅ Environmental/industrial chemical screening

**NOT designed for:**
- ❌ Late-stage lead optimization (pair with docking tools)
- ❌ Clinical candidate selection
- ❌ Replacement for wet-lab toxicity assays

---

## Key References

| Paper | Application in This Project |
|-------|-----------------------------|
| Zhang et al. (2025). *AI-Driven Drug Toxicity Prediction.* Toxics. | Representation ablation framework, SHAP open problem, regulatory framing |
| Feitosa et al. *Cyto-Safe.* | Primary prior art — differentiation argument (stops at visualization) |
| Barua et al. *Organ Toxicity ML.* | SMOTE + OPTUNA validation, benchmark AUC values |
| Rogers & Hahn (2010). *Extended-Connectivity Fingerprints.* JCIM. | Morgan fingerprint justification |
| Lundberg & Lee (2017). *SHAP.* NeurIPS. | Explainability method theoretical basis |
| Jiang et al. (2021). *GNNs vs Fingerprints.* | Justification for ECFP over GNN on small datasets |
| Lin et al. (2017). *Focal Loss.* ICCV. | Focal Loss for extreme class imbalance |
| Wu et al. (2018). *MoleculeNet.* Chemical Science. | Dataset benchmark and metric standards |
| Brenk et al. (2008). *Screening Libraries for Drug Discovery.* | Structural alert curation |

---

## License

This project is for academic and research purposes.

---

## Acknowledgements

Built using the [Tox21 Challenge](https://tripod.nih.gov/tox21/challenge/) dataset. Structural alert validation powered by [OCHEM](https://ochem.eu/), [ChEMBL](https://www.ebi.ac.uk/chembl/), and [DrugBank](https://go.drugbank.com/).
