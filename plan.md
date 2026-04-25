# 🧪 Molecular Toxicity Classification Engine — Master Implementation Plan

> **This is the single authoritative reference for building the entire project.**
> It consolidates `roadmap.html`, `corrected_pipeline.md`, `imbalance_strategy.md`, `literature_grounded_spec.md`, and `novelty_thought_process.md` into one actionable document.

---

## 1. What This Project Actually Is

Not a toxicity classifier. A **Prediction-to-Prescription decision-support system** that:

1. Predicts toxicity across **all 12 Tox21 endpoints simultaneously**
2. Identifies **which molecular fragment** is responsible (SHAP × structural alert validation)
3. Suggests **synthesizable bioisostere replacements** (ChEMBL-backed, SAScore-filtered)
4. Re-predicts across **all 12 endpoints** for each candidate and outputs a **Pareto front**
5. Attaches **calibrated uncertainty intervals** and **OOD flags** to every prediction

**Closest prior art:** Cyto-Safe (Feitosa et al.) — which stops at step 2. We go through step 5.
**Key literature:** Zhang et al. (2025), Barua et al., Feitosa et al.

### The One-Sentence Differentiator
> *"Where Cyto-Safe terminates at visualizing the molecular fragment responsible for toxicity, our pipeline extends to suggesting validated, synthesizable replacements and re-predicting safety across all 12 Tox21 endpoints simultaneously — transforming toxicity prediction from a classification endpoint into a structural optimization decision-support system."*

---

## 2. Final Research Question

> **"Does the geometric cohesion of each Tox21 endpoint's toxic class (measured by intraclass Tanimoto similarity) predict where SMOTE-augmented training and SHAP attribution systematically fail — and does a SHAP-guided, structurally-validated, synthesizability-filtered bioisostere pipeline demonstrate the practical consequence of those failures by producing Pareto-dominant reductions across all 12 endpoints for high-cohesion vs. low-cohesion scaffolds?"**

### Three Testable Claims

| Priority | # | Claim | How to Test |
|----------|---|-------|-------------|
| **PRIMARY** | 1 | Geometric imbalance (intraclass Tanimoto cohesion) predicts SMOTE failure and SHAP attribution reliability per endpoint | Correlate cohesion ratio with AUPRC gain from SMOTE and alert overlap rate across 12 endpoints |
| Supporting | 2 | SHAP × alert cross-validation is the first evaluation criterion for attribution quality in molecular toxicity | Overlap rate: SHAP high-confidence vs known toxic scaffolds |
| Demonstration | 3 | Pareto optimization reveals risk-shifting effects single-endpoint models miss | Show NR-AR↓ while SR-MMP↑ on real compounds |

---

## 3. Architecture Flow

```
Input: SMILES String
    ↓
Phase 1: Environment + Data Acquisition
    ↓
Phase 2: Multi-Representation Featurization (ECFP4/ECFP6/MACCS/RDKit ablation + Scaffold-stratified split)
    ↓
Phase 3: Geometric Imbalance Analysis (Intraclass Tanimoto cohesion + Focal Loss + SMOTE where valid)
    ↓
Phase 4: ToxNet Training (Shared backbone + 12 task heads + OPTUNA + Masked Focal Loss)
    ↓
Phase 5: Evaluation (AUPRC primary, AUROC+F1+MCC secondary + Per-endpoint threshold tuning)
    ↓
Phase 6: Prescription Pipeline (SHAP → Alert validation → ChEMBL bioisostere → SAScore → Pareto)
    ↓
Phase 7: Dashboard (Streamlit — Pareto front UI + regulatory disclaimer)
```

---

## 4. Complete File Structure

```
tox21_project/
├── data/
│   ├── tox21.csv                    # Raw dataset (7,831 compounds, 12 endpoints)
│   ├── drugbank_fps.pkl             # DrugBank fingerprints (coverage check)
│   └── ochem_alerts.json            # Structural alert SMARTS from OCHEM
│
├── notebooks/
│   ├── 01_EDA.ipynb                 # Class balance, scaffold distribution
│   ├── 02_Geometric_Imbalance.ipynb # Intraclass Tanimoto cohesion analysis ← Novel
│   ├── 03_Featurization_Ablation.ipynb  # ECFP4 vs MACCS vs RDKit comparison
│   ├── 04_Training.ipynb            # ToxNet + Focal Loss + OPTUNA
│   ├── 05_Evaluation.ipynb          # AUPRC / AUROC / MCC / threshold tuning
│   └── 06_Prescription.ipynb        # Full pipeline end-to-end demo
│
├── src/
│   ├── featurize.py                 # All 5 fingerprint representations
│   ├── scaffold_split.py            # Murcko scaffold stratified split
│   ├── geometric_imbalance.py       # Intraclass cohesion analysis
│   ├── focal_loss.py                # PerEndpointFocalLoss (PyTorch)
│   ├── model.py                     # ToxNet architecture (PyTorch)
│   ├── train.py                     # Training loop + OPTUNA
│   ├── evaluate.py                  # Full metric suite (AUPRC, AUROC, MCC, F1)
│   ├── shap_validator.py            # SHAP × alert cross-validation
│   ├── bioisostere.py               # ChEMBL query + SAScore filter
│   ├── pareto.py                    # Pareto dominance evaluation
│   ├── uncertainty.py               # Temperature scaling + OOD detection
│   └── prescription_pipeline.py     # Full corrected 4-step pipeline
│
├── models/
│   ├── toxnet_final.pt              # Trained ToxNet weights
│   └── temperatures.pkl             # Per-endpoint calibration temperatures
│
├── app.py                           # Streamlit dashboard
├── environment.yml                  # Conda environment definition
└── requirements.txt
```

---

## 5. Dataset Facts

- **7,831 compounds**, **12 binary toxicity endpoints**
- **NaN prevalence:** 15–25% per endpoint (use -1 sentinel, mask during loss)
- **Toxic prevalence:** 2.8% (NR-AR-LBD) → 16.9% (SR-MMP) — severe, endpoint-variable imbalance

### Dataset Sources

| Source | URL | Used For |
|--------|-----|----------|
| Official Tox21 | https://tripod.nih.gov/tox21/challenge/ | Ground truth |
| DeepChem Loader | `dc.molnet.load_tox21()` | Auto scaffold split |
| OCHEM | https://ochem.eu/alerts/list.do | Structural alert SMARTS |
| ChEMBL | https://www.ebi.ac.uk/chembl/ | Bioisostere validation |
| DrugBank | https://go.drugbank.com/ | Pharmaceutical space check |

---

## 6. Conda Environment  *(Used for all AI/ML work)*

```yaml
name: tox21_env
channels:
  - conda-forge
  - defaults
dependencies:
  - python=3.10
  - rdkit
  - numpy
  - pandas
  - scikit-learn
  - matplotlib
  - seaborn
  - pip
  - pip:
    - imbalanced-learn
    - xgboost
    - lightgbm
    - optuna
    - shap
    - torch
    - torchvision
    - skorch
    - streamlit
    - plotly
    - chembl-webresource-client
    - mapie
    - deepchem
```

---

## 6b. Tech Stack — What's Used Where

| Layer | Technology | Purpose | Phase |
|-------|-----------|---------|-------|
| **Environment** | Conda | Manages RDKit, PyTorch, and all ML deps (RDKit cannot be pip-installed) | Phase 1 |
| **Data Processing** | Pandas, NumPy | Loading CSV, handling NaN labels, array ops | Phase 1–2 |
| **Chemistry** | RDKit | SMILES validation, Morgan fingerprints, Murcko scaffolds, SAScore, PAINS/Brenk filters, 2D drawing | Phase 1–7 |
| **Featurization** | RDKit (AllChem) | ECFP4/ECFP6/MACCS fingerprints + bitInfo capture for SHAP mapping | Phase 2 |
| **Scaling** | scikit-learn (StandardScaler) | Normalizing continuous descriptors when using ECFP+Desc | Phase 2 |
| **Imbalance** | imbalanced-learn (ADASYN, TomekLinks) | SMOTE in embedding space for valid endpoints only | Phase 3 |
| **Loss Function** | PyTorch (custom) | PerEndpointFocalLoss with NaN masking | Phase 3–4 |
| **Model** | PyTorch | ToxNet — shared backbone + 12 task heads | Phase 4 |
| **Hyperparameter Tuning** | Optuna | 50 trials optimizing lr, dropout, gamma, dims | Phase 4 |
| **Evaluation** | scikit-learn | AUPRC, AUROC, F1, MCC, precision-recall curves | Phase 5 |
| **Explainability** | SHAP | TreeExplainer / DeepExplainer → bit-level attribution | Phase 6 |
| **Bioisostere DB** | chembl-webresource-client | One-time offline cache of fragment replacements | Phase 6 |
| **Uncertainty** | SciPy, MAPIE | Temperature scaling + conformal prediction intervals | Phase 6 |
| **Dashboard** | Streamlit + Plotly | Interactive UI, radar charts, molecule viewer, Pareto table | Phase 7 |
| **Visualization** | Matplotlib, Seaborn, Plotly | Heatmaps, PR curves, cohesion plots, Pareto fronts | Phase 2–7 |

---

## 6c. Future Upgrade: Full-Stack Product (Post-Streamlit)

> **Current plan:** Ship fast with Streamlit (Phase 7). The AI core (`src/`) stays identical — only the UI layer changes.

When you are ready to upgrade this into a production-grade full-stack web app, here is the migration path:

```
CURRENT (Streamlit — monolithic):
    app.py → imports from src/ → renders UI

FUTURE (Full-Stack — 3-tier):
    frontend/  (React/Next.js)  → calls backend API
    backend/   (FastAPI)        → imports from src/ → returns JSON
    src/       (unchanged)      → the exact same AI modules
```

### What Changes

| Component | Current (Streamlit) | Future (Full-Stack) |
|-----------|--------------------|--------------------|
| UI | `app.py` (Streamlit) | `frontend/` (React + TailwindCSS) |
| API | None (direct Python imports) | `backend/` (FastAPI — `/predict`, `/analyze`, `/optimize`) |
| AI Logic | `src/` | `src/` — **zero changes** |
| Deployment | `streamlit run app.py` | `docker-compose up` (Frontend + Backend containers) |
| Styling | Streamlit defaults | Premium glassmorphism, dark mode, micro-animations |

### Why This Works
The `src/` folder is designed as a **pure Python library** with no UI dependencies. FastAPI endpoints simply `import` the same functions that Streamlit currently uses. No rewriting of AI logic needed.

> **Bottom line:** Build the science now (Phases 1–6) + ship a working demo (Phase 7 with Streamlit). Upgrade the frontend/backend later without touching a single line of AI code.

---

## 7. Development Methodology: Test-Driven, Verify-at-Every-Stage

We follow a **build → verify → proceed** approach. No phase begins until the previous phase passes its verification checklist. This ensures we catch errors early (bad SMILES, scaling bugs, NaN propagation) instead of debugging them 10 phases downstream.

```
For EVERY module:
  1. Write the function
  2. Write a quick test (assert / print check / notebook cell)
  3. Run it on a small sample (100 molecules, 1 endpoint)
  4. Verify output shape, type, and sanity
  5. Only then proceed to the next module
```

---

## 8. Phase-by-Phase Implementation

---

### Phase 1 — Environment + Data (Day 1–2)

**Goal:** Set up Conda, load data, validate SMILES.

| Task | File | Details |
|------|------|---------|
| Create Conda env | `environment.yml` | `conda env create -f environment.yml` — RDKit CANNOT be pip-installed |
| Load Tox21 | `data/tox21.csv` | 7,831 compounds, 12 binary labels |
| SMILES validation gate | `src/featurize.py` | `Chem.MolFromSmiles()` returns `None` silently for invalid inputs — run this FIRST, always |
| Download OCHEM alerts | `data/ochem_alerts.json` | Brenk + PAINS SMARTS patterns for structural alert validation |
| Generate DrugBank FPs | `data/drugbank_fps.pkl` | Morgan FPs for coverage analysis in Phase 6 |

> **⚠️ WARNING:** Every downstream function assumes valid SMILES. Invalid SMILES propagate as `IndexError` or `NaN` loss hours into training.

**Deliverable:** Conda env running, Tox21 loaded, invalid SMILES dropped.

**✅ Verification Checkpoint (must pass before Phase 2):**
```python
# Run these checks in a notebook or terminal:
assert df.shape[0] > 7000, "Dataset too small — load failed"
assert 'smiles' in df.columns, "Missing SMILES column"
assert all(Chem.MolFromSmiles(s) is not None for s in df['smiles']), "Invalid SMILES remain"
assert df[TARGET_COLS].isin([0, 1, -1]).all().all(), "Labels must be 0, 1, or -1 (NaN sentinel)"
print(f"✅ Phase 1 PASSED: {df.shape[0]} valid compounds, {len(TARGET_COLS)} endpoints")
```

---

### Phase 2 — Multi-Representation Featurization + Scaffold Split (Day 3–4)

**Goal:** Empirically justify representation choice. This is novel — Zhang et al. (2025) describes these theoretically, you are the first to compare them empirically on Tox21 with AUPRC.

| Task | File | Details |
|------|------|---------|
| Implement 5 representations | `src/featurize.py` | ECFP4_1024, ECFP4_2048, ECFP6_2048, MACCS (166 bits), RDKit_FP, ECFP+Desc |
| Morgan FP with bitInfo capture | `src/featurize.py` | Declare `bit_info = {}` OUTSIDE the function call — or SHAP-to-atom mapping fails silently in Phase 6 |
| Descriptor scaling | `src/featurize.py` | If using ECFP+Desc, StandardScaler on TRAIN descriptors ONLY — unscaled MW destroys gradients |
| Scaffold splitting | `src/scaffold_split.py` | `MurckoScaffold` — ensures structural separation between train/val/test. Mandatory, not optional |
| Ablation notebook | `notebooks/03_Featurization_Ablation.ipynb` | Train same XGBoost model with each representation, report AUPRC per endpoint |

> **⚠️ WARNING:** Random splitting on molecular data causes structural leakage — inflates AUC falsely. Use Murcko scaffold split always.

**Deliverable:** 5-representation comparison table (AUPRC). Result: "ECFP4_2048 achieves best macro AUPRC and is the only representation compatible with the SHAP attribution pathway."

**✅ Verification Checkpoint (must pass before Phase 3):**
```python
# Test featurization on a known molecule
fp, bit_info = smiles_to_morgan_with_info('c1ccccc1')  # Benzene
assert fp.shape == (2048,), f"Wrong FP shape: {fp.shape}"
assert len(bit_info) > 0, "bitInfo is empty — SHAP mapping will fail in Phase 6"
assert fp.dtype == np.int8 or fp.dtype == np.uint8, "FP should be binary"

# Test scaffold split
assert len(set(train_idx) & set(test_idx)) == 0, "Train/test overlap — structural leakage!"
print(f"✅ Phase 2 PASSED: FP shape={fp.shape}, {len(bit_info)} active bits, no scaffold leakage")
```

---

### Phase 3 — Geometric Imbalance Analysis + Focal Loss (Day 5)

**Goal:** Quantify the geometric imbalance per endpoint — a finding no prior Tox21 paper has quantified. This IS the primary research contribution.

| Task | File | Details |
|------|------|---------|
| Intraclass Tanimoto cohesion | `src/geometric_imbalance.py` | Measure pairwise Tanimoto within toxic class vs non-toxic class per endpoint |
| SMOTE validity map | `src/geometric_imbalance.py` | If `cohesion_ratio > 1.5` → SMOTE is invalid for that endpoint (just reinforces existing cluster) |
| Per-endpoint Focal Loss | `src/focal_loss.py` | `PerEndpointFocalLoss` in PyTorch — `FL(p) = -α(1-p)^γ * log(p)` with NaN masking |
| SMOTE in embedding space | `src/focal_loss.py` | Only for endpoints where `smote_valid == True` — use ADASYN + TomekLinks on learned embeddings, NOT raw fingerprints |

> **⚠️ CAUTION:** SMOTE directly on Morgan fingerprints interpolates binary vectors → `[0.5, 0.5, ...]` which is NOT a valid fingerprint. Always SMOTE in continuous embedding space.

> **ℹ️ IMPORTANT:** Do NOT use standard BCE or `class_weight='balanced'` as primary. Focal Loss (γ=2) with per-endpoint alpha weights is the correct approach.

**Deliverable:** Per-endpoint cohesion table + SMOTE validity map. This table IS a publishable research result.

**✅ Verification Checkpoint (must pass before Phase 4):**
```python
# Test geometric imbalance on one endpoint
result = compute_intraclass_cohesion(smiles_train, y_train_NR_AR, 'NR-AR')
assert 0 < result['toxic_cohesion'] < 1, "Cohesion out of range"
assert 0 < result['nontoxic_cohesion'] < 1, "Cohesion out of range"
assert isinstance(result['smote_valid'], bool), "SMOTE flag must be boolean"

# Test Focal Loss shape
loss_fn = PerEndpointFocalLoss(pos_weights, TARGET_COLS, gamma=2.0)
dummy_logits = torch.randn(32, 12)
dummy_targets = torch.randint(0, 2, (32, 12)).float()
loss = loss_fn(dummy_logits, dummy_targets)
assert loss.dim() == 0, "Loss must be a scalar"
assert not torch.isnan(loss), "Loss is NaN — check alpha computation"
print(f"✅ Phase 3 PASSED: Cohesion ratio={result['cohesion_ratio']}, Loss={loss.item():.4f}")
```

---

### Phase 4 — ToxNet Training with OPTUNA (Day 6–9)

**Goal:** Train the multi-task neural network.

| Task | File | Details |
|------|------|---------|
| ToxNet architecture | `src/model.py` | Shared backbone `[1024→512→256]` with BatchNorm + GELU + Dropout → 12 independent task heads |
| OPTUNA tuning | `src/train.py` | Tune `lr`, `dropout`, `batch_size`, `focal_gamma`, `shared_dims`, `weight_decay` — 50 trials maximizing macro AUPRC |
| Training loop | `src/train.py` | PyTorch training with `PerEndpointFocalLoss`, NaN label masking via -1 sentinel |
| Training notebook | `notebooks/04_Training.ipynb` | Full run with best representation from Phase 2 |

> **⚠️ CAUTION:** Do NOT tune XGBoost tree parameters (`n_estimators`, `max_depth`) for ToxNet — those are tree-based hyperparameters. ToxNet is a PyTorch neural network. Tune neural parameters instead.

**Deliverable:** Trained `toxnet_final.pt` with optimized hyperparameters.

**✅ Verification Checkpoint (must pass before Phase 5):**
```python
# Test model forward pass
model = ToxNet(input_dim=2048)
test_input = torch.randn(16, 2048)
output = model(test_input)
assert output.shape == (16, 12), f"Wrong output shape: {output.shape}"

# Test embeddings extraction
emb = model.get_embeddings(test_input)
assert emb.shape == (16, 256), f"Wrong embedding shape: {emb.shape}"

# Verify model saved & loadable
model_loaded = ToxNet()
model_loaded.load_state_dict(torch.load('models/toxnet_final.pt'))
assert model_loaded(test_input).shape == (16, 12)
print(f"✅ Phase 4 PASSED: Model outputs {output.shape}, embeddings {emb.shape}, weights loadable")
```

---

### Phase 5 — Evaluation Suite (Day 10–11)

**Goal:** Evaluate rigorously. Primary metric is AUPRC, NEVER accuracy.

| Task | File | Details |
|------|------|---------|
| Full metric suite | `src/evaluate.py` | AUPRC (primary), AUROC, F1, MCC per endpoint + AUPRC Lift (vs random baseline) |
| Per-endpoint threshold tuning | `src/evaluate.py` | Recall floor = 0.85 — maximize precision subject to this floor |
| Geometric failure correlation | `notebooks/05_Evaluation.ipynb` | Correlate geometric cohesion ratio with AUPRC drops — this proves Claim 1 |

> **⚠️ CAUTION:** AUPRC is lower than AUROC numerically — that's expected. A model that looks great on AUROC but collapses on AUPRC is gaming the majority class.

### Benchmark Targets

| Stage | Expected Macro AUPRC | Expected Macro AUROC |
|-------|:---:|:---:|
| Logistic Regression (baseline) | 0.20–0.28 | 0.72–0.76 |
| XGBoost + ECFP4 | 0.35–0.45 | 0.82–0.86 |
| ToxNet + Focal Loss + OPTUNA | 0.42–0.52 | 0.84–0.88 |

**Deliverable:** AUPRC/AUROC/MCC table, per-endpoint threshold calibration, geometric failure correlation plot.

**✅ Verification Checkpoint (must pass before Phase 6):**
```python
# Test evaluation produces valid metrics
results_df = evaluate_all_endpoints(Y_test, Y_proba, Y_pred, TARGET_COLS)
assert results_df.shape[0] == 13, "Should have 12 endpoints + 1 macro avg row"
assert all(0 <= v <= 1 for v in results_df['AUPRC'].dropna()), "AUPRC out of [0,1]"
assert all(0 <= v <= 1 for v in results_df['AUROC'].dropna()), "AUROC out of [0,1]"

# Sanity: AUPRC should beat random baseline (prevalence)
for _, row in results_df.iloc[:-1].iterrows():
    prev = float(row['Prevalence'].strip('%')) / 100
    assert row['AUPRC'] > prev, f"{row['Endpoint']}: AUPRC {row['AUPRC']} <= prevalence {prev}"

# Test threshold tuning
thresh = tune_threshold(y_true_NR_AR, y_prob_NR_AR)
assert 0 < thresh['threshold'] < 1, "Threshold out of range"
assert thresh['recall'] >= 0.80, f"Recall {thresh['recall']} below floor"
print(f"✅ Phase 5 PASSED: Macro AUPRC={results_df.iloc[-1]['AUPRC']:.4f}")
```

---

### Phase 6 — Prescription Pipeline (Day 12–13)

**Goal:** The 4-Step Corrected Flow — this is what makes the project novel.

| Task | File | Details |
|------|------|---------|
| SHAP attribution | `src/shap_validator.py` | `shap.TreeExplainer` or `shap.DeepExplainer` → top SHAP bits → map to atom indices via RDKit `bitInfo` |
| Alert cross-validation | `src/shap_validator.py` | Check if SHAP-flagged fragment matches Brenk/PAINS/OCHEM alerts. If no match → flag as "LOW CONFIDENCE" |
| ChEMBL cache builder | `src/bioisostere.py` | **Offline script** — run ONCE, not in dashboard. ChEMBL rate-limits and causes crashes. Save as `chembl_cache.pkl` |
| SAScore filter | `src/bioisostere.py` | `SAScore < 4.0` = viable. Discard impractical suggestions |
| ADME preservation filter | `src/bioisostere.py` | `|ΔLogP| < 0.5` and `|ΔMW| < 25 Da` to prevent destroying drug absorption |
| OOD detection | `src/uncertainty.py` | Tanimoto to training set — if `max_sim < 0.4` → predictions unreliable |
| Temperature scaling | `src/uncertainty.py` | Per-endpoint calibration on validation logits |
| Pareto evaluation | `src/pareto.py` | Check all 12 endpoints: DOMINATES / TRADE-OFF / DOMINATED / NO_CHANGE |
| Full pipeline | `src/prescription_pipeline.py` | Chains all steps: predict → SHAP → validate → suggest → filter → re-predict → Pareto rank |

### The 4-Step Corrected Flow

```
Step 1:  Predict all 12 endpoints → SHAP per endpoint
Step 1b: Cross-validate SHAP against Brenk + PAINS + OCHEM alerts
         → Flag low-confidence attributions (Zhang et al. open problem)
         → Only proceed if alert overlap confirmed

Step 2:  Map top SHAP bit → molecular fragment (RDKit atom-info)

Step 3:  Query ChEMBL cache for compounds containing replacement fragment
         → Filter A (Synthesizability): SAScore < 4.0
         → Filter B (Pharmacokinetics): |ΔLogP| < 0.5 and |ΔMW| < 25 Da
         → Flag: Tanimoto to DrugBank < 0.4 (out-of-pharma-space warning)

Step 4:  Re-predict all 12 endpoints for each viable candidate
         → Temperature-scaled calibrated probabilities
         → Apply "Intended Mechanism" override
         → OOD detection via Tanimoto to training set
         → Pareto dominance classification
         → Output: sorted Pareto front with ADME metrics and uncertainty intervals
```

> **⚠️ WARNING:** The prescription pipeline must query ChEMBL as an offline cache, NOT live in the dashboard. ChEMBL rate-limits cause the dashboard to hang or crash.

**Deliverable:** End-to-end: SMILES → Pareto front output.

**✅ Verification Checkpoint (must pass before Phase 7):**
```python
# Test SHAP validator
validation = validate_shap_attribution('Nc1ccccc1', top_bit=42, atom_indices=[0,1])
assert validation['shap_confidence'] in ['HIGH', 'MEDIUM', 'LOW']
assert validation['recommendation'] in ['PROCEED', 'DO NOT SUBSTITUTE — attribution unvalidated']

# Test Pareto evaluation
orig = np.array([0.8, 0.7, 0.3, 0.6, 0.5, 0.4, 0.2, 0.1, 0.9, 0.5, 0.3, 0.6])
mod  = np.array([0.3, 0.2, 0.3, 0.5, 0.4, 0.3, 0.2, 0.1, 0.8, 0.4, 0.2, 0.5])
result = evaluate_swap_pareto(orig, mod, TARGET_COLS)
assert result['status'] in ['DOMINATES', 'TRADE-OFF', 'DOMINATED', 'NO_CHANGE']

# Test full pipeline end-to-end
pipeline_result = corrected_prescription_pipeline('Nc1ccccc1', model, ...)
assert 'candidates' in pipeline_result
assert 'scope_disclaimer' in pipeline_result
print(f"✅ Phase 6 PASSED: {len(pipeline_result['candidates'])} candidates generated")
```

---

### Phase 7 — Dashboard (Day 14–16)

**Goal:** Build the Streamlit UI with all panels.

| Panel | Content | Technical Source |
|-------|---------|-----------------|
| Molecule viewer | 2D structure + flagged atoms | RDKit Draw + SHAP atoms |
| Therapeutic Override | Dropdown to ignore specific endpoints | User Input |
| Radar chart | 12-endpoint toxicity profile | Plotly radar |
| Prescription table | Ranked Pareto candidates | Pipeline Step 4 |
| ADME Preservation | ΔLogP, ΔMolecular Weight | RDKit Descriptors |
| Endpoint delta grid | Per-endpoint before/after | Pareto evaluation |
| Uncertainty bands | [lower, upper] per endpoint | Temperature scaling + OOD |
| SHAP confidence | HIGH/MEDIUM/LOW + alert name | Alert cross-validation |
| Coverage warning | Pharma-space proximity | DrugBank Tanimoto |
| Scope disclaimer | In-vitro vs In-vivo metabolite warning | Hardcoded regulatory text |

**Deliverable:** Streamlit app (`app.py`) with all panels working.

**✅ Verification Checkpoint (final):**
```
1. Run: streamlit run app.py
2. Input known toxic SMILES: "Nc1ccccc1" (aniline — known structural alert)
3. Verify: Radar chart shows 12-endpoint scores
4. Verify: SHAP heatmap highlights the -NH2 group
5. Verify: SHAP confidence shows "HIGH" (aniline alert detected)
6. Verify: At least 1 bioisostere candidate appears in Pareto table
7. Verify: Uncertainty bands show [lower, upper] per endpoint
8. Verify: Scope disclaimer is visible
9. Verify: Therapeutic override dropdown excludes selected endpoints
```

---

## 8. Key Academic References

| Paper | Where Used |
|-------|-----------|
| Zhang et al. (2025). *AI-Driven Drug Toxicity Prediction.* | Representation table, SHAP open problem, regulatory framing |
| Feitosa et al. *Cyto-Safe.* | Primary prior art — differentiation argument |
| Barua et al. *Organ Toxicity ML.* | SMOTE+OPTUNA validation, benchmark AUC values |
| Rogers & Hahn (2010). *ECFP.* JCIM. | Morgan fingerprint justification |
| Lundberg & Lee (2017). *SHAP.* NeurIPS. | Explainability method |
| Jiang et al. (2021). *GNNs vs fingerprints.* | Justification for ECFP over GNN |
| Lin et al. (2017). *Focal Loss.* ICCV. | Focal Loss theoretical basis |
| Wu et al. (2018). *MoleculeNet.* Chemical Science. | Dataset + benchmark metrics |

---

## 9. Timeline Summary

| Phase | Duration | Key Deliverable | Status |
|-------|----------|-----------------|--------|
| Phase 1: Environment + Data | Day 1–2 | Conda env, Tox21 loaded, OCHEM alerts | ⬜ |
| Phase 2: Featurization Ablation | Day 3–4 | 5-representation comparison table (AUPRC) | ⬜ |
| Phase 3: Geometric Imbalance | Day 5 | Per-endpoint cohesion table + SMOTE validity map | ⬜ |
| Phase 4: Training | Day 6–9 | ToxNet + OPTUNA on best representation | ⬜ |
| Phase 5: Evaluation | Day 10–11 | AUPRC/AUROC/MCC table, threshold calibration | ⬜ |
| Phase 6: Prescription Pipeline | Day 12–13 | End-to-end: SMILES → Pareto front output | ⬜ |
| Phase 7: Dashboard | Day 14–16 | Streamlit app + all panels | ⬜ |

---

## 10. The Publishable Claim

> *"To our knowledge, this is the first open-source pipeline to combine SHAP-guided bioisostere substitution, multi-endpoint Pareto optimization, synthesizability filtering, and out-of-distribution uncertainty quantification for the Tox21 benchmark — along with a systematic characterization of pipeline failure modes stratified by pharmaceutical chemical space coverage."*
