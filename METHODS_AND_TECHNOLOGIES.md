# Methods and Technologies Used in ToxNet Prescription Engine

This file is a compact reference for the methods, tools, and libraries used in the project.
It is written so someone can quickly understand what the repository actually does and in what order the workflow is built.

## 1. Project Goal

The project is a molecular toxicity prediction and prescription system for the Tox21 benchmark.
It does not stop at classification. It also explains toxicity, proposes safer replacements, and ranks them.

## 2. Main Methods Used

| Stage | Method | Where It Is Used | Purpose |
|---|---|---|---|
| Data cleaning | SMILES validation | `01_data_cleaning.ipynb`, `src/api.py` | Reject invalid molecules early and avoid downstream failures |
| Data cleaning | Salt stripping and normalization | `01_data_cleaning.ipynb` | Standardize molecules before featurization |
| Splitting | Murcko scaffold split | `src/scaffold_split.py`, `02_Featurization_and_Split.ipynb` | Prevent scaffold leakage between train/validation/test |
| Featurization | Morgan fingerprints (ECFP4/ECFP6) | `src/featurize.py`, `02_Featurization_and_Split.ipynb` | Convert molecules to fixed-length binary vectors |
| Explainability support | RDKit bitInfo capture | `src/featurize.py`, `src/bioisostere.py` | Map fingerprint bits back to atom environments and fragments |
| Feature study | Multi-representation ablation | `02_Featurization_and_Split.ipynb` | Compare fingerprint and descriptor options rather than assuming one representation |
| Imbalance analysis | Intraclass Tanimoto cohesion | `03_Geometric_Imbalance.ipynb` | Measure how clustered each endpoint's toxic class is |
| Imbalance handling | SMOTE/ADASYN in embedding space | `src/train.py`, `03_Geometric_Imbalance.ipynb` | Generate synthetic samples only where interpolation is valid |
| Imbalance handling | Tomek Links cleanup | `src/train.py` | Remove noisy majority samples after augmentation |
| Model training | Two-pass bootstrap training | `src/train.py`, `04_Training.ipynb` | Train a fast embedding model first, then a final predictor |
| Model training | ToxNetLite pass | `src/model.py`, `src/train.py` | Learn continuous embeddings from raw fingerprints |
| Model training | ToxNet pass | `src/model.py`, `src/train.py` | Predict all 12 endpoints from learned embeddings |
| Model training | Focal Loss with NaN masking | `src/focal_loss.py`, `src/train.py` | Handle severe imbalance and missing labels |
| Model tuning | OPTUNA hyperparameter search | `src/train.py`, `04_Training.ipynb` | Search learning rate, dropout, batch size, gamma, width, weight decay |
| Thresholding | Per-endpoint threshold calibration | `src/explain.py` | Choose endpoint-specific classification thresholds |
| Explainability | SHAP expected gradients | `src/explain.py`, `05_Explainability.ipynb` | Attribute endpoint predictions to fingerprint bits |
| Safety validation | PAINS and Brenk alert checks | `src/prescription_pipeline.py` | Confirm whether SHAP-highlighted fragments match known alert motifs |
| Fragment mapping | Top SHAP bit to fragment extraction | `src/bioisostere.py`, `src/prescription_pipeline.py` | Convert important bits into chemical substructures |
| Candidate generation | Local ChEMBL cache lookup | `src/bioisostere.py`, `scripts/build_chembl_cache.py` | Retrieve replacement fragments without live API bottlenecks |
| Candidate filtering | Synthesizability consensus | `src/bioisostere.py` | Filter by SAScore and SCScore |
| Candidate filtering | ADME delta filtering | `src/bioisostere.py` | Keep candidates close in logP and molecular weight |
| Candidate ranking | Pareto dominance ranking | `src/pareto.py` | Rank candidates by toxicity, synthesizeability, and desirability |
| Reliability | OOD similarity / Tanimoto proximity | `src/pareto.py` | Flag candidates far from the training distribution |
| Generalization check | Literature-based generalization tests | `scripts/check_generalization.py` | Compare predictions on known molecules outside the training set |
| End-to-end testing | Pipeline smoke test | `scripts/test_pipeline.py` | Verify the full SMILES → report flow |

## 3. Core Modeling Methods

### Two-pass training

The model is trained in two stages:

1. `ToxNetLite` learns from raw 4096-bit Morgan fingerprints.
2. `ToxNet` is trained on the 256-dimensional embeddings created by `ToxNetLite`.

This lets the project separate representation learning from final classification.

### Multi-task learning

The predictor handles all 12 Tox21 endpoints at once.
Each endpoint has its own head, while the backbone is shared.

### Class imbalance handling

The project uses a combination of:

- focal loss,
- per-endpoint weighting,
- SMOTE/ADASYN in embedding space,
- and endpoint-specific threshold calibration.

### Explainability and prescription

The pipeline does not stop at "toxic" or "safe".
It also:

- maps SHAP to fingerprint bits,
- converts bits to fragments,
- validates fragments against structural alerts,
- searches for replacement chemistry,
- and ranks the resulting candidates.

## 4. Technologies Used

| Area | Technology | Role |
|---|---|---|
| Language | Python | Backend, training, cheminformatics, analysis |
| Language | TypeScript | Frontend app code |
| UI Framework | React | Dashboard UI |
| Build Tool | Vite | Frontend build and dev server |
| Styling | Tailwind CSS | Visual design and layout |
| Backend Framework | FastAPI | REST API for analysis requests |
| ML Framework | PyTorch | Neural network training and inference |
| Cheminformatics | RDKit | SMILES parsing, fingerprints, scaffolds, fragments, descriptors |
| Explainability | SHAP | Attribution of endpoint predictions |
| Imbalance Handling | imbalanced-learn | ADASYN and Tomek Links |
| Hyperparameter Tuning | OPTUNA | Search over neural hyperparameters |
| Classical ML Metrics | scikit-learn | AUPRC, F1, precision-recall utilities |
| Candidate Search | chembl-webresource-client | ChEMBL lookup and cache building |
| Synthesizability | SCScore / `scscore` submodule | Candidate synthesis difficulty scoring |
| Synthesizability | SAScore | RDKit SA score for synthetic accessibility |
| 3D / Visualization | React dashboard + chemistry rendering | Human-readable report output |
| Containerization | Docker / docker-compose | Reproducible environment and GPU-ready setup |
| Notebook Workflow | JupyterLab | Research notebooks and experiments |
| Plotting | matplotlib / Plotly / Recharts | Analysis and dashboard visualization |

## 5. Important Libraries and Why They Matter

- `rdkit`: required for almost every chemistry step in the repository.
- `torch`: used for the full ToxNet model and ToxNetLite.
- `shap`: used for attribution and explanation.
- `optuna`: used to tune the two-pass training pipeline.
- `imbalanced-learn`: used for ADASYN and Tomek Links.
- `fastapi` and `uvicorn`: used to serve the model API.
- `react`, `typescript`, `vite`, `tailwindcss`: used for the frontend report dashboard.
- `chembl-webresource-client`: used for fragment-based candidate lookup during cache building.

## 6. Workflow Order

If you want the project in the correct order, use this sequence:

1. Clean and standardize the molecules.
2. Split by scaffold.
3. Build fingerprints and descriptor features.
4. Measure geometric imbalance.
5. Train `ToxNetLite`.
6. Extract embeddings and augment them when valid.
7. Train `ToxNet`.
8. Calibrate thresholds and compute SHAP.
9. Run prescription inference.
10. Rank replacement candidates and show them in the dashboard.

## 7. Methods Your Friend Should Not Miss

These are the methods that are easy to overlook if someone follows a different roadmap:

- Murcko scaffold split
- Morgan bitInfo capture
- two-pass bootstrap training
- focal loss with NaN masking
- ADASYN in embedding space
- threshold calibration per endpoint
- SHAP expected gradients
- PAINS and Brenk validation
- fragment extraction from SHAP bits
- local ChEMBL cache lookup
- SAScore and SCScore consensus filtering
- ADME delta filtering
- Pareto ranking
- OOD similarity check

## 8. How to Run the Project

### Backend

```powershell
cd Project_Implementation
python src/api.py
```

### Frontend

```powershell
cd Project_Implementation/frontend
npm install
npm run dev
```

### Docker

```bash
cd Project_Implementation
docker compose up --build
```

## 9. Suggested References Inside the Repo

If someone wants to study the project properly, the best files to read are:

1. `README.md`
2. `Project_Implementation/src/train.py`
3. `Project_Implementation/src/model.py`
4. `Project_Implementation/src/explain.py`
5. `Project_Implementation/src/prescription_pipeline.py`
6. `Project_Implementation/src/bioisostere.py`
7. `Project_Implementation/src/pareto.py`
8. `Project_Implementation/notebooks/04_Training.ipynb`

## 10. Short Summary

The project combines:

- cheminformatics preprocessing,
- scaffold-aware data splitting,
- two-pass deep learning,
- imbalance correction,
- explainability,
- fragment-to-replacement search,
- and a chemist-facing dashboard.

That is the complete methods-and-technologies stack used in the repository.
