# 🧪 Full-Stack Product: Implementation Plan

> **This document is the UPGRADE PATH.** It describes how to take the AI core you already built (Phases 1–6 in `plan.md`) and wrap it in a production-grade FastAPI + React web application.
>
> **The `src/` folder does not change.** Only the UI and API layers are added on top.

---

## 1. How This Relates to plan.md

| | plan.md (Current Work) | implementation_plan.md (This File) |
|---|---|---|
| **Focus** | Build the AI science | Wrap the AI in a product |
| **UI** | Streamlit (`app.py`) | React/Next.js (`frontend/`) |
| **API** | None (direct Python imports) | FastAPI (`backend/`) |
| **AI Logic** | `src/` | `src/` — **zero changes** |
| **Deploy** | `streamlit run app.py` | `docker-compose up` |
| **When** | NOW (Day 1–16) | LATER (after AI core is complete) |

```
plan.md builds this:           This file adds this on top:
┌──────────────┐               ┌──────────────────────┐
│   app.py     │  ──UPGRADE──► │   frontend/ (React)  │
│  (Streamlit) │               │   backend/  (FastAPI) │
├──────────────┤               ├──────────────────────┤
│    src/      │               │    src/  (SAME)       │
│   data/      │               │   data/  (SAME)       │
│  models/     │               │  models/ (SAME)       │
└──────────────┘               └──────────────────────┘
```

---

## 2. Full-Stack Tech Stack

| Layer | Technology | Purpose |
|-------|-----------|---------|
| **AI Core** | PyTorch, RDKit, SHAP, Optuna | The science — unchanged from `plan.md` |
| **Backend API** | FastAPI + Uvicorn | Serves AI logic as REST endpoints |
| **Data Validation** | Pydantic | SMILES validation, request/response schemas |
| **Frontend** | React / Next.js | Premium interactive dashboard |
| **Styling** | TailwindCSS | Glassmorphism, dark mode, micro-animations |
| **Charts** | Plotly.js / Recharts | Radar chart, Pareto front, endpoint delta grid |
| **Molecule Rendering** | RDKit-JS (or Ketcher) | 2D structure with SHAP heatmap overlay |
| **State Management** | Zustand or Redux | Client-side state for molecule data |
| **API Client** | Axios / Fetch | Frontend → Backend communication |
| **Containerization** | Docker + docker-compose | Orchestrates frontend + backend |
| **Environment** | Conda (inside Docker) | Backend container manages RDKit/PyTorch deps |

---

## 3. Full-Stack File Structure

```
PDS_exp10/
│
├── src/                             # ✅ AI CORE — IDENTICAL TO plan.md
│   ├── featurize.py                 # All 5 fingerprint representations
│   ├── scaffold_split.py            # Murcko scaffold stratified split
│   ├── geometric_imbalance.py       # Intraclass cohesion analysis
│   ├── focal_loss.py                # PerEndpointFocalLoss (PyTorch)
│   ├── model.py                     # ToxNet architecture (PyTorch)
│   ├── train.py                     # Training loop + OPTUNA
│   ├── evaluate.py                  # Full metric suite
│   ├── shap_validator.py            # SHAP × alert cross-validation
│   ├── bioisostere.py               # ChEMBL query + SAScore filter
│   ├── pareto.py                    # Pareto dominance evaluation
│   ├── uncertainty.py               # Temperature scaling + OOD detection
│   └── prescription_pipeline.py     # Full corrected 4-step pipeline
│
├── data/                            # ✅ DATA — IDENTICAL TO plan.md
│   ├── tox21.csv
│   ├── drugbank_fps.pkl
│   └── ochem_alerts.json
│
├── models/                          # ✅ TRAINED WEIGHTS — IDENTICAL
│   ├── toxnet_final.pt
│   └── temperatures.pkl
│
├── notebooks/                       # ✅ RESEARCH NOTEBOOKS — IDENTICAL
│   ├── 01_EDA.ipynb
│   ├── 02_Geometric_Imbalance.ipynb
│   ├── 03_Featurization_Ablation.ipynb
│   ├── 04_Training.ipynb
│   ├── 05_Evaluation.ipynb
│   └── 06_Prescription.ipynb
│
├── backend/                         # 🆕 FastAPI Gateway
│   ├── app/
│   │   ├── __init__.py
│   │   ├── main.py                  # FastAPI entry point + CORS
│   │   ├── core/
│   │   │   ├── config.py            # App settings (model paths, API keys)
│   │   │   └── model_manager.py     # Load & cache ToxNet + temperatures on startup
│   │   ├── api/
│   │   │   ├── router.py            # Main API router
│   │   │   └── endpoints/
│   │   │       ├── predict.py       # POST /api/predict    → 12-endpoint scores + uncertainty
│   │   │       ├── analyze.py       # POST /api/analyze    → SHAP + alert validation
│   │   │       └── optimize.py      # POST /api/optimize   → bioisostere suggestions + Pareto
│   │   └── schemas/
│   │       ├── molecule.py          # Pydantic: SMILES input validation
│   │       └── prediction.py        # Pydantic: response models (scores, Pareto, alerts)
│   ├── tests/
│   │   ├── test_predict.py          # Pytest: known toxic → correct flags
│   │   ├── test_analyze.py          # Pytest: SHAP + alert overlap
│   │   └── test_optimize.py         # Pytest: bioisostere pipeline
│   ├── requirements.txt             # fastapi, uvicorn, pydantic, python-multipart
│   └── Dockerfile                   # Python 3.10 + Conda for RDKit
│
├── frontend/                        # 🆕 React / Next.js Dashboard
│   ├── src/
│   │   ├── app/                     # Next.js App Router (or pages/)
│   │   │   ├── layout.tsx           # Root layout with dark mode + fonts
│   │   │   ├── page.tsx             # Landing page — SMILES input
│   │   │   └── results/
│   │   │       └── page.tsx         # Results dashboard (all panels)
│   │   ├── components/
│   │   │   ├── Molecule/
│   │   │   │   ├── StructureViewer.tsx   # RDKit-JS 2D renderer
│   │   │   │   └── ShapHeatmap.tsx       # SHAP atom overlay on structure
│   │   │   ├── Dashboard/
│   │   │   │   ├── RadarChart.tsx        # 12-endpoint toxicity profile (Plotly)
│   │   │   │   ├── ParetoFront.tsx       # Interactive Pareto chart
│   │   │   │   ├── EndpointDelta.tsx     # Before/after per-endpoint grid
│   │   │   │   └── UncertaintyBands.tsx  # [lower, upper] per endpoint
│   │   │   ├── Prescription/
│   │   │   │   ├── CandidateTable.tsx    # Ranked Pareto candidates
│   │   │   │   ├── ADMEPanel.tsx         # ΔLogP, ΔMW preservation
│   │   │   │   └── ScopeDisclaimer.tsx   # Regulatory warning
│   │   │   └── Layout/
│   │   │       ├── Navbar.tsx
│   │   │       ├── Sidebar.tsx
│   │   │       └── ThemeToggle.tsx       # Light/dark mode switch
│   │   ├── services/
│   │   │   └── api.ts               # Axios wrappers: predict(), analyze(), optimize()
│   │   ├── hooks/
│   │   │   ├── usePrediction.ts     # Custom hook for /api/predict
│   │   │   └── useOptimize.ts       # Custom hook for /api/optimize
│   │   ├── store/
│   │   │   └── moleculeStore.ts     # Zustand store for current molecule state
│   │   └── styles/
│   │       └── globals.css          # Premium design system
│   ├── public/                      # Favicons, fonts, images
│   ├── tailwind.config.js           # Custom colors, glassmorphism utilities
│   ├── package.json
│   └── Dockerfile                   # Node 20 + production build
│
├── Documents/                       # Research & Specs (already exists)
│   ├── research_papers/
│   ├── literature_grounded_spec.md
│   ├── roadmap.html
│   ├── corrected_pipeline.md.resolved
│   ├── imbalance_strategy.md.resolved
│   ├── novelty_thought_process.md.resolved
│   └── tox21_market_usp.md.resolved
│
├── app.py                           # Streamlit dashboard (Phase 7 of plan.md — keep as backup)
├── environment.yml                  # Conda env for AI/ML work
├── docker-compose.yml               # 🆕 Orchestrates frontend + backend containers
├── .gitignore
├── .env                             # API keys (DO NOT COMMIT)
├── plan.md                          # ← AI core roadmap (current work)
├── implementation_plan.md           # ← This file (full-stack upgrade)
└── README.md
```

---

## 4. API Endpoints (Backend)

| Method | Endpoint | Input | Output | Maps to `src/` |
|--------|----------|-------|--------|-----------------|
| `POST` | `/api/predict` | `{ "smiles": "CCO" }` | 12 endpoint scores + uncertainty intervals + OOD flag | `src/model.py` → `src/uncertainty.py` |
| `POST` | `/api/analyze` | `{ "smiles": "CCO" }` | SHAP top bits + atom indices + alert validation + confidence level | `src/shap_validator.py` |
| `POST` | `/api/optimize` | `{ "smiles": "CCO", "override_endpoints": ["NR-AR"] }` | Pareto-ranked bioisostere candidates + ADME deltas + scope disclaimer | `src/prescription_pipeline.py` |
| `GET`  | `/api/health` | — | `{ "status": "ok", "model_loaded": true }` | — |

### Example: `/api/predict` Response

```json
{
  "smiles": "Nc1ccccc1",
  "valid": true,
  "predictions": {
    "NR-AR":     { "score": 0.82, "interval": [0.71, 0.93], "ood": false, "confidence": "HIGH" },
    "NR-AR-LBD": { "score": 0.31, "interval": [0.19, 0.43], "ood": false, "confidence": "HIGH" },
    "SR-MMP":    { "score": 0.58, "interval": [0.43, 0.73], "ood": false, "confidence": "MEDIUM" }
  },
  "flagged_endpoints": ["NR-AR"],
  "ood_warning": null
}
```

---

## 5. Frontend Dashboard Panels

| Panel | Component | Data Source | Notes |
|-------|-----------|------------|-------|
| SMILES Input | `page.tsx` | User input | Auto-validate via `/api/predict` |
| Molecule Viewer | `StructureViewer.tsx` | RDKit-JS | 2D structure with atom highlighting |
| SHAP Heatmap | `ShapHeatmap.tsx` | `/api/analyze` | Color atoms by SHAP attribution score |
| Radar Chart | `RadarChart.tsx` | `/api/predict` | 12-axis toxicity profile (Plotly) |
| Therapeutic Override | Dropdown | User input | Exclude endpoints from optimization |
| Candidate Table | `CandidateTable.tsx` | `/api/optimize` | Ranked by Pareto status: DOMINATES → TRADE-OFF |
| Endpoint Delta Grid | `EndpointDelta.tsx` | `/api/optimize` | Per-endpoint before/after bars |
| ADME Panel | `ADMEPanel.tsx` | `/api/optimize` | ΔLogP, ΔMW, SAScore |
| Uncertainty Bands | `UncertaintyBands.tsx` | `/api/predict` | [lower, upper] per endpoint |
| SHAP Confidence | Part of `ShapHeatmap` | `/api/analyze` | HIGH/MEDIUM/LOW + alert name |
| Coverage Warning | Part of `ADMEPanel` | `/api/optimize` | DrugBank Tanimoto proximity |
| Scope Disclaimer | `ScopeDisclaimer.tsx` | Hardcoded | Regulatory in-vitro vs in-vivo warning |

---

## 6. Upgrade Phases (Post plan.md)

### Phase A: Backend (FastAPI)
*   **Goal:** Expose the `src/` functions as REST endpoints.
*   **Environment:** Conda (same `tox21_env`) + FastAPI/Uvicorn added via pip.
*   **Tasks:**
    *   Create `backend/app/core/model_manager.py` — loads `toxnet_final.pt` and `temperatures.pkl` once on startup, caches in memory.
    *   Create Pydantic schemas for SMILES validation and response typing.
    *   Implement `/api/predict` — calls `src/model.py` and `src/uncertainty.py`.
    *   Implement `/api/analyze` — calls `src/shap_validator.py`.
    *   Implement `/api/optimize` — calls `src/prescription_pipeline.py`.
    *   Write Pytest tests for each endpoint with known toxic/safe molecules.

### Phase B: Frontend (React/Next.js)
*   **Goal:** Build the premium interactive dashboard.
*   **Environment:** Node 20 + npm.
*   **Tasks:**
    *   Initialize Next.js project with TailwindCSS.
    *   Build the SMILES input page with live validation.
    *   Build the results dashboard with all panels from Section 5.
    *   Connect to backend via Axios service layer.
    *   Add dark mode, glassmorphism, micro-animations.

### Phase C: Deployment (Docker)
*   **Goal:** Ship the full product.
*   **Environment:** Docker + docker-compose.
*   **Tasks:**
    *   Write `backend/Dockerfile` — Conda base image + FastAPI.
    *   Write `frontend/Dockerfile` — Node base + production build.
    *   Write `docker-compose.yml` — orchestrates both containers.
    *   Add health check endpoint.
    *   Write deployment README.

---

## 7. docker-compose.yml (Preview)

```yaml
version: '3.8'

services:
  backend:
    build: ./backend
    ports:
      - "8000:8000"
    volumes:
      - ./src:/app/src          # Mount the AI core
      - ./data:/app/data        # Mount data files
      - ./models:/app/models    # Mount trained weights
    environment:
      - MODEL_PATH=/app/models/toxnet_final.pt
      - TEMP_PATH=/app/models/temperatures.pkl
    command: uvicorn app.main:app --host 0.0.0.0 --port 8000

  frontend:
    build: ./frontend
    ports:
      - "3000:3000"
    depends_on:
      - backend
    environment:
      - NEXT_PUBLIC_API_URL=http://backend:8000
```

---

## 8. Summary

| What | Where | When |
|------|-------|------|
| Build the AI science | `plan.md` → Phases 1–6 | **NOW** (Day 1–13) |
| Ship a working demo | `plan.md` → Phase 7 (Streamlit) | **NOW** (Day 14–16) |
| Upgrade to full-stack | This file → Phases A–C | **LATER** |
| AI code changes needed for upgrade | **ZERO** | — |
