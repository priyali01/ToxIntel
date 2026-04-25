# 💬 Planning Session Chat Summary
## Session: Defining Architecture & Creating plan.md + implementation_plan.md
**Date:** 2026-04-25 (22:38 IST – 00:45 IST)

---

## What Happened in This Session

This session was entirely about **planning** — no code was written. We iterated through multiple architectural approaches before settling on the final two-file system.

---

## Key Decisions Made

### 1. Two-File System
We decided on two complementary reference documents:

| File | Purpose | When to Use |
|------|---------|-------------|
| `plan.md` | The AI core science + Streamlit dashboard | **NOW** (Day 1–16) |
| `implementation_plan.md` | Full-stack upgrade (FastAPI + React + Docker) | **LATER** |

### 2. Strictly Follow roadmap.html
After several iterations where the architecture drifted toward over-engineering (FastAPI/React/Docker from the start), we course-corrected to **strictly follow `roadmap.html`**:
- **Model:** ToxNet (PyTorch), NOT XGBoost
- **Dashboard:** Streamlit, NOT React
- **Environment:** Conda, NOT Docker (for development)

### 3. The src/ Folder is a Pure Library
The `src/` folder is designed with **zero UI dependencies**. This means:
- Streamlit `app.py` imports from `src/` directly
- Later, FastAPI endpoints will import from the same `src/` — zero rewrites needed

### 4. Conda for ML, Docker for Deployment (Later)
- **Conda** manages the heavy ML dependencies (RDKit, PyTorch) during development
- **Docker** will be used later to containerize the full-stack product for deployment

---

## Evolution of Architecture (The Journey)

The architecture went through several iterations during this chat:

```
Iteration 1: User asked for file structure
  → I proposed a flat ML pipeline (data/, src/, scripts/)

Iteration 2: User said "this needs frontend and backend too"
  → I proposed ai_core/ + backend/ (FastAPI) + frontend/ (React)

Iteration 3: User said "keep adding detail to ai_core/"
  → I expanded ai_core with detailed sub-modules

Iteration 4: User said "strictly follow roadmap.html"
  → I read the entire roadmap.html and realized it uses
     ToxNet (PyTorch) + Streamlit, NOT XGBoost/React
  → Rewrote everything to match roadmap.html exactly

Iteration 5: User said "plan is scattered in 2 files, make ONE"
  → I read ALL Documents/ files (roadmap.html, corrected_pipeline.md,
     imbalance_strategy.md, novelty_thought_process.md, tox21_market_usp.md)
  → Created one comprehensive plan.md consolidating everything

Iteration 6: User said "add tech stack info and can we upgrade later?"
  → Added Section 6b (Tech Stack table) and Section 6c (Future Upgrade Path)

Iteration 7: User said "do the same for implementation_plan.md"
  → Created a full-stack implementation_plan.md showing the upgrade path
     with complete directory tree, API specs, frontend components, docker-compose
```

---

## Files Created / Modified

| File | Action | Description |
|------|--------|-------------|
| `plan.md` | Created (overwritten multiple times) | Single authoritative reference for AI core work |
| `implementation_plan.md` | Created (overwritten multiple times) | Full-stack upgrade path reference |

---

## Final State of plan.md

Contains 10 sections:
1. What the project is (Prediction-to-Prescription)
2. Research question & 3 testable claims
3. Architecture flow (7 phases)
4. Complete file structure (data/, notebooks/, src/, models/, app.py)
5. Dataset facts & sources
6. Conda environment spec
6b. Tech stack table (what tech is used where and in which phase)
6c. Future upgrade path (Streamlit → FastAPI + React)
7. Phase-by-phase implementation (Phases 1–7, each with task tables)
8. Academic references
9. Timeline summary (Day 1–16)
10. The publishable claim

---

## Final State of implementation_plan.md

Contains 8 sections:
1. How it relates to plan.md (comparison table)
2. Full-stack tech stack (PyTorch, FastAPI, React, Docker, etc.)
3. Complete full-stack file structure (src/ + backend/ + frontend/)
4. API endpoint specs with example JSON responses
5. Frontend dashboard panel breakdown
6. Three upgrade phases (A: Backend, B: Frontend, C: Docker)
7. docker-compose.yml preview
8. Summary timeline (NOW vs LATER)

### 5. Test-Driven Development (Build → Verify → Proceed)
Every phase now has a **✅ Verification Checkpoint** — a block of Python asserts that must pass before moving to the next phase. No phase begins until the previous one is verified. This catches errors early (bad SMILES, NaN loss, shape mismatches) instead of debugging them deep into training.

---

## What's Next

**Phase 1 from plan.md:** Set up Conda environment, load Tox21 data, validate SMILES, download OCHEM alerts.
Then verify with the Phase 1 checkpoint before moving to Phase 2.
