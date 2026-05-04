# ToxIntel: Research Presentation Outline
**Use this document to build your presentation slides. It contains detailed speaker notes and recommendations for which images to include on each slide.**

---

## Slide 1: Title Slide
*   **Title:** ToxIntel: Addressing Geometric Imbalance in Multi-Task Molecular Toxicity Prediction via Shared-Backbone Neural Architectures
*   **Subtitle:** A Prediction-to-Prescription Decision Support System
*   **Presenter:** Priyali Sharma
*   **Visual:** Clean, professional title layout with university/institution logo.

---

## Slide 2: The Core Problem in Drug Discovery
*   **Heading:** The Cost of Late-Stage Toxicity Failures
*   **Key Bullet Points:**
    *   Developing a novel therapeutic takes $10+$ years and costs $>\$1$ Billion.
    *   $\sim$90\% of clinical candidates fail; **drug-induced toxicity** is a leading cause.
    *   Toxicity often surfaces too late (in Phase III or post-market).
    *   **The Goal:** "Shift Left" – identify and fix toxicity computationally before any synthesis occurs.
*   **Speaker Notes:** Emphasize that current methods just predict "toxic" or "safe" but don't tell chemists *how* to fix the molecule.

---

## Slide 3: Challenges in Toxicity Modeling
*   **Heading:** Class Imbalance \& Geometric Imbalance
*   **Key Bullet Points:**
    *   **Class Imbalance:** Toxic compounds are rare (average 8\% prevalence in Tox21).
    *   **Geometric Imbalance (Our Novel Focus):** Toxic molecules cluster tightly in specific chemical spaces (scaffolds).
    *   Standard oversampling (like SMOTE) fails here—it just generates synthetic points inside the same dense toxic cluster without improving generalizability.
*   **Visual Suggestion:** Include `01_class_balance.png` to show the extreme imbalance across endpoints.

---

## Slide 4: Introducing ToxIntel
*   **Heading:** A Prediction-to-Prescription Pipeline
*   **Key Bullet Points:**
    *   Moves beyond binary classification to actionable structural optimization.
    *   Powered by **ToxNet**, a multi-task neural network.
    *   Evaluates 12 distinct nuclear receptor and stress response endpoints simultaneously.
*   **Speaker Notes:** This is the core thesis of your paper. You are shifting the paradigm from just finding problems to offering solutions.

---

## Slide 5: Methodology I - Data \& Features
*   **Heading:** Rigorous Evaluation on the Tox21 Dataset
*   **Key Bullet Points:**
    *   Dataset: Tox21 (7,831 compounds, 12 endpoints).
    *   **Scaffold-Stratified Split:** Bemis-Murcko splitting ensures test molecules share *no ring structures* with training data (tests true generalization).
    *   **Optimal Featurization:** `ecfp4_desc` (Morgan Fingerprints + 10 physicochemical descriptors).
*   **Visual Suggestion:** Include `03_featurization_ablation.png` to prove why you chose this feature set.

---

## Slide 6: Methodology II - ToxNet Architecture
*   **Heading:** Shared-Backbone Multi-Task Learning
*   **Key Bullet Points:**
    *   4 hidden layers (1024 $\rightarrow$ 512 $\rightarrow$ 512 $\rightarrow$ 256) with Residual Connections and GELU.
    *   Branches into 12 independent task heads.
    *   **Per-Endpoint Masked Focal Loss:** Masks missing labels (-1) and down-weights easily classified majority samples ($\gamma=2.0$) to focus on rare toxic cases.
*   **Visual Suggestion:** Include `04_final_training.png` to show the stable training and validation loss convergence.

---

## Slide 7: Quantifying Geometric Imbalance
*   **Heading:** Intraclass Tanimoto Cohesion
*   **Key Bullet Points:**
    *   We mathematically define clustering using the cohesion ratio $\rho$.
    *   $\rho > 1.5$ indicates severe spatial clustering of toxic compounds.
    *   For highly cohesive endpoints, we proved that synthetic oversampling (SMOTE) is ineffective.
*   **Visual Suggestion:** Include `02_geometric_imbalance.png` (The bar chart showing the cohesion ratios per endpoint).

---

## Slide 8: The Prescriptive Pipeline (Steps 1 & 2)
*   **Heading:** From Prediction to Attribution
*   **Key Bullet Points:**
    *   **Step 1 (Attribution):** Use SHAP DeepExplainer to identify specific atoms causing the toxicity alert.
    *   **Step 2 (Validation):** Cross-reference these SHAP-flagged fragments against known databases (PAINS, OCHEM, Brenk).
*   **Speaker Notes:** We don't just blindly trust the AI; we validate its attributions against established chemical knowledge.

---

## Slide 9: The Prescriptive Pipeline (Steps 3 & 4)
*   **Heading:** Bioisostere Replacement \& Pareto Optimization
*   **Key Bullet Points:**
    *   **Step 3 (Prescription):** Query ChEMBL for known bioisosteres (replacements) for the toxic fragment, filtering for synthesizability (SAScore < 4.0).
    *   **Step 4 (Pareto Optimization):** ToxNet re-evaluates the new candidate across all 12 endpoints.
    *   Ensures a substitution doesn't fix liver toxicity but cause heart toxicity (multi-organ safety).
*   **Speaker Notes:** Mention that Conformal Prediction guarantees 90\% reliability on these final suggestions.

---

## Slide 10: Results - Model Performance
*   **Heading:** State-of-the-Art Minority Class Detection
*   **Key Bullet Points:**
    *   AUPRC (Precision-Recall) is the primary metric due to severe imbalance.
    *   Logistic Regression: 0.240 AUPRC
    *   XGBoost: 0.310 AUPRC
    *   **ToxNet (Ours): 0.364 AUPRC (Macro AUROC: 0.842)**
    *   4.5x improvement over random guessing on entirely novel molecular scaffolds.
*   **Visual Suggestion:** Recreate the results table from the paper on this slide.

---

## Slide 11: Results - Validating Geometric Imbalance
*   **Heading:** Geometric Cohesion Predicts SMOTE Efficacy
*   **Key Bullet Points:**
    *   Empirical evidence supports our hypothesis.
    *   Endpoints with high geometric imbalance ($\rho > 1.5$) saw negligible/negative benefit from SMOTE.
    *   Endpoints with diffuse toxic distributions ($\rho < 1.5$) benefited significantly.
*   **Visual Suggestion:** Include `05_geometric_correlation.png` showing the correlation plot.

---

## Slide 12: Conclusion \& Future Work
*   **Heading:** Transforming Computational Toxicology
*   **Key Bullet Points:**
    *   ToxIntel successfully shifts from diagnostic classification to prescriptive structural optimization.
    *   Geometric imbalance is a critical, previously under-explored factor in cheminformatics.
    *   **Future Work:** Integrating 3D equivariant Graph Neural Networks (GNNs) and predicting *in vivo* metabolite toxicity via CYP450 pathways.
*   **Speaker Notes:** End on a strong note about how this approach makes AI directly actionable for medicinal chemists.
