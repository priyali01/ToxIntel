# ToxIntel™: Project Pitch & Presentation Guide

This guide is designed to help you confidently present ToxIntel to your professor or review panel. It's structured to highlight the *scientific depth*, *novelty*, and *practical application* of your work.

---

## 1. The Hook (The Problem)

"Ma'am, standard toxicity prediction models have a major flaw. They treat toxicity as a simple 'Yes' or 'No' classification problem. But in the real world of drug discovery, knowing a molecule is toxic isn't enough. A chemist needs to know *why* it's toxic, and more importantly, *how to fix it*. Furthermore, these standard models often fail silently because they ignore the underlying 'geometric imbalance' of chemical space—meaning toxic molecules often clump together, causing models to memorize scaffolds instead of learning true toxic mechanisms."

## 2. The Solution (ToxIntel)

"To solve this, I built **ToxIntel™**: a complete **Prediction-to-Prescription** decision-support system. It doesn’t just predict toxicity; it prescribes synthesizable, safer alternatives. 

My pipeline does four things:
1. **Multi-Task Prediction:** It predicts toxicity across 12 distinct biological endpoints simultaneously using a custom PyTorch Neural Network (ToxNet).
2. **Explainability:** It uses SHAP attribution to highlight the exact molecular fragment causing the toxicity, cross-validated against known structural alerts.
3. **Prescription:** It queries a bioisostere database to find structural replacements, filtering them so they are actually synthesizable and maintain their ADME properties (like Molecular Weight and LogP).
4. **Pareto Optimization & Reliability:** Finally, it re-evaluates the new candidates and ranks them using Pareto dominance (ensuring we don't fix one toxicity just to cause another), while providing strict 90% confidence guarantees using Mondrian Conformal Prediction."

## 3. The Novel Contribution (The "Wow" Factor)

"The core research novelty of my project is the analysis of **Geometric Imbalance**. I wrote a custom algorithm to measure 'Intraclass Tanimoto Cohesion'. I proved that for certain endpoints (like the Androgen Receptor), toxic molecules are tightly clustered. I demonstrated that for these highly cohesive endpoints, standard techniques like SMOTE actually *hurt* performance because they just reinforce the cluster. Instead, I successfully handled this severe imbalance using a masked Focal Loss function tailored for multi-task learning. This specific analysis has not been published before on the Tox21 benchmark."

## 4. The Engineering & UI

"To make this accessible, I didn't just stop at Jupyter Notebooks. I built a research-grade **Streamlit Dashboard** that visualizes the entire 4-step pipeline. You can input any SMILES string, view its 12-axis toxicity radar profile against calibrated safety thresholds, see reliability badges warning you if the molecule is 'Out-Of-Distribution', and interactively compare the original molecule side-by-side with the system's prescribed safer bioisosteres."

---

## Anticipated Questions & How to Answer Them

**Q: Why didn't you use standard Accuracy as your metric?**
*A:* "The Tox21 dataset is severely imbalanced—some endpoints are only 2.9% toxic. If a model just guesses 'Safe' every time, it gets 97% accuracy but is completely useless. I used Macro AUPRC (Area Under the Precision-Recall Curve) because it explicitly measures how well the model detects the minority toxic class."

**Q: Why did you choose ECFP4 descriptors over Graph Neural Networks (GNNs)?**
*A:* "I performed a rigorous ablation study comparing 5 different representations. I found that combining ECFP4 fingerprints with 10 continuous RDKit descriptors (`ecfp4_desc`) yielded the highest stability and AUPRC. Furthermore, ECFP4 retains a `bitInfo` mapping, which is absolutely mandatory for mapping SHAP attribution scores back to specific atoms so the bioisostere engine knows what part of the molecule to replace. Standard GNNs make this atom-level extraction much harder."

**Q: What is Mondrian Conformal Prediction?**
*A:* "Standard probabilities from neural networks are often overconfident. Mondrian Conformal Prediction mathematically guarantees that the true label will fall within the predicted set 90% of the time, even if the molecule belongs to a novel scaffold (Out-Of-Distribution). It categorizes molecules into 'Known', 'Similar', or 'Novel' spaces to give the user a clear reliability signal."

**Q: How did you split your data?**
*A:* "I specifically used a **Murcko Scaffold-Stratified Split**. Random splitting causes structural leakage—where variations of the same scaffold end up in both training and testing, artificially inflating the score. My split ensures the model is tested on unseen chemical scaffolds, providing an honest evaluation."
