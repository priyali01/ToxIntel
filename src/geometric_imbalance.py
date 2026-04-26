"""
src/geometric_imbalance.py — Intraclass Tanimoto Cohesion Analysis

THIS IS THE NOVEL RESEARCH CONTRIBUTION.

No existing Tox21 paper has quantified geometric imbalance per endpoint.
Standard imbalance metrics only count how many toxic vs non-toxic samples exist.
Geometric imbalance measures HOW the toxic samples are distributed in chemical space.

Key insight:
    If toxic molecules for an endpoint all share the same scaffold (high cohesion),
    SMOTE just generates more of the same — it reinforces the cluster instead of
    adding diversity. SHAP attribution may also point to a scaffold-specific feature
    rather than a mechanistically relevant toxic substructure.

Output:
    Per-endpoint cohesion table with:
    - toxic_cohesion: mean pairwise Tanimoto within the toxic class
    - nontoxic_cohesion: mean pairwise Tanimoto within the non-toxic class
    - cohesion_ratio: toxic / nontoxic
    - smote_valid: whether SMOTE is likely to help (ratio < 1.5)
"""

import numpy as np
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem


def compute_pairwise_tanimoto(fps: list, max_pairs: int = 5000) -> float:
    """
    Compute mean pairwise Tanimoto similarity within a set of fingerprints.

    For large sets, we sample pairs randomly instead of computing all O(n^2) pairs.

    Args:
        fps: List of RDKit fingerprint objects
        max_pairs: Maximum number of pairs to sample (for speed)

    Returns:
        Mean Tanimoto similarity (float between 0 and 1)
    """
    n = len(fps)
    if n < 2:
        return 0.0

    # If small enough, compute all pairs
    if n * (n - 1) // 2 <= max_pairs:
        similarities = []
        for i in range(n):
            for j in range(i + 1, n):
                sim = DataStructs.TanimotoSimilarity(fps[i], fps[j])
                similarities.append(sim)
        return float(np.mean(similarities))

    # Otherwise, sample random pairs
    rng = np.random.RandomState(42)
    similarities = []
    for _ in range(max_pairs):
        i, j = rng.choice(n, size=2, replace=False)
        sim = DataStructs.TanimotoSimilarity(fps[i], fps[j])
        similarities.append(sim)
    return float(np.mean(similarities))


def smiles_to_fp_object(smiles: str, radius: int = 2, n_bits: int = 2048):
    """
    Convert SMILES to an RDKit fingerprint OBJECT (not numpy array).
    Needed for Tanimoto computation via DataStructs.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)


def compute_intraclass_cohesion(smiles_list: list, labels: np.ndarray,
                                 endpoint_name: str) -> dict:
    """
    Measure the geometric cohesion of the toxic vs non-toxic class for one endpoint.

    This is the core function for Claim 1 of the research question:
        "Does geometric cohesion predict SMOTE failure?"

    Args:
        smiles_list: List of SMILES strings
        labels: Array of labels (0, 1, or -1 for missing)
        endpoint_name: Name of the endpoint (for printing)

    Returns:
        dict with keys:
            - endpoint: name
            - toxic_cohesion: mean pairwise Tanimoto within toxic class
            - nontoxic_cohesion: mean pairwise Tanimoto within non-toxic class
            - cohesion_ratio: toxic / nontoxic
            - smote_valid: True if ratio < 1.5 (SMOTE may help)
            - n_toxic: number of toxic samples
            - n_nontoxic: number of non-toxic samples
    """
    labels = np.array(labels)

    # Separate toxic and non-toxic (ignore -1 = missing)
    toxic_idx = np.where(labels == 1)[0]
    nontoxic_idx = np.where(labels == 0)[0]

    # Generate fingerprint objects
    toxic_fps = []
    for i in toxic_idx:
        fp = smiles_to_fp_object(smiles_list[i])
        if fp is not None:
            toxic_fps.append(fp)

    nontoxic_fps = []
    for i in nontoxic_idx:
        fp = smiles_to_fp_object(smiles_list[i])
        if fp is not None:
            nontoxic_fps.append(fp)

    # Compute cohesion
    toxic_cohesion = compute_pairwise_tanimoto(toxic_fps)
    nontoxic_cohesion = compute_pairwise_tanimoto(nontoxic_fps)

    # Cohesion ratio: how much more clustered toxic is vs non-toxic
    # If ratio > 1.5, toxic class is significantly more clustered
    if nontoxic_cohesion > 0:
        cohesion_ratio = toxic_cohesion / nontoxic_cohesion
    else:
        cohesion_ratio = float('inf')

    # SMOTE validity: if toxic class is too clustered, SMOTE just
    # reinforces the existing cluster instead of adding diversity
    smote_valid = cohesion_ratio < 1.5

    result = {
        'endpoint': endpoint_name,
        'toxic_cohesion': round(toxic_cohesion, 4),
        'nontoxic_cohesion': round(nontoxic_cohesion, 4),
        'cohesion_ratio': round(cohesion_ratio, 4),
        'smote_valid': smote_valid,
        'n_toxic': len(toxic_fps),
        'n_nontoxic': len(nontoxic_fps),
    }

    return result


def analyze_all_endpoints(df, target_cols: list) -> list:
    """
    Run geometric imbalance analysis across all 12 Tox21 endpoints.

    This produces the per-endpoint cohesion table — a publishable result.

    Args:
        df: DataFrame with 'smiles' column and target columns
        target_cols: List of 12 endpoint column names

    Returns:
        List of result dicts (one per endpoint)
    """
    smiles_list = df['smiles'].tolist()
    results = []

    print("Computing geometric imbalance per endpoint...")
    print(f"{'Endpoint':20s} {'Toxic Coh':>10s} {'NonTox Coh':>10s} {'Ratio':>8s} {'SMOTE?':>8s} {'N_tox':>6s}")
    print("-" * 70)

    for col in target_cols:
        labels = df[col].values
        result = compute_intraclass_cohesion(smiles_list, labels, col)
        results.append(result)

        flag = "YES" if result['smote_valid'] else "NO"
        print(f"{col:20s} {result['toxic_cohesion']:10.4f} {result['nontoxic_cohesion']:10.4f} "
              f"{result['cohesion_ratio']:8.4f} {flag:>8s} {result['n_toxic']:6d}")

    # Summary
    valid_count = sum(1 for r in results if r['smote_valid'])
    print(f"\nSMOTE valid for {valid_count}/{len(results)} endpoints")

    return results


# ── Run when called directly (for quick testing) ──────────────────
if __name__ == '__main__':
    import sys
    sys.path.insert(0, '.')
    from src.featurize import load_and_clean_tox21, TARGET_COLS

    df = load_and_clean_tox21()

    print("\n--- Phase 3: Geometric Imbalance Analysis ---\n")
    results = analyze_all_endpoints(df, TARGET_COLS)
