"""
src/scaffold_split.py — Murcko Scaffold-Stratified Splitting

Why NOT random split?
    If molecule A (aspirin) is in train and molecule B (slightly modified aspirin)
    is in test, the model memorizes the scaffold and gets a falsely high AUC.
    This is called "structural leakage."

Why scaffold split?
    We extract the core ring system (Murcko scaffold) of every molecule.
    All molecules sharing the same scaffold go into the SAME split.
    The model is then tested on structurally DIFFERENT molecules — a harder
    but honest evaluation.

Reference: Wu et al. (2018) MoleculeNet recommends scaffold splitting
           for all molecular property prediction benchmarks.
"""

from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold
from collections import defaultdict
import numpy as np
import pandas as pd


def get_scaffold(smiles: str) -> str:
    """
    Extract the Murcko scaffold (core ring system) from a SMILES string.

    Example:
        "CC(=O)Oc1ccccc1C(=O)O" (Aspirin) → "c1ccccc1" (benzene ring)

    All side chains are stripped, leaving only the ring backbone.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return ''
    scaffold = MurckoScaffold.MurckoScaffoldSmiles(
        mol=mol, includeChirality=False
    )
    return scaffold


def scaffold_split(smiles_list: list, train_frac: float = 0.7,
                   val_frac: float = 0.1, calib_frac: float = 0.1, test_frac: float = 0.1,
                   seed: int = 42) -> tuple:
    """
    Split molecule indices by Murcko scaffold.

    How it works:
        1. Group all molecules by their scaffold
        2. Sort scaffold groups by size (largest first)
        3. Assign entire groups to train/val/test until the target
           fraction is reached

    This guarantees NO structural leakage between splits —
    every molecule in test has a scaffold that NEVER appeared in train.

    Args:
        smiles_list: List of SMILES strings
        train_frac: Fraction for training (default 0.7)
        val_frac: Fraction for validation (default 0.1)
        calib_frac: Fraction for Mondrian calibration (default 0.1)
        test_frac: Fraction for testing (default 0.1)
        seed: Random seed for reproducibility

    Returns:
        (train_indices, val_indices, calib_indices, test_indices) — lists of integer indices
    """
    assert abs(train_frac + val_frac + calib_frac + test_frac - 1.0) < 1e-6, \
        "Fractions must sum to 1.0"

    # Step 1: Group molecule indices by scaffold
    scaffold_to_indices = defaultdict(list)
    for i, smi in enumerate(smiles_list):
        scaffold = get_scaffold(smi)
        scaffold_to_indices[scaffold].append(i)

    # Step 2: Sort scaffold groups by size (largest first for stability)
    rng = np.random.RandomState(seed)
    scaffold_groups = list(scaffold_to_indices.values())
    rng.shuffle(scaffold_groups)
    # Sort by size descending — puts large scaffold groups in train first
    scaffold_groups.sort(key=len, reverse=True)

    # Step 3: Assign groups to splits
    n_total = len(smiles_list)
    n_train = int(n_total * train_frac)
    n_val = int(n_total * val_frac)
    n_calib = int(n_total * calib_frac)

    train_indices = []
    val_indices = []
    calib_indices = []
    test_indices = []

    for group in scaffold_groups:
        if len(train_indices) < n_train:
            train_indices.extend(group)
        elif len(val_indices) < n_val:
            val_indices.extend(group)
        elif len(calib_indices) < n_calib:
            calib_indices.extend(group)
        else:
            test_indices.extend(group)

    print(f"Scaffold split results:")
    print(f"  Unique scaffolds: {len(scaffold_to_indices)}")
    print(f"  Train: {len(train_indices):,} ({len(train_indices)/n_total*100:.1f}%)")
    print(f"  Val:   {len(val_indices):,} ({len(val_indices)/n_total*100:.1f}%)")
    print(f"  Calib: {len(calib_indices):,} ({len(calib_indices)/n_total*100:.1f}%)")
    print(f"  Test:  {len(test_indices):,} ({len(test_indices)/n_total*100:.1f}%)")

    # Verify no overlap
    train_set = set(train_indices)
    val_set = set(val_indices)
    calib_set = set(calib_indices)
    test_set = set(test_indices)
    assert len(train_set & val_set) == 0, "Train/Val overlap detected!"
    assert len(train_set & calib_set) == 0, "Train/Calib overlap detected!"
    assert len(train_set & test_set) == 0, "Train/Test overlap detected!"
    assert len(val_set & calib_set) == 0, "Val/Calib overlap detected!"
    assert len(val_set & test_set) == 0, "Val/Test overlap detected!"
    assert len(calib_set & test_set) == 0, "Calib/Test overlap detected!"
    print(f"  Overlap check: PASSED (no structural leakage)")

    return train_indices, val_indices, calib_indices, test_indices


# ── Run when called directly (for quick testing) ──────────────────
if __name__ == '__main__':
    from featurize import load_and_clean_tox21

    df = load_and_clean_tox21()

    print("\n--- Phase 2: Scaffold Split Test ---")
    train_idx, val_idx, calib_idx, test_idx = scaffold_split(
        df['smiles'].tolist(), train_frac=0.7, val_frac=0.1, calib_frac=0.1, test_frac=0.1
    )

    # Show scaffold examples
    print(f"\nExample scaffolds in TRAIN:")
    for i in train_idx[:3]:
        smi = df['smiles'].iloc[i]
        scaf = get_scaffold(smi)
        print(f"  '{smi[:40]}...' -> scaffold: '{scaf}'")

    print(f"\nExample scaffolds in TEST:")
    for i in test_idx[:3]:
        smi = df['smiles'].iloc[i]
        scaf = get_scaffold(smi)
        print(f"  '{smi[:40]}...' -> scaffold: '{scaf}'")
