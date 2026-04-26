"""
src/featurize.py — SMILES Validation & Molecular Featurization

Phase 1: validate_smiles() — drops invalid SMILES before anything else
Phase 2: Morgan fingerprints, MACCS keys, RDKit descriptors (added later)
"""

from rdkit import Chem
import pandas as pd
import numpy as np


# ── The 12 Tox21 endpoint columns ──────────────────────────────────
TARGET_COLS = [
    'NR-AR', 'NR-AR-LBD', 'NR-AhR', 'NR-Aromatase',
    'NR-ER', 'NR-ER-LBD', 'NR-PPAR-gamma',
    'SR-ARE', 'SR-ATAD5', 'SR-HSE', 'SR-MMP', 'SR-p53'
]


def validate_smiles(smiles_list: list) -> tuple:
    """
    Check every SMILES string using RDKit. Drop any that RDKit cannot parse.

    Why this matters:
        Chem.MolFromSmiles() returns None silently for invalid inputs.
        If we don't catch this here, it propagates as IndexError or NaN
        loss hours into training.

    Args:
        smiles_list: List of SMILES strings from the dataset

    Returns:
        valid_indices: List of row indices with valid SMILES
        invalid_records: List of (index, smiles) tuples that failed
    """
    valid_indices = []
    invalid_records = []

    for i, smi in enumerate(smiles_list):
        mol = Chem.MolFromSmiles(smi)
        if mol is not None:
            valid_indices.append(i)
        else:
            invalid_records.append((i, smi))

    if invalid_records:
        print(f"WARNING: {len(invalid_records)} invalid SMILES dropped:")
        for idx, smi in invalid_records[:5]:  # Show first 5
            print(f"  Row {idx}: '{smi[:50]}'")

    print(f"VALID: {len(valid_indices)}/{len(smiles_list)} SMILES passed validation")
    return valid_indices, invalid_records


def load_and_clean_tox21(csv_path: str = 'data/tox21.csv') -> pd.DataFrame:
    """
    Load the Tox21 dataset, validate SMILES, and replace NaN labels with -1.

    The -1 sentinel is used throughout the pipeline:
        - Focal Loss masks out -1 labels (doesn't penalize unknowns)
        - Evaluation metrics skip -1 entries
        - This is standard practice for multi-label datasets with missing labels

    Args:
        csv_path: Path to the raw tox21.csv file

    Returns:
        df: Cleaned DataFrame with only valid SMILES and -1 for missing labels
    """
    # Load raw data
    df = pd.read_csv(csv_path)
    print(f"Raw dataset: {df.shape[0]} compounds, {df.shape[1]} columns")

    # Validate SMILES — drop any that RDKit can't parse
    valid_idx, invalid = validate_smiles(df['smiles'].tolist())
    df = df.iloc[valid_idx].reset_index(drop=True)

    # Replace NaN labels with -1 sentinel
    # NaN means "this compound was not tested for this endpoint"
    # We use -1 so we can mask these during training (Focal Loss ignores -1)
    for col in TARGET_COLS:
        df[col] = df[col].fillna(-1).astype(int)

    # Print summary
    print(f"\nCleaned dataset: {df.shape[0]} compounds")
    print(f"\nPer-endpoint toxic prevalence:")
    for col in TARGET_COLS:
        known = df[col] != -1
        toxic = (df[col] == 1).sum()
        total = known.sum()
        pct = (toxic / total * 100) if total > 0 else 0
        nan_pct = (~known).sum() / len(df) * 100
        print(f"  {col:20s}  toxic={toxic:4d}/{total:4d}  ({pct:5.1f}%)  NaN={nan_pct:.0f}%")

    return df


# ═══════════════════════════════════════════════════════════════════
# Phase 2: Molecular Fingerprinting
# ═══════════════════════════════════════════════════════════════════

from rdkit.Chem import AllChem, MACCSkeys, Descriptors


def smiles_to_morgan(smiles: str, radius: int = 2, n_bits: int = 2048) -> np.ndarray:
    """
    Convert a SMILES string to a Morgan (ECFP) fingerprint.

    Morgan fingerprints encode circular substructures around each atom.
    - radius=2 → ECFP4 (captures neighborhoods up to 4 bonds away)
    - radius=3 → ECFP6 (captures wider neighborhoods)

    Args:
        smiles: Valid SMILES string (must pass validate_smiles first)
        radius: Circular radius (2=ECFP4, 3=ECFP6)
        n_bits: Length of the bit vector (1024 or 2048)

    Returns:
        Numpy array of shape (n_bits,) with binary values 0/1
    """
    mol = Chem.MolFromSmiles(smiles)
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)
    return np.array(fp, dtype=np.int8)


def smiles_to_morgan_with_info(smiles: str, radius: int = 2, n_bits: int = 2048) -> tuple:
    """
    Same as smiles_to_morgan, but also returns bitInfo for SHAP atom mapping.

    CRITICAL: bitInfo maps each "on" bit → which atoms activated it.
    This is required in Phase 6 to go from "SHAP says bit 42 is important"
    to "bit 42 corresponds to the -NH2 group at atom index 7".

    WARNING: bitInfo must be declared OUTSIDE the function call, or
    Python's garbage collector may delete it before SHAP reads it.

    Returns:
        (fingerprint_array, bit_info_dict)
    """
    mol = Chem.MolFromSmiles(smiles)
    bit_info = {}  # Declared OUTSIDE GetMorganFingerprintAsBitVect
    fp = AllChem.GetMorganFingerprintAsBitVect(
        mol, radius, nBits=n_bits, bitInfo=bit_info
    )
    return np.array(fp, dtype=np.int8), bit_info


def smiles_to_maccs(smiles: str) -> np.ndarray:
    """
    Convert SMILES to MACCS keys — 166-bit structural key fingerprint.

    Unlike Morgan (which finds arbitrary substructures), MACCS uses
    a fixed set of 166 predefined structural patterns.
    Simpler but less expressive than Morgan.

    Returns:
        Numpy array of shape (167,) — bit 0 is always 0, bits 1-166 are the keys
    """
    mol = Chem.MolFromSmiles(smiles)
    fp = MACCSkeys.GenMACCSKeys(mol)
    return np.array(fp, dtype=np.int8)


def smiles_to_rdkit_descriptors(smiles: str) -> np.ndarray:
    """
    Compute RDKit molecular descriptors (continuous values like MolWt, LogP, etc.)

    These are NOT fingerprints — they are physicochemical properties.
    Used in ECFP+Desc representation where we concatenate fingerprints
    with descriptors.

    WARNING: These values are on different scales (MolWt ~ 300, LogP ~ 2).
    Must apply StandardScaler on TRAIN set only before training.

    Returns:
        Numpy array of descriptor values
    """
    mol = Chem.MolFromSmiles(smiles)
    desc_names = [
        'MolWt', 'MolLogP', 'NumHDonors', 'NumHAcceptors',
        'TPSA', 'NumRotatableBonds', 'RingCount', 'NumAromaticRings',
        'FractionCSP3', 'HeavyAtomCount'
    ]
    values = []
    for name in desc_names:
        func = getattr(Descriptors, name, None)
        if func:
            values.append(func(mol))
        else:
            # Fallback for descriptors accessed differently
            values.append(0.0)
    return np.array(values, dtype=np.float32)


def featurize_dataset(smiles_list: list, method: str = 'ecfp4_2048') -> np.ndarray:
    """
    Convert a list of SMILES strings into a feature matrix.

    This is the main function that Phase 3/4 will call.

    Args:
        smiles_list: List of valid SMILES strings
        method: One of:
            - 'ecfp4_1024'  → Morgan radius=2, 1024 bits
            - 'ecfp4_2048'  → Morgan radius=2, 2048 bits  ← DEFAULT (best for SHAP)
            - 'ecfp6_2048'  → Morgan radius=3, 2048 bits
            - 'maccs'       → MACCS 166-bit keys
            - 'rdkit_desc'  → 10 physicochemical descriptors

    Returns:
        X: Feature matrix of shape (n_molecules, n_features)
    """
    method = method.lower()

    if method == 'ecfp4_1024':
        return np.array([smiles_to_morgan(s, radius=2, n_bits=1024) for s in smiles_list])
    elif method == 'ecfp4_2048':
        return np.array([smiles_to_morgan(s, radius=2, n_bits=2048) for s in smiles_list])
    elif method == 'ecfp6_2048':
        return np.array([smiles_to_morgan(s, radius=3, n_bits=2048) for s in smiles_list])
    elif method == 'maccs':
        return np.array([smiles_to_maccs(s) for s in smiles_list])
    elif method == 'rdkit_desc':
        return np.array([smiles_to_rdkit_descriptors(s) for s in smiles_list])
    else:
        raise ValueError(f"Unknown method: {method}. Use ecfp4_1024/ecfp4_2048/ecfp6_2048/maccs/rdkit_desc")


# ── Run when called directly (for quick testing) ──────────────────
if __name__ == '__main__':
    df = load_and_clean_tox21()
    print(f"\nFinal shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")

    # Phase 2 test: Featurize first 5 molecules
    print("\n--- Phase 2: Featurization Test ---")
    test_smiles = df['smiles'].iloc[:5].tolist()

    for method in ['ecfp4_2048', 'ecfp6_2048', 'maccs', 'rdkit_desc']:
        X = featurize_dataset(test_smiles, method=method)
        print(f"  {method:15s}  shape={str(X.shape):15s}  dtype={X.dtype}")

    # Test bitInfo capture
    fp, info = smiles_to_morgan_with_info(test_smiles[0])
    print(f"\n  bitInfo test: {len(info)} active bits captured for '{test_smiles[0][:30]}...'")
