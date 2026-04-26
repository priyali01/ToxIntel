import json
import os
import numpy as np
from rdkit import Chem

def load_structural_alerts(filepath='data/ochem_alerts.json'):
    """
    Loads structural alerts (Brenk, PAINS, OCHEM) from a JSON file.
    If the file doesn't exist, we fallback to a small default set of known toxicophores.
    """
    if os.path.exists(filepath):
        try:
            with open(filepath, 'r') as f:
                return json.load(f)
        except Exception as e:
            print(f"Warning: Failed to load alerts from {filepath}: {e}")
    
    # Fallback to a core set of known toxicophores (Anilines, Quinones, Epoxides, etc.)
    return {
        "aniline": "c1ccccc1N",
        "quinone": "O=C1C=CC(=O)C=C1",
        "epoxide": "C1OC1",
        "michael_acceptor": "C=CC(=O)",
        "nitro_aromatic": "c1ccccc1[N+](=O)[O-]",
        "polycyclic_aromatic": "c1ccc2c(c1)ccc3ccccc23"
    }

def get_top_shap_bits(shap_values, top_k=3):
    """
    Given a vector of SHAP values for a single prediction, returns the indices
    of the bits with the highest positive contribution to toxicity.
    """
    # Sort indices by descending SHAP value
    sorted_indices = np.argsort(shap_values)[::-1]
    
    # Only keep bits that actually contribute to toxicity (SHAP > 0)
    top_indices = [idx for idx in sorted_indices[:top_k] if shap_values[idx] > 0]
    return top_indices

def validate_shap_attribution(mol, bit_info, top_bits, alerts_dict=None):
    """
    Step 1b: Validates if the SHAP-identified bits map to known toxicophores.
    
    Args:
        mol: RDKit molecule.
        bit_info: Dictionary from RDKit Morgan fingerprint generator.
        top_bits: List of bit indices flagged by SHAP.
        alerts_dict: Dictionary of SMARTS patterns for known alerts.
        
    Returns:
        Validation dictionary with confidence rating.
    """
    if alerts_dict is None:
        alerts_dict = load_structural_alerts()
        
    # Find all atoms involved in the top SHAP bits
    flagged_atoms = set()
    for bit in top_bits:
        if bit in bit_info:
            for atom_idx, radius in bit_info[bit]:
                # In RDKit, bitInfo maps bit -> ((atom_idx, radius), ...)
                # Add the root atom. For a full radius mapping, we'd need FindAtomEnvironmentOfRadiusN
                flagged_atoms.add(atom_idx)
                
    # Check if the molecule contains any known structural alerts
    alert_matches = []
    matched_atoms = set()
    
    for name, smarts in alerts_dict.items():
        pattern = Chem.MolFromSmarts(smarts)
        if pattern:
            matches = mol.GetSubstructMatches(pattern)
            for match in matches:
                alert_matches.append(name)
                matched_atoms.update(match)
                
    # Calculate overlap
    overlap = flagged_atoms.intersection(matched_atoms)
    
    confidence = 'LOW'
    recommendation = 'DO NOT SUBSTITUTE \u2014 attribution unvalidated'
    
    if len(flagged_atoms) == 0:
        confidence = 'LOW'
    elif len(overlap) > 0:
        confidence = 'HIGH'
        recommendation = 'PROCEED'
    elif len(alert_matches) > 0:
        # Alerts exist, but SHAP pointed somewhere else
        confidence = 'MEDIUM'
        recommendation = 'PROCEED WITH CAUTION'
        
    return {
        'shap_confidence': confidence,
        'recommendation': recommendation,
        'flagged_atoms': list(flagged_atoms),
        'matched_alerts': list(set(alert_matches)),
        'overlap_atoms': list(overlap)
    }
