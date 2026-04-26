import os
import sys
import numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors
from rdkit.Chem import RDConfig

# Import SA_Score from RDKit Contrib
try:
    sys.path.append(os.path.join(RDConfig.RDContribDir, 'SA_Score'))
    import sascorer
    SA_SCORE_AVAILABLE = True
except ImportError:
    SA_SCORE_AVAILABLE = False
    print("Warning: RDKit SA_Score module not found. Synthesizability filtering will be approximated.")

def get_sascore(mol):
    if SA_SCORE_AVAILABLE:
        return sascorer.calculateScore(mol)
    else:
        # Crude approximation based on rings and stereocenters if contrib module is missing
        rings = Chem.GetSSSR(mol)
        stereo = len(Chem.FindMolChiralCenters(mol, includeUnassigned=True))
        return min(10.0, 1.0 + (rings * 0.5) + (stereo * 0.5))

def load_mock_chembl_cache():
    """
    Returns a mocked, localized cache of common bioisosteric replacements.
    In production, this queries an offline ChEMBL fragment database.
    Format: "toxic_smarts": ["safe_smarts_1", "safe_smarts_2"]
    """
    return {
        # Aniline (often toxic/metabolically unstable) -> Pyridine or Aliphatic amine
        "c1ccccc1N": ["c1ccncc1", "C1CCCCC1N"],
        # Quinone (Michael acceptor) -> Aromatic ring or saturated ring
        "O=C1C=CC(=O)C=C1": ["c1ccccc1", "C1CCCCC1"],
        # Epoxide (reactive) -> Diol or Alkane
        "C1OC1": ["C(O)C(O)", "CC"],
        # Nitro aromatic (hepatotoxic) -> Cyano aromatic or Amide
        "c1ccccc1[N+](=O)[O-]": ["c1ccccc1C#N", "c1ccccc1C(=O)N"],
        # generic aromatic ring -> aliphatic ring
        "c1ccccc1": ["C1CCCCC1"]
    }

def get_bioisostere_replacements(mol, flagged_smarts, cache=None):
    """
    Step 3: Suggests safe replacements for the toxic fragment.
    """
    if cache is None:
        cache = load_mock_chembl_cache()
        
    suggestions = []
    
    if flagged_smarts not in cache:
        return suggestions
        
    replacements = cache[flagged_smarts]
    pattern = Chem.MolFromSmarts(flagged_smarts)
    
    if not pattern or not mol.HasSubstructMatch(pattern):
        return suggestions
        
    original_logp = Descriptors.MolLogP(mol)
    original_mw = Descriptors.MolWt(mol)
    
    # We use ReplaceSubstructs to apply the replacements
    for safe_smarts in replacements:
        # Use MolFromSmiles so the replacement is a real molecule, not a query molecule
        safe_pattern = Chem.MolFromSmiles(safe_smarts)
        if not safe_pattern:
            continue
            
        try:
            # Replace the toxic fragment with the safe one
            new_mols = Chem.ReplaceSubstructs(mol, pattern, safe_pattern)
            
            for new_mol in new_mols:
                Chem.SanitizeMol(new_mol)
                new_smiles = Chem.MolToSmiles(new_mol)
                
                # Filter A: Synthesizability
                sa_score = get_sascore(new_mol)
                if sa_score > 4.0:
                    continue
                    
                # Filter B: ADME Preservation
                new_logp = Descriptors.MolLogP(new_mol)
                new_mw = Descriptors.MolWt(new_mol)
                
                if abs(new_logp - original_logp) > 0.5:
                    continue
                if abs(new_mw - original_mw) > 25.0:
                    continue
                    
                suggestions.append({
                    'smiles': new_smiles,
                    'mol': new_mol,
                    'replacement_type': f"{flagged_smarts} -> {safe_smarts}",
                    'sa_score': round(sa_score, 2),
                    'delta_logp': round(new_logp - original_logp, 2),
                    'delta_mw': round(new_mw - original_mw, 2)
                })
        except Exception as e:
            # Substructure replacement can fail due to valency issues
            pass
            
    return suggestions
