import numpy as np
import torch
from rdkit import Chem
from rdkit import DataStructs
from rdkit.Chem import AllChem

def get_tanimoto_similarity(smiles1, smiles2):
    """Computes Tanimoto similarity between two SMILES."""
    mol1 = Chem.MolFromSmiles(smiles1)
    mol2 = Chem.MolFromSmiles(smiles2)
    if not mol1 or not mol2:
        return 0.0
        
    fp1 = AllChem.GetMorganFingerprintAsBitVect(mol1, 2, nBits=2048)
    fp2 = AllChem.GetMorganFingerprintAsBitVect(mol2, 2, nBits=2048)
    
    return DataStructs.TanimotoSimilarity(fp1, fp2)

def detect_ood(target_smiles, training_smiles_list, threshold=0.4):
    """
    Step 4c: OOD detection.
    Computes the maximum Tanimoto similarity to the training set.
    If max similarity < threshold, it flags the prediction as Out-Of-Distribution (OOD).
    
    Args:
        target_smiles: SMILES string of the generated molecule.
        training_smiles_list: List of SMILES strings from the training set.
        threshold: Tanimoto similarity threshold.
    """
    # For a real system, we'd precompute and store the training fingerprints in a matrix.
    # For Phase 6 pipeline demonstration, we assume a representative subset or full list is passed.
    
    target_mol = Chem.MolFromSmiles(target_smiles)
    if not target_mol:
        return {'is_ood': True, 'max_sim': 0.0, 'message': 'Invalid SMILES'}
        
    target_fp = AllChem.GetMorganFingerprintAsBitVect(target_mol, 2, nBits=2048)
    
    max_sim = 0.0
    # In production, vectorize this with BulkTanimotoSimilarity
    for train_smi in training_smiles_list:
        train_mol = Chem.MolFromSmiles(train_smi)
        if not train_mol:
            continue
        train_fp = AllChem.GetMorganFingerprintAsBitVect(train_mol, 2, nBits=2048)
        sim = DataStructs.TanimotoSimilarity(target_fp, train_fp)
        if sim > max_sim:
            max_sim = sim
            
    is_ood = max_sim < threshold
    return {
        'is_ood': is_ood,
        'max_sim': round(max_sim, 4),
        'message': 'Warning: Low confidence (OOD)' if is_ood else 'In-Distribution'
    }

def apply_temperature_scaling(logits, temperatures=None):
    """
    Calibrates logits into probabilities using temperature scaling.
    If temperatures are None, defaults to T=1.0 (standard sigmoid).
    """
    # In a full implementation, we'd fit T per endpoint on the validation set using LBFGS.
    # Here we apply the scaling operation.
    if temperatures is None:
        temperatures = torch.ones(logits.shape[1])
        
    scaled_logits = logits / temperatures
    return torch.sigmoid(scaled_logits).detach().numpy()
