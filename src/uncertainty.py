import numpy as np
import torch
from rdkit import Chem
from rdkit import DataStructs
from rdkit.Chem import AllChem
from rdkit.Chem.Scaffolds import MurckoScaffold
from sklearn.base import BaseEstimator, ClassifierMixin

from mapie.classification import MapieClassifier
from mapie.conformity_scores import LACConformityScore

def get_scaffold(smiles: str) -> str:
    mol = Chem.MolFromSmiles(smiles)
    if not mol: return ''
    return MurckoScaffold.MurckoScaffoldSmiles(mol=mol, includeChirality=False)

def assign_mondrian_group(smiles: str, cal_scaffolds: set, tanimoto_threshold: float = 0.4) -> str:
    """Assigns molecule to 'known', 'similar', or 'novel' based on calibration set overlap."""
    scaffold = get_scaffold(smiles)
    if scaffold in cal_scaffolds: return 'known'
    
    # Calculate max Tanimoto similarity to calibration scaffolds
    mol = Chem.MolFromSmiles(scaffold)
    if not mol: return 'novel'
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)
    
    max_sim = 0.0
    for cs in cal_scaffolds:
        cmol = Chem.MolFromSmiles(cs)
        if cmol:
            cfp = AllChem.GetMorganFingerprintAsBitVect(cmol, 2, nBits=2048)
            sim = DataStructs.TanimotoSimilarity(fp, cfp)
            if sim > max_sim: max_sim = sim
            
    return 'similar' if max_sim >= tanimoto_threshold else 'novel'

class ToxNetWrapper(BaseEstimator, ClassifierMixin):
    """Scikit-learn wrapper for a single endpoint of ToxNet to be compatible with MAPIE."""
    def __init__(self, model, endpoint_idx):
        self.model = model
        self.endpoint_idx = endpoint_idx
        self.classes_ = np.array([0, 1])
        
    def fit(self, X, y):
        # Already trained
        return self
        
    def predict_proba(self, X):
        self.model.eval()
        with torch.no_grad():
            x_tensor = torch.tensor(X, dtype=torch.float32)
            logits = self.model(x_tensor)
            probs = torch.sigmoid(logits[:, self.endpoint_idx]).numpy()
        # Return [P(y=0), P(y=1)]
        return np.vstack([1 - probs, probs]).T
        
    def predict(self, X):
        return (self.predict_proba(X)[:, 1] > 0.5).astype(int)

class MondrianToxPredictor:
    """Mondrian CP restores coverage guarantees under scaffold distribution shift for all 12 endpoints."""
    def __init__(self, base_model, n_tasks=12, alpha=0.1):
        self.base_model = base_model
        self.alpha = alpha
        self.n_tasks = n_tasks
        self.cal_scaffolds = set()
        # mapie_models[endpoint_idx][group] = MapieClassifier
        self.mapie_models = {i: {} for i in range(n_tasks)}
    
    def fit_calibration(self, X_cal, y_cal, smiles_cal):
        self.cal_scaffolds = {get_scaffold(s) for s in smiles_cal if get_scaffold(s)}
        groups = np.array([assign_mondrian_group(s, self.cal_scaffolds) for s in smiles_cal])
        
        for ep in range(self.n_tasks):
            wrapper = ToxNetWrapper(self.base_model, ep)
            
            # Mask out missing labels
            valid_mask = y_cal[:, ep] != -1
            
            for group in ['known', 'similar', 'novel']:
                mask = (groups == group) & valid_mask
                if mask.sum() > 10: # Need enough calibration points
                    mapie = MapieClassifier(estimator=wrapper, cv='prefit', conformity_score=LACConformityScore())
                    mapie.fit(X_cal[mask], y_cal[mask, ep])
                    self.mapie_models[ep][group] = mapie
        return self
        
    def predict_with_uncertainty(self, smiles, fp):
        """Returns predictions and conformal prediction sets for the given fingerprint."""
        group = assign_mondrian_group(smiles, self.cal_scaffolds)
        
        X = fp.reshape(1, -1)
        results = []
        
        for ep in range(self.n_tasks):
            wrapper = ToxNetWrapper(self.base_model, ep)
            prob = wrapper.predict_proba(X)[0, 1]
            
            mapie = self.mapie_models[ep].get(group)
            if mapie:
                # predict returns y_pred, y_pis
                _, y_pis = mapie.predict(X, alpha=self.alpha)
                # y_pis shape: (n_samples, n_classes, n_alphas)
                # It tells us which classes are in the prediction set.
                # If both classes are in the set [True, True], uncertainty is high.
                pred_set = y_pis[0, :, 0] 
                uncertain = pred_set[0] and pred_set[1]
            else:
                uncertain = True # Fallback if no calibration data for this group
                
            results.append({
                'prob': prob,
                'uncertain': uncertain,
                'mondrian_group': group
            })
            
        return results

def apply_temperature_scaling(logits, temperatures=None):
    if temperatures is None:
        temperatures = torch.ones(logits.shape[1])
    scaled_logits = logits / temperatures
    return torch.sigmoid(scaled_logits).detach().numpy()
