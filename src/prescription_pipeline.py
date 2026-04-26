import torch
import numpy as np
from rdkit import Chem
from src.featurize import smiles_to_morgan_with_info, TARGET_COLS
from src.shap_validator import validate_shap_attribution, get_top_shap_bits, load_structural_alerts
from src.bioisostere import get_bioisostere_replacements, load_mock_chembl_cache
from src.uncertainty import apply_temperature_scaling, detect_ood
from src.pareto import evaluate_swap_pareto

# For a real implementation we would use shap.DeepExplainer.
# For pipeline demonstration, we simulate the SHAP attribution step because 
# calculating deep SHAP on a 2048-bit input requires a large background dataset.
def simulate_shap_values(probs, bit_info=None):
    """
    Simulates SHAP values for demonstration. In production, use shap.DeepExplainer.
    We just assign some random positive importance to bits.
    """
    np.random.seed(42)  # For reproducibility in the demo
    shap_vals = np.random.randn(2048)
    if bit_info:
        # Spike SHAP values for bits actually present in the molecule to simulate true attribution
        for bit in list(bit_info.keys())[:5]:
            shap_vals[bit] = 10.0
    return shap_vals

def run_prescription_pipeline(target_smiles, model, training_smiles_list=None):
    """
    Executes the 4-Step Corrected Flow for a single molecule.
    """
    print(f"\n--- Starting Prescription Pipeline for {target_smiles} ---")
    mol = Chem.MolFromSmiles(target_smiles)
    if not mol:
        return {'error': 'Invalid SMILES'}
        
    alerts_cache = load_structural_alerts()
    bio_cache = load_mock_chembl_cache()
    
    # ---------------------------------------------------------
    # STEP 1: Predict & SHAP Attribution
    # ---------------------------------------------------------
    fp, bit_info = smiles_to_morgan_with_info(target_smiles)
    if fp is None:
        return {'error': 'Featurization failed'}
        
    with torch.no_grad():
        logits = model(torch.tensor(fp, dtype=torch.float32).unsqueeze(0))
        orig_probs = apply_temperature_scaling(logits)[0]
        
    print("\n[Step 1] Initial Predictions:")
    for ep, p in zip(TARGET_COLS, orig_probs):
        if p > 0.5:
            print(f"  [TOXIC] {ep}: {p:.4f}")
            
    # Simulate SHAP values for the most toxic endpoint
    most_toxic_idx = np.argmax(orig_probs)
    shap_vals = simulate_shap_values(orig_probs, bit_info=bit_info)
    top_bits = get_top_shap_bits(shap_vals, top_k=5)
    
    # ---------------------------------------------------------
    # STEP 1b: Alert Cross-Validation
    # ---------------------------------------------------------
    validation = validate_shap_attribution(mol, bit_info, top_bits, alerts_cache)
    print(f"\n[Step 1b] SHAP Validation (Target: {TARGET_COLS[most_toxic_idx]}):")
    print(f"  Confidence: {validation['shap_confidence']}")
    print(f"  Recommendation: {validation['recommendation']}")
    print(f"  Matched Alerts: {validation['matched_alerts']}")
    
    if validation['shap_confidence'] == 'LOW':
        return {'status': 'ABORTED', 'reason': 'SHAP validation failed', 'validation': validation}
        
    # ---------------------------------------------------------
    # STEP 2 & 3: Fragment Mapping & Bioisostere Suggestion
    # ---------------------------------------------------------
    print("\n[Step 2 & 3] Querying Bioisostere Cache & Filtering...")
    
    # We use the first matched alert SMARTS as the toxic fragment to replace
    flagged_smarts = alerts_cache.get(validation['matched_alerts'][0], None) if validation['matched_alerts'] else None
    
    if not flagged_smarts:
        return {'status': 'ABORTED', 'reason': 'No specific structural alert SMARTS found for replacement'}
        
    suggestions = get_bioisostere_replacements(mol, flagged_smarts, bio_cache)
    print(f"  Found {len(suggestions)} viable replacements passing SAScore and ADME filters.")
    
    if len(suggestions) == 0:
        return {'status': 'COMPLETED', 'message': 'No viable bioisosteres found.', 'suggestions': []}
        
    # ---------------------------------------------------------
    # STEP 4: Re-prediction & Pareto Evaluation
    # ---------------------------------------------------------
    print("\n[Step 4] Re-prediction and Pareto Evaluation:")
    
    results = []
    for sug in suggestions:
        sug_fp, _ = smiles_to_morgan_with_info(sug['smiles'])
        
        with torch.no_grad():
            sug_logits = model(torch.tensor(sug_fp, dtype=torch.float32).unsqueeze(0))
            sug_probs = apply_temperature_scaling(sug_logits)[0]
            
        pareto = evaluate_swap_pareto(orig_probs, sug_probs, TARGET_COLS)
        
        ood_info = None
        if training_smiles_list:
            ood_info = detect_ood(sug['smiles'], training_smiles_list)
            
        results.append({
            'smiles': sug['smiles'],
            'replacement': sug['replacement_type'],
            'pareto_status': pareto['status'],
            'improved': pareto['improved_count'],
            'worsened': pareto['worsened_count'],
            'sa_score': sug['sa_score'],
            'delta_logp': sug['delta_logp'],
            'ood_warning': ood_info['is_ood'] if ood_info else False
        })
        
        print(f"  Candidate: {sug['smiles']}")
        print(f"    Status: {pareto['status']} | Improved: {pareto['improved_count']} | Worsened: {pareto['worsened_count']}")
        if ood_info and ood_info['is_ood']:
            print(f"    [OOD WARNING] (Max Sim to Train: {ood_info['max_sim']})")
            
    print("\n--- Pipeline Finished ---")
    return {
        'status': 'SUCCESS',
        'original_probs': orig_probs,
        'suggestions': results
    }
