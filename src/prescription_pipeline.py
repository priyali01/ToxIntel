import torch
import numpy as np
from rdkit import Chem
from src.featurize import smiles_to_morgan_with_info, TARGET_COLS
from src.shap_validator import validate_shap_attribution, get_top_shap_bits, load_structural_alerts
from src.bioisostere import get_bioisostere_replacements, load_mock_chembl_cache
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

def run_prescription_pipeline(target_smiles, model_artifact):
    """
    Executes the 4-Step Corrected Flow for a single molecule using the serialized model artifact.
    """
    print(f"\n--- Starting Prescription Pipeline for {target_smiles} ---")
    mol = Chem.MolFromSmiles(target_smiles)
    if not mol:
        return {'error': 'Invalid SMILES'}
        
    alerts_cache = load_structural_alerts()
    bio_cache = load_mock_chembl_cache()
    
    model = model_artifact['mondrian_predictor'].base_model
    mondrian = model_artifact['mondrian_predictor']
    
    # ---------------------------------------------------------
    # STEP 1: Predict & SHAP Attribution
    # ---------------------------------------------------------
    fp, bit_info = smiles_to_morgan_with_info(target_smiles)
    if fp is None:
        return {'error': 'Featurization failed'}
        
    orig_results = mondrian.predict_with_uncertainty(target_smiles, fp)
    orig_probs = np.array([r['prob'] for r in orig_results])
        
    print("\n[Step 1] Initial Predictions (Mondrian Calibrated):")
    for ep, r in zip(TARGET_COLS, orig_results):
        p = r['prob']
        u = r['uncertain']
        g = r['mondrian_group']
        if p > 0.5:
            print(f"  [TOXIC] {ep}: {p:.4f} (Uncertain: {u}, Group: {g})")
            
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
        
        sug_results = mondrian.predict_with_uncertainty(sug['smiles'], sug_fp)
        sug_probs = np.array([r['prob'] for r in sug_results])
            
        pareto = evaluate_swap_pareto(orig_probs, sug_probs, TARGET_COLS)
        
        # We consider the molecule OOD if it is in the 'novel' group
        # or if the predictions are highly uncertain.
        is_novel = sug_results[0]['mondrian_group'] == 'novel'
        highly_uncertain = sum([1 for r in sug_results if r['uncertain']]) > 6
            
        results.append({
            'smiles': sug['smiles'],
            'replacement': sug['replacement_type'],
            'pareto_status': pareto['status'],
            'improved': pareto['improved_count'],
            'worsened': pareto['worsened_count'],
            'sa_score': sug['sa_score'],
            'delta_logp': sug['delta_logp'],
            'ood_warning': is_novel or highly_uncertain,
            'mondrian_group': sug_results[0]['mondrian_group']
        })
        
        print(f"  Candidate: {sug['smiles']}")
        print(f"    Status: {pareto['status']} | Improved: {pareto['improved_count']} | Worsened: {pareto['worsened_count']}")
        if is_novel or highly_uncertain:
            print(f"    [OOD WARNING] Mondrian Group: {sug_results[0]['mondrian_group']}")
            
    print("\n--- Pipeline Finished ---")
    return {
        'status': 'SUCCESS',
        'original_probs': orig_probs,
        'suggestions': results
    }
