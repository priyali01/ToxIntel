import numpy as np

def evaluate_swap_pareto(orig_probs, new_probs, target_cols, threshold=0.01):
    """
    Step 4d: Pareto evaluation of the original vs new molecule.
    
    A lower probability is better (less toxic).
    If all endpoints are strictly better (lower) or equal, it DOMINATES.
    If all endpoints are strictly worse (higher) or equal, it is DOMINATED.
    If some are better and some are worse, it's a TRADE-OFF.
    
    Args:
        orig_probs: numpy array of original molecule probabilities (12,)
        new_probs: numpy array of new molecule probabilities (12,)
        target_cols: list of endpoint names
        threshold: minimum difference in probability to be considered a change
    
    Returns:
        dict containing the status and details of improved/worsened endpoints.
    """
    diffs = orig_probs - new_probs  # Positive diff = improvement (safer)
    
    improved = []
    worsened = []
    
    for i, col in enumerate(target_cols):
        if diffs[i] > threshold:
            improved.append({'endpoint': col, 'delta': round(diffs[i], 4)})
        elif diffs[i] < -threshold:
            worsened.append({'endpoint': col, 'delta': round(diffs[i], 4)})
            
    status = "NO_CHANGE"
    if len(improved) > 0 and len(worsened) == 0:
        status = "DOMINATES"
    elif len(worsened) > 0 and len(improved) == 0:
        status = "DOMINATED"
    elif len(improved) > 0 and len(worsened) > 0:
        status = "TRADE-OFF"
        
    return {
        'status': status,
        'improved_count': len(improved),
        'worsened_count': len(worsened),
        'improved_details': improved,
        'worsened_details': worsened
    }
