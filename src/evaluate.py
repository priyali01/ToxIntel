import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score, 
    roc_auc_score, 
    f1_score, 
    matthews_corrcoef,
    precision_recall_curve
)

def evaluate_all_endpoints(Y_true, Y_proba, cols):
    """
    Evaluates model predictions across all endpoints.
    Args:
        Y_true: numpy array (N, 12), true labels. -1 indicates missing.
        Y_proba: numpy array (N, 12), predicted probabilities.
        cols: list of endpoint names.
    Returns:
        pd.DataFrame containing metrics per endpoint.
    """
    results = []
    
    for i, ep in enumerate(cols):
        y_t = Y_true[:, i]
        y_p = Y_proba[:, i]
        
        # Filter out missing (-1)
        valid_mask = y_t >= 0
        y_t_valid = y_t[valid_mask]
        y_p_valid = y_p[valid_mask]
        
        n_valid = len(y_t_valid)
        n_toxic = (y_t_valid == 1).sum()
        
        if n_valid < 10 or n_toxic < 2:
            results.append({
                'Endpoint': ep,
                'AUPRC': None,
                'AUROC': None,
                'F1': None,
                'MCC': None,
                'Prevalence': f"{0}%",
                'Valid N': n_valid
            })
            continue
            
        prevalence = n_toxic / n_valid
        
        # Metrics
        auprc = average_precision_score(y_t_valid, y_p_valid)
        auroc = roc_auc_score(y_t_valid, y_p_valid)
        
        # For F1 and MCC, use 0.5 threshold initially
        y_pred_50 = (y_p_valid >= 0.5).astype(int)
        f1 = f1_score(y_t_valid, y_pred_50, zero_division=0)
        mcc = matthews_corrcoef(y_t_valid, y_pred_50)
        
        results.append({
            'Endpoint': ep,
            'AUPRC': round(auprc, 4),
            'AUROC': round(auroc, 4),
            'F1': round(f1, 4),
            'MCC': round(mcc, 4),
            'Prevalence': f"{prevalence*100:.1f}%",
            'Valid N': n_valid
        })
        
    df = pd.DataFrame(results)
    
    # Add macro average row
    valid_auprc = [r['AUPRC'] for r in results if r['AUPRC'] is not None]
    valid_auroc = [r['AUROC'] for r in results if r['AUROC'] is not None]
    valid_f1 = [r['F1'] for r in results if r['F1'] is not None]
    valid_mcc = [r['MCC'] for r in results if r['MCC'] is not None]
    
    macro_row = {
        'Endpoint': 'MACRO AVERAGE',
        'AUPRC': round(np.mean(valid_auprc), 4) if valid_auprc else None,
        'AUROC': round(np.mean(valid_auroc), 4) if valid_auroc else None,
        'F1': round(np.mean(valid_f1), 4) if valid_f1 else None,
        'MCC': round(np.mean(valid_mcc), 4) if valid_mcc else None,
        'Prevalence': 'N/A',
        'Valid N': 'N/A'
    }
    
    # Use pd.concat instead of append
    df = pd.concat([df, pd.DataFrame([macro_row])], ignore_index=True)
    return df

def tune_threshold(y_true, y_prob, recall_floor=0.85):
    """
    Finds the optimal threshold that maximizes precision 
    while strictly keeping recall >= recall_floor.
    """
    precision, recall, thresholds = precision_recall_curve(y_true, y_prob)
    
    # precision_recall_curve returns len(thresholds)+1 points for p/r, 
    # last one is p=1, r=0. We drop the last one.
    precision = precision[:-1]
    recall = recall[:-1]
    
    # Filter by recall floor
    valid_idx = np.where(recall >= recall_floor)[0]
    
    if len(valid_idx) == 0:
        # If model is terrible and can't even hit the floor, fallback
        return {'threshold': 0.5, 'precision': 0.0, 'recall': 0.0}
        
    # Find the one with highest precision among those that meet the recall floor
    best_idx = valid_idx[np.argmax(precision[valid_idx])]
    
    best_threshold = thresholds[best_idx]
    best_precision = precision[best_idx]
    best_recall = recall[best_idx]
    
    return {
        'threshold': best_threshold,
        'precision': best_precision,
        'recall': best_recall
    }
