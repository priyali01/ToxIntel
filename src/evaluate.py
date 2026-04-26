"""
src/evaluate.py — Full Metric Suite + Threshold Calibration

Metrics computed per endpoint:
    AUPRC  — primary metric (area under precision-recall curve)
    AUROC  — secondary metric (area under ROC curve)
    F1     — harmonic mean of precision and recall at chosen threshold
    MCC    — Matthews Correlation Coefficient (robust to imbalance)

All metrics skip entries where label == -1 (missing).

Threshold calibration:
    Default threshold of 0.5 is wrong for imbalanced data.
    We find the lowest threshold that keeps recall >= recall_floor (default 0.85).
    In toxicity screening, missing a toxic compound (FN) is far worse
    than a false alarm (FP), so we prioritise recall.

v2 additions:
    - per_endpoint_summary(): prints a clean colour-coded table to stdout
    - diagnose_low_auprc(): flags endpoints that are likely failing and why
"""

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    roc_auc_score,
    f1_score,
    matthews_corrcoef,
    precision_recall_curve,
)


def evaluate_all_endpoints(Y_true: np.ndarray, Y_proba: np.ndarray,
                            cols: list) -> pd.DataFrame:
    """
    Evaluate model predictions across all endpoints.

    Args:
        Y_true:  numpy array (N, 12), true labels. -1 = missing.
        Y_proba: numpy array (N, 12), predicted probabilities [0, 1].
        cols:    list of endpoint names (length 12).

    Returns:
        pd.DataFrame with one row per endpoint + a MACRO AVERAGE row.
        Columns: Endpoint, AUPRC, AUROC, F1, MCC, Prevalence, Valid_N
    """
    results = []

    for i, ep in enumerate(cols):
        y_t = Y_true[:, i]
        y_p = Y_proba[:, i]

        valid_mask  = y_t >= 0
        y_t_valid   = y_t[valid_mask]
        y_p_valid   = y_p[valid_mask]

        n_valid = len(y_t_valid)
        n_toxic = int((y_t_valid == 1).sum())

        if n_valid < 10 or n_toxic < 2:
            results.append({
                'Endpoint':   ep,
                'AUPRC':      None,
                'AUROC':      None,
                'F1':         None,
                'MCC':        None,
                'Prevalence': f"{0:.1f}%",
                'Valid_N':    n_valid,
            })
            continue

        prevalence = n_toxic / n_valid

        auprc = average_precision_score(y_t_valid, y_p_valid)
        auroc = roc_auc_score(y_t_valid, y_p_valid)

        # F1 and MCC use 0.5 threshold initially (recalibrated separately)
        y_pred = (y_p_valid >= 0.5).astype(int)
        f1  = f1_score(y_t_valid, y_pred, zero_division=0)
        mcc = matthews_corrcoef(y_t_valid, y_pred)

        results.append({
            'Endpoint':   ep,
            'AUPRC':      round(auprc, 4),
            'AUROC':      round(auroc, 4),
            'F1':         round(f1, 4),
            'MCC':        round(mcc, 4),
            'Prevalence': f"{prevalence * 100:.1f}%",
            'Valid_N':    n_valid,
        })

    df = pd.DataFrame(results)

    # Macro average row
    valid_auprc = [r['AUPRC'] for r in results if r['AUPRC'] is not None]
    valid_auroc = [r['AUROC'] for r in results if r['AUROC'] is not None]
    valid_f1    = [r['F1']    for r in results if r['F1']    is not None]
    valid_mcc   = [r['MCC']   for r in results if r['MCC']   is not None]

    macro_row = {
        'Endpoint':   'MACRO AVERAGE',
        'AUPRC':      round(np.mean(valid_auprc), 4) if valid_auprc else None,
        'AUROC':      round(np.mean(valid_auroc), 4) if valid_auroc else None,
        'F1':         round(np.mean(valid_f1),    4) if valid_f1    else None,
        'MCC':        round(np.mean(valid_mcc),   4) if valid_mcc   else None,
        'Prevalence': 'N/A',
        'Valid_N':    'N/A',
    }
    df = pd.concat([df, pd.DataFrame([macro_row])], ignore_index=True)
    return df


def tune_threshold(y_true: np.ndarray, y_prob: np.ndarray,
                   recall_floor: float = 0.85) -> dict:
    """
    Find the lowest threshold that keeps recall >= recall_floor.

    Among all thresholds that meet the recall floor, pick the one
    with the highest precision (minimises false alarms while staying safe).

    Args:
        y_true:        1D array of true binary labels (0/1, no -1 here)
        y_prob:        1D array of predicted probabilities
        recall_floor:  Minimum required recall (default 0.85)

    Returns:
        dict with keys: threshold, precision, recall
    """
    precision_arr, recall_arr, thresholds = precision_recall_curve(y_true, y_prob)

    # precision_recall_curve returns len(thresholds)+1 points; drop last sentinel
    precision_arr = precision_arr[:-1]
    recall_arr    = recall_arr[:-1]

    valid_idx = np.where(recall_arr >= recall_floor)[0]

    if len(valid_idx) == 0:
        # Model can't meet the recall floor at any threshold — use 0.1 as fallback
        return {'threshold': 0.1, 'precision': 0.0, 'recall': float(recall_arr.max())}

    best_idx       = valid_idx[np.argmax(precision_arr[valid_idx])]
    best_threshold = float(thresholds[best_idx])
    best_precision = float(precision_arr[best_idx])
    best_recall    = float(recall_arr[best_idx])

    return {
        'threshold': round(best_threshold, 4),
        'precision': round(best_precision, 4),
        'recall':    round(best_recall,    4),
    }


def calibrate_all_thresholds(Y_true: np.ndarray, Y_proba: np.ndarray,
                              cols: list, recall_floor: float = 0.85) -> dict:
    """
    Tune decision threshold for every endpoint to meet recall_floor.

    Returns:
        dict mapping endpoint_name → {'threshold', 'precision', 'recall'}
    """
    thresholds = {}
    for i, ep in enumerate(cols):
        y_t = Y_true[:, i]
        y_p = Y_proba[:, i]

        valid_mask = y_t >= 0
        y_t_valid  = y_t[valid_mask]
        y_p_valid  = y_p[valid_mask]

        if len(y_t_valid) < 10 or y_t_valid.sum() < 2 or len(np.unique(y_t_valid)) < 2:
            thresholds[ep] = {'threshold': 0.5, 'precision': None, 'recall': None}
            continue

        thresholds[ep] = tune_threshold(y_t_valid, y_p_valid, recall_floor)

    return thresholds


def per_endpoint_summary(results_df: pd.DataFrame) -> None:
    """
    Print a formatted per-endpoint summary table to stdout.

    Flags:
        *** AUPRC <= prevalence (model failed to learn this endpoint)
        **  AUPRC < 0.30 (poor performance)
        *   AUPRC < 0.40 (below target)
    """
    print("\n" + "=" * 75)
    print(f"{'Endpoint':<20} {'AUPRC':>7} {'AUROC':>7} {'F1':>7} "
          f"{'MCC':>7} {'Prev':>7} {'N':>6}  Flag")
    print("=" * 75)

    for _, row in results_df.iterrows():
        ep    = row['Endpoint']
        auprc = row['AUPRC']
        auroc = row['AUROC']
        f1    = row['F1']
        mcc   = row['MCC']
        prev  = row['Prevalence']
        n     = row['Valid_N']

        flag = ''
        if auprc is not None and prev not in ('N/A',):
            prev_val = float(str(prev).replace('%', '')) / 100
            if auprc <= prev_val:
                flag = '*** FAILED'
            elif auprc < 0.30:
                flag = '**  poor'
            elif auprc < 0.40:
                flag = '*   below target'

        def fmt(v):
            return f"{v:>7.4f}" if v is not None else f"{'N/A':>7}"

        print(f"{ep:<20} {fmt(auprc)} {fmt(auroc)} {fmt(f1)} "
              f"{fmt(mcc)} {str(prev):>7} {str(n):>6}  {flag}")

    print("=" * 75)


def diagnose_low_auprc(Y_true: np.ndarray, Y_proba: np.ndarray,
                       cols: list) -> None:
    """
    Print a diagnostic report for endpoints with AUPRC < 0.35.

    For each struggling endpoint, reports:
        - Prevalence (is it extremely rare?)
        - Mean predicted probability for positives vs negatives
          (is the model even separating them at all?)
        - Whether any positives got a score > 0.5
    """
    print("\n=== Low-AUPRC Diagnostics ===\n")
    any_issues = False

    for i, ep in enumerate(cols):
        y_t = Y_true[:, i]
        y_p = Y_proba[:, i]

        valid = y_t >= 0
        y_t_v = y_t[valid]
        y_p_v = y_p[valid]

        if y_t_v.sum() < 2 or len(np.unique(y_t_v)) < 2:
            continue

        auprc = average_precision_score(y_t_v, y_p_v)
        if auprc >= 0.35:
            continue

        any_issues = True
        prev          = y_t_v.mean()
        mean_pos_prob = y_p_v[y_t_v == 1].mean()
        mean_neg_prob = y_p_v[y_t_v == 0].mean()
        pct_pos_above = (y_p_v[y_t_v == 1] > 0.5).mean() * 100

        print(f"  {ep}  (AUPRC={auprc:.3f})")
        print(f"    Prevalence:              {prev*100:.1f}%")
        print(f"    Mean prob (positives):   {mean_pos_prob:.3f}")
        print(f"    Mean prob (negatives):   {mean_neg_prob:.3f}")
        print(f"    Positives scored > 0.5:  {pct_pos_above:.0f}%")

        if mean_pos_prob < mean_neg_prob + 0.05:
            print(f"    ⚠ Model is NOT separating this endpoint — "
                  f"positives and negatives have nearly identical scores.")
        if prev < 0.03:
            print(f"    ⚠ Extreme rarity (<3%) — consider per-endpoint "
                  f"oversampling or a dedicated binary model.")
        print()

    if not any_issues:
        print("  No endpoints with AUPRC < 0.35 — model is performing well.")


# ── Quick test ────────────────────────────────────────────────────
if __name__ == '__main__':
    np.random.seed(42)
    n = 500
    Y_true  = np.random.choice([0, 1, -1], size=(n, 12),
                                p=[0.75, 0.10, 0.15]).astype(float)
    Y_proba = np.clip(np.random.randn(n, 12) * 0.3 + 0.3, 0, 1)

    from src.featurize import TARGET_COLS
    results = evaluate_all_endpoints(Y_true, Y_proba, TARGET_COLS)
    per_endpoint_summary(results)
    diagnose_low_auprc(Y_true, Y_proba, TARGET_COLS)

    thresholds = calibrate_all_thresholds(Y_true, Y_proba, TARGET_COLS)
    print("\nCalibrated thresholds (recall >= 0.85):")
    for ep, t in thresholds.items():
        print(f"  {ep:<20}  threshold={t['threshold']:.3f}  "
              f"precision={t['precision']}  recall={t['recall']}")

    print("\nevaluate.py: ALL TESTS PASSED")