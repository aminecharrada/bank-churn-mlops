import pandas as pd
import numpy as np
from scipy import stats
import os

def detect_drift(reference_file: str, production_file: str, threshold: float = 0.05) -> dict:
    """
    Detects data drift between reference and production datasets.
    
    Args:
        reference_file: Path to the reference dataset (CSV).
        production_file: Path to the production dataset (CSV).
        threshold: P-value threshold for drift detection.
        
    Returns:
        dict: A dictionary containing drift detection results for each feature.
    """
    results = {}
    
    if not os.path.exists(reference_file):
        # If reference file doesn't exist, we can't detect drift
        return {}
        
    if not os.path.exists(production_file):
        # If production file doesn't exist, we can't detect drift
        return {}

    try:
        ref_df = pd.read_csv(reference_file)
        prod_df = pd.read_csv(production_file)
        
        # Identify common columns
        common_cols = list(set(ref_df.columns) & set(prod_df.columns))
        
        for col in common_cols:
            # Skip non-numeric columns for now, or handle them differently
            if not pd.api.types.is_numeric_dtype(ref_df[col]) or not pd.api.types.is_numeric_dtype(prod_df[col]):
                continue
                
            # Perform Kolmogorov-Smirnov test for numerical features
            statistic, p_value = stats.ks_2samp(ref_df[col], prod_df[col])
            
            drift_detected = p_value < threshold
            
            results[col] = {
                "drift_detected": bool(drift_detected),
                "p_value": float(p_value),
                "statistic": float(statistic),
                "type": "ks_test"
            }
            
    except Exception as e:
        print(f"Error in detect_drift: {e}")
        return {}

    return results
