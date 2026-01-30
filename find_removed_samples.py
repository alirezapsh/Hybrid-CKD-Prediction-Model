"""
Script to identify which samples are removed during initial loading.
This helps identify the 3 samples that are removed (400 -> 397).
"""

import pandas as pd
import numpy as np
from preprocessing import load_arff_file

def find_removed_samples():
    """Find which samples have invalid target labels."""
    
    print("=" * 80)
    print("FINDING REMOVED SAMPLES")
    print("=" * 80)
    
    # Load the ARFF file
    print("\n1. Loading ARFF file...")
    df = load_arff_file('data/chronic_kidney_disease_full.arff')
    
    print(f"\n   Loaded {len(df)} samples from file")
    print(f"   Original shape: {df.shape}")
    
    # Get target column
    target_column = 'class'
    y = df[target_column].copy()
    
    print(f"\n2. Analyzing target column '{target_column}'...")
    print(f"   Unique values in target: {y.unique()}")
    print(f"   Value counts:\n{y.value_counts()}")
    
    # Encode target variable (same as preprocessing.py)
    if y.dtype == object or str(y.dtype).startswith('category'):
        y_clean = y.astype(str).str.strip().str.lower()
        class_map = {'ckd': 1, 'notckd': 0, '1': 1, '0': 0}
        y_numeric = y_clean.map(class_map)
    else:
        y_numeric = pd.to_numeric(y, errors='coerce')
    
    # Find invalid target labels
    valid_mask = y_numeric.isin([0, 1])
    invalid_mask = ~valid_mask
    
    removed_count = int(invalid_mask.sum())
    
    print(f"\n3. Invalid Target Labels Analysis:")
    print(f"   Valid samples: {valid_mask.sum()}")
    print(f"   Invalid samples: {removed_count}")
    
    if removed_count > 0:
        print(f"\n4. REMOVED SAMPLES (Row indices and target values):")
        print("=" * 80)
        
        removed_indices = df.index[invalid_mask].tolist()
        removed_targets = y[invalid_mask]
        
        for idx in removed_indices:
            original_idx = idx  # This is the DataFrame index
            target_value = y.loc[idx]
            cleaned_value = y_clean.loc[idx] if idx in y_clean.index else 'N/A'
            mapped_value = y_numeric.loc[idx] if idx in y_numeric.index else 'N/A'
            
            print(f"\n   Row Index: {original_idx}")
            print(f"   Original target value: '{target_value}'")
            print(f"   Cleaned value: '{cleaned_value}'")
            print(f"   Mapped value: {mapped_value}")
            print(f"   Reason: Target value is not 'ckd', 'notckd', '1', or '0'")
            
            # Show a few feature values for context
            print(f"   Sample features (first 5): {df.loc[idx, df.columns[:5]].tolist()}")
        
        print("\n" + "=" * 80)
        print(f"\nSUMMARY:")
        print(f"   Total samples in file: {len(df)}")
        print(f"   Valid samples: {valid_mask.sum()}")
        print(f"   Removed samples: {removed_count}")
        print(f"   Final count: {valid_mask.sum()} (matches 'Initial Data Shape' in preprocessing)")
    else:
        print("\n   ✓ No samples removed - all target labels are valid!")
    
    # Also check for any other potential issues
    print(f"\n5. Additional Checks:")
    print(f"   Missing values in target column: {y.isnull().sum()}")
    print(f"   Empty strings in target: {(y.astype(str).str.strip() == '').sum()}")
    
    return df, invalid_mask

if __name__ == "__main__":
    df, invalid_mask = find_removed_samples()

