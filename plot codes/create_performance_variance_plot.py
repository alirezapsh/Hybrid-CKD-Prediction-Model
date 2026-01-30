"""
Standalone script to create performance variance boxplots from saved CV results.

This script can:
1. Read CV fold results from saved files and create boxplots
2. Aggregate test set results from multiple runs (if available)

Usage:
    python create_performance_variance_plot.py

Requirements:
    - results/cv_fold_details.csv (from CV run)
    - OR results/model_results.csv from multiple runs (if you saved them separately)
"""

import os
import json
import pandas as pd
import numpy as np
from utils.evaluation import ComprehensiveEvaluator

def create_cv_variance_boxplot():
    """Create boxplot from CV fold results."""
    cv_file = 'results/cv_fold_details.csv'
    
    if not os.path.exists(cv_file):
        print(f"Error: {cv_file} not found!")
        print("Please run the main script first to generate CV results.")
        return None
    
    # Read CV fold results
    df = pd.read_csv(cv_file)
    
    # Extract accuracy values (convert to percentage if needed)
    accuracy_values = df['Accuracy'].values * 100  # Convert to percentage
    
    # Create boxplot
    evaluator = ComprehensiveEvaluator(class_names=['No CKD', 'CKD'])
    evaluator.plot_performance_variance(
        algorithm_results={'Cross-Validation Folds': accuracy_values.tolist()},
        metric_name='Accuracy',
        save_path='results/figures/cv_performance_variance_boxplot.png'
    )
    
    print(f"✅ CV performance variance boxplot created!")
    print(f"   File: results/figures/cv_performance_variance_boxplot.png")
    print(f"   Shows variance across {len(accuracy_values)} CV folds")
    print(f"   Mean: {np.mean(accuracy_values):.2f}%, Std: {np.std(accuracy_values, ddof=1):.2f}%")
    
    return accuracy_values

def create_multi_run_variance_boxplot():
    """
    Create boxplot from multiple optimization runs.
    
    This requires you to have saved results from multiple runs.
    You can manually collect test set results from each run and create the boxplot.
    """
    # Example: If you have results from multiple runs, you can create a dictionary like this:
    # algorithm_results = {
    #     'EO': [98.04, 97.50],  # Run 1, Run 2
    #     'GWO': [94.12, 95.20],  # Run 1, Run 2
    #     'PSO': [100.00, 86.27],  # Run 1, Run 2
    #     'GA': [82.35, 98.04],  # Run 1, Run 2
    #     'WOA': [96.08, 97.00]  # Run 1, Run 2
    # }
    
    # For now, check if we have multiple model_results.csv files
    results_dir = 'results'
    model_results_files = []
    
    if os.path.exists(results_dir):
        # Look for model_results files (you might have saved them with different names)
        for file in os.listdir(results_dir):
            if 'model_results' in file and file.endswith('.csv'):
                model_results_files.append(os.path.join(results_dir, file))
    
    if len(model_results_files) < 2:
        print("\n⚠️  Not enough runs found for multi-run variance boxplot.")
        print("   You need at least 2 runs per algorithm to create this plot.")
        print("   Current files found:", model_results_files)
        print("\n   To create multi-run boxplots:")
        print("   1. Run each algorithm multiple times")
        print("   2. Save model_results.csv from each run (with different names)")
        print("   3. Or manually collect test set accuracies from each run")
        return None
    
    # If you have multiple files, aggregate them
    # This is a placeholder - you'll need to customize based on how you saved results
    print(f"\nFound {len(model_results_files)} result files.")
    print("To create multi-run boxplots, manually collect test accuracies from each run.")
    
    return None

def main():
    """Main function."""
    print("=" * 80)
    print("PERFORMANCE VARIANCE BOXPLOT GENERATOR")
    print("=" * 80)
    
    os.makedirs('results/figures', exist_ok=True)
    
    # Option 1: Create boxplot from CV fold results (available now)
    print("\n[Option 1] Creating boxplot from CV fold results...")
    cv_values = create_cv_variance_boxplot()
    
    # Option 2: Create boxplot from multiple runs (if available)
    print("\n[Option 2] Checking for multiple run results...")
    create_multi_run_variance_boxplot()
    
    print("\n" + "=" * 80)
    print("COMPLETE!")
    print("=" * 80)
    print("\n📊 Generated plots:")
    print("   - results/figures/cv_performance_variance_boxplot.png")
    print("\n💡 Note: CV results show variance across 5 folds (same for all algorithms).")
    print("   For algorithm-specific variance, you need multiple optimization runs per algorithm.")

if __name__ == "__main__":
    main()
