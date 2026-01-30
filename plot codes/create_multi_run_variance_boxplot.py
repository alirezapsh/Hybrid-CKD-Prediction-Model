"""
Script to create variance boxplots across 3 runs for each algorithm.

This script extracts test set results from OPTIMIZATION_RESULTS_WITH_CV.md
and creates boxplots showing variance across the 3 runs for each algorithm.

Usage:
    python create_multi_run_variance_boxplot.py

Output:
    - results/figures/multi_run_variance_boxplot_accuracy.png
    - results/figures/multi_run_variance_boxplot_f1_score.png
    - results/figures/multi_run_variance_boxplot_roc_auc.png
"""

import os
import re
import numpy as np
import matplotlib.pyplot as plt
from utils.evaluation import ComprehensiveEvaluator

def extract_results_from_markdown(md_file='OPTIMIZATION_RESULTS_WITH_CV.md'):
    """
    Extract test set results for all 3 runs of each algorithm from markdown file.
    
    Returns:
        dict: {
            'EO': {'accuracy': [run1, run2, run3], 'f1_score': [...], 'roc_auc': [...]},
            'GWO': {...},
            ...
        }
    """
    if not os.path.exists(md_file):
        print(f"Error: {md_file} not found!")
        return None
    
    with open(md_file, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Dictionary to store results
    results = {
        'EO': {'accuracy': [], 'f1_score': [], 'roc_auc': []},
        'GWO': {'accuracy': [], 'f1_score': [], 'roc_auc': []},
        'PSO': {'accuracy': [], 'f1_score': [], 'roc_auc': []},
        'GA': {'accuracy': [], 'f1_score': [], 'roc_auc': []},
        'WOA': {'accuracy': [], 'f1_score': [], 'roc_auc': []}
    }
    
    # Pattern to match test set results table
    # Format: | **Algorithm (with CV)** | Run X | accuracy% | precision | recall | f1_score | roc_auc | pr_auc |
    pattern = r'\|\s*\*\*(\w+)\s*\(with CV\)\*\*\s*\|\s*Run\s+(\d+)\s*\|\s*([\d.]+)%\s*\|\s*([\d.]+)\s*\|\s*([\d.]+)\s*\|\s*([\d.]+)\s*\|\s*([\d.]+)\s*\|\s*([\d.]+)\s*\|'
    
    matches = re.findall(pattern, content)
    
    for match in matches:
        algorithm = match[0].upper()
        run_num = int(match[1])
        accuracy = float(match[2])
        precision = float(match[3])
        recall = float(match[4])
        f1_score = float(match[5])
        roc_auc = float(match[6])
        pr_auc = float(match[7])
        
        if algorithm in results:
            # Store results (will be sorted by run number later)
            results[algorithm][f'run_{run_num}'] = {
                'accuracy': accuracy,
                'f1_score': f1_score,
                'roc_auc': roc_auc
            }
    
    # Reorganize by metric
    final_results = {}
    for alg in results:
        final_results[alg] = {
            'accuracy': [],
            'f1_score': [],
            'roc_auc': []
        }
        
        # Extract runs in order (1, 2, 3)
        for run_num in [1, 2, 3]:
            run_key = f'run_{run_num}'
            if run_key in results[alg]:
                final_results[alg]['accuracy'].append(results[alg][run_key]['accuracy'])
                final_results[alg]['f1_score'].append(results[alg][run_key]['f1_score'])
                final_results[alg]['roc_auc'].append(results[alg][run_key]['roc_auc'])
    
    return final_results

def create_variance_boxplots(results_dict, save_dir='results/figures'):
    """
    Create variance boxplots for each metric across all algorithms.
    
    Args:
        results_dict: Dictionary with algorithm results
        save_dir: Directory to save plots
    """
    os.makedirs(save_dir, exist_ok=True)
    
    evaluator = ComprehensiveEvaluator(class_names=['No CKD', 'CKD'])
    
    metrics = {
        'accuracy': 'Accuracy',
        'f1_score': 'F1-Score',
        'roc_auc': 'ROC-AUC'
    }
    
    for metric_key, metric_name in metrics.items():
        # Prepare data for boxplot
        algorithm_results = {}
        for alg, metrics_dict in results_dict.items():
            if metric_key in metrics_dict and len(metrics_dict[metric_key]) > 0:
                # Convert to percentage for accuracy, keep as decimal for others
                if metric_key == 'accuracy':
                    values = metrics_dict[metric_key]  # Already in percentage
                else:
                    # Convert to percentage for F1-Score and ROC-AUC
                    values = [v * 100 for v in metrics_dict[metric_key]]
                
                algorithm_results[alg] = values
        
        if not algorithm_results:
            print(f"Warning: No data found for {metric_name}")
            continue
        
        # Create boxplot
        print(f"\nCreating {metric_name} variance boxplot...")
        print(f"   Algorithms: {list(algorithm_results.keys())}")
        for alg, values in algorithm_results.items():
            print(f"   {alg}: {values} (mean={np.mean(values):.2f}%, std={np.std(values, ddof=1):.2f}%)")
        
        # Use the existing plot_performance_variance function
        save_path = os.path.join(save_dir, f'multi_run_variance_boxplot_{metric_key}.png')
        
        evaluator.plot_performance_variance(
            algorithm_results=algorithm_results,
            metric_name=metric_name,
            save_path=save_path
        )
        
        print(f"   ✅ Saved to: {save_path}")

def main():
    """Main function."""
    print("=" * 80)
    print("MULTI-RUN VARIANCE BOXPLOT GENERATOR")
    print("=" * 80)
    
    # Extract results from markdown
    print("\n[Step 1] Extracting results from OPTIMIZATION_RESULTS_WITH_CV.md...")
    results = extract_results_from_markdown()
    
    if results is None:
        print("Error: Could not extract results from markdown file.")
        return
    
    # Display extracted results
    print("\n[Step 2] Extracted results:")
    for alg, metrics_dict in results.items():
        print(f"\n{alg}:")
        for metric, values in metrics_dict.items():
            if values:
                print(f"  {metric}: {values} (n={len(values)})")
    
    # Create boxplots
    print("\n[Step 3] Creating variance boxplots...")
    create_variance_boxplots(results)
    
    print("\n" + "=" * 80)
    print("COMPLETE!")
    print("=" * 80)
    print("\n📊 Generated plots:")
    print("   - results/figures/multi_run_variance_boxplot_accuracy.png")
    print("   - results/figures/multi_run_variance_boxplot_f1_score.png")
    print("   - results/figures/multi_run_variance_boxplot_roc_auc.png")
    print("\n💡 These plots show variance across 3 runs for each algorithm.")
    print("   Lower variance = more stable/consistent algorithm.")

if __name__ == "__main__":
    main()
