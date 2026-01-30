"""
Statistical Significance Test Table: Proposed Algorithm vs Baselines
Following exact requirements:
1. Friedman test (global test)
2. Wilcoxon signed-rank test (Proposed vs each baseline)
3. Holm-Bonferroni correction
4. Cliff's delta effect size
5. Formatted Markdown table with specific columns
"""

import numpy as np
import pandas as pd
from scipy import stats

def cliff_delta(x, y):
    """
    Calculate Cliff's delta effect size.
    Returns value between -1 and 1:
    - |d| < 0.147: negligible
    - |d| < 0.33: small
    - |d| < 0.474: medium
    - |d| >= 0.474: large
    """
    n_x = len(x)
    n_y = len(y)
    
    # Count how many times x > y and x < y
    dominance = 0
    for xi in x:
        for yj in y:
            if xi > yj:
                dominance += 1
            elif xi < yj:
                dominance -= 1
    
    delta = dominance / (n_x * n_y)
    return delta

def holm_bonferroni_correction(p_values, alpha=0.05):
    """
    Apply Holm-Bonferroni correction for multiple testing.
    """
    n = len(p_values)
    sorted_indices = np.argsort(p_values)
    corrected = np.zeros(n)
    
    for i, idx in enumerate(sorted_indices):
        corrected[idx] = min(p_values[idx] * (n - i), 1.0)
    
    return corrected

def interpret_effect_size(delta):
    """
    Interpret Cliff's delta effect size.
    """
    abs_delta = abs(delta)
    if abs_delta < 0.147:
        return 'Negligible'
    elif abs_delta < 0.33:
        return 'Small'
    elif abs_delta < 0.474:
        return 'Medium'
    else:
        return 'Large'

# ============================================================================
# DATA: Test Set Accuracy Results (3 runs per algorithm)
# ============================================================================
# Extract from OPTIMIZATION_RESULTS_WITH_CV.md
# Proposed Algorithm: EO (Equilibrium Optimizer)
# Baseline Algorithms: GA, PSO, GWO, WOA

proposed_algorithm = 'Transformer + EO'
proposed_results = np.array([98.04, 96.08, 94.12])  # Run 1, Run 2, Run 3

baseline_algorithms = {
    'Transformer + GA': np.array([82.35, 98.04, 92.16]),  # Run 1, Run 2, Run 3
    'Transformer + PSO': np.array([98.04, 96.08, 96.08]),  # Run 1, Run 2, Run 3
    'Transformer + GWO': np.array([94.12, 94.12, 98.04]),  # Run 1, Run 2, Run 3
    'Transformer + WOA': np.array([96.08, 98.04, 96.08])  # Run 1, Run 2, Run 3
}

# All algorithms for Friedman test
all_results = {proposed_algorithm: proposed_results}
all_results.update(baseline_algorithms)

print("=" * 100)
print("STATISTICAL SIGNIFICANCE TEST: PROPOSED ALGORITHM vs BASELINES")
print("=" * 100)
print("\nData: Test Set Accuracy (3 independent runs per algorithm)")
print(f"\nProposed Algorithm: {proposed_algorithm}")
print(f"  Results: {proposed_results} (Mean: {np.mean(proposed_results):.2f}%)")
print("\nBaseline Algorithms:")
for alg_name, alg_results in baseline_algorithms.items():
    print(f"  {alg_name}: {alg_results} (Mean: {np.mean(alg_results):.2f}%)")

# ============================================================================
# STEP 1: FRIEDMAN TEST (Global Test)
# ============================================================================
print("\n" + "=" * 100)
print("STEP 1: FRIEDMAN TEST (Overall Differences)")
print("=" * 100)

# Prepare data for Friedman test (algorithms as columns, runs as rows)
data_matrix = np.array([all_results[alg] for alg in all_results.keys()]).T
friedman_stat, friedman_p = stats.friedmanchisquare(*data_matrix.T)

print(f"Friedman Test Statistic: {friedman_stat:.4f}")
print(f"P-Value: {friedman_p:.6f}")
if friedman_p < 0.05:
    print("[OK] Significant overall difference detected (p < 0.05)")
    print("     Proceeding with pairwise comparisons...")
else:
    print("[NO] No significant overall difference (p >= 0.05)")
    print("     Note: This does not preclude significant pairwise differences")

# ============================================================================
# STEP 2: PAIRWISE COMPARISONS (Wilcoxon Signed-Rank Test)
# ============================================================================
print("\n" + "=" * 100)
print("STEP 2: PAIRWISE COMPARISONS (Wilcoxon Signed-Rank Test)")
print("=" * 100)
print(f"Comparing {proposed_algorithm} vs each baseline algorithm...")

results = []
raw_p_values = []

for baseline_name, baseline_results in baseline_algorithms.items():
    # Wilcoxon signed-rank test (paired, non-parametric)
    try:
        statistic, p_value = stats.wilcoxon(
            proposed_results, 
            baseline_results, 
            alternative='two-sided'
        )
        test_name = 'Wilcoxon Signed-Rank'
    except Exception as e:
        # Fallback to paired t-test if Wilcoxon fails (e.g., all differences are zero)
        statistic, p_value = stats.ttest_rel(proposed_results, baseline_results)
        test_name = 'Paired t-test (fallback)'
        print(f"  Warning: Wilcoxon failed for {baseline_name}, using t-test")
    
    # Effect size (Cliff's delta)
    effect_size = cliff_delta(proposed_results, baseline_results)
    interpretation = interpret_effect_size(effect_size)
    
    # Calculate means for reference
    mean_proposed = np.mean(proposed_results)
    mean_baseline = np.mean(baseline_results)
    mean_diff = mean_proposed - mean_baseline
    
    results.append({
        'Comparison': f'Proposed vs {baseline_name.replace("Transformer + ", "")}',
        'Test Used': test_name,
        'Raw p-value': p_value,
        'Statistic': statistic,
        'Mean Proposed': mean_proposed,
        'Mean Baseline': mean_baseline,
        'Mean Difference': mean_diff,
        'Effect Size (Cliff\'s Delta)': effect_size,
        'Interpretation': interpretation
    })
    
    raw_p_values.append(p_value)
    print(f"  {baseline_name}: p = {p_value:.6f}, Effect Size = {effect_size:.4f} ({interpretation})")

# ============================================================================
# STEP 3: MULTIPLE TESTING CORRECTION (Holm-Bonferroni)
# ============================================================================
print("\n" + "=" * 100)
print("STEP 3: MULTIPLE TESTING CORRECTION (Holm-Bonferroni)")
print("=" * 100)

raw_p_values = np.array(raw_p_values)
corrected_p_values = holm_bonferroni_correction(raw_p_values, alpha=0.05)

# Add corrected p-values and significance to results
for i, result in enumerate(results):
    result['Corrected p-value (Holm)'] = corrected_p_values[i]
    result['Significance (alpha=0.05)'] = 'Yes' if corrected_p_values[i] < 0.05 else 'No'
    print(f"  {result['Comparison']}: Raw p = {result['Raw p-value']:.6f}, "
          f"Corrected p = {corrected_p_values[i]:.6f}, "
          f"Significant: {result['Significance (alpha=0.05)']}")

# ============================================================================
# STEP 4: CREATE FINAL TABLE
# ============================================================================
print("\n" + "=" * 100)
print("STEP 4: FINAL STATISTICAL SIGNIFICANCE TABLE")
print("=" * 100)

# Create DataFrame with required columns
table_data = []
for result in results:
    effect_size_val = result['Effect Size (Cliff\'s Delta)']
    table_data.append({
        'Comparison': result['Comparison'],
        'Test Used': result['Test Used'],
        'Raw p-value': f"{result['Raw p-value']:.6f}",
        'Corrected p-value (Holm)': f"{result['Corrected p-value (Holm)']:.6f}",
        'Significance (Yes/No at alpha=0.05)': result['Significance (alpha=0.05)'],
        'Effect Size (Cliff\'s Delta)': f"{effect_size_val:.4f}",
        'Interpretation': result['Interpretation']
    })

df = pd.DataFrame(table_data)

# Sort by corrected p-value (lowest to highest)
df = df.sort_values('Corrected p-value (Holm)', key=lambda x: x.astype(float))

# Display table
print("\n" + "-" * 100)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', 30)
print(df.to_string(index=False))
print("-" * 100)

# Summary
n_significant = sum(df['Significance (Yes/No at alpha=0.05)'] == 'Yes')
n_total = len(df)
print(f"\nSummary:")
print(f"  - Total comparisons: {n_total}")
print(f"  - Significant differences (corrected p < 0.05): {n_significant}")
print(f"  - Non-significant differences: {n_total - n_significant}")

# ============================================================================
# SAVE RESULTS
# ============================================================================
# Save to CSV
csv_path = 'results/statistical_significance_proposed_vs_baselines.csv'
df.to_csv(csv_path, index=False)
print(f"\n[OK] Table saved to: {csv_path}")

# Save to Markdown
md_path = 'results/statistical_significance_proposed_vs_baselines.md'
with open(md_path, 'w', encoding='utf-8') as f:
    f.write("# Statistical Significance Test: Proposed Algorithm vs Baselines\n\n")
    f.write("## Methodology\n\n")
    f.write("1. **Global Test**: Friedman test to detect overall differences\n")
    f.write("2. **Pairwise Test**: Wilcoxon signed-rank test (non-parametric, paired)\n")
    f.write("3. **P-value Correction**: Holm-Bonferroni method for multiple testing\n")
    f.write("4. **Effect Size**: Cliff's delta to measure magnitude of difference\n\n")
    
    f.write("## Data\n\n")
    f.write(f"- **Proposed Algorithm**: {proposed_algorithm}\n")
    f.write(f"  - Results: {proposed_results} (Mean: {np.mean(proposed_results):.2f}%)\n")
    f.write("\n- **Baseline Algorithms**:\n")
    for alg_name, alg_results in baseline_algorithms.items():
        f.write(f"  - {alg_name}: {alg_results} (Mean: {np.mean(alg_results):.2f}%)\n")
    
    f.write("\n## Global Test Results (Friedman Test)\n\n")
    f.write(f"- **Test Statistic**: {friedman_stat:.4f}\n")
    f.write(f"- **P-Value**: {friedman_p:.6f}\n")
    if friedman_p < 0.05:
        f.write(f"- **Conclusion**: Significant overall difference detected (p < 0.05)\n")
    else:
        f.write(f"- **Conclusion**: No significant overall difference (p >= 0.05)\n")
    
    f.write("\n## Statistical Significance Test Results\n\n")
    f.write("**Note**: Results sorted by corrected p-value (lowest to highest)\n")
    f.write("**Significance level**: alpha = 0.05\n")
    f.write("**Effect size interpretation**: |d| < 0.147 (negligible), < 0.33 (small), < 0.474 (medium), >= 0.474 (large)\n\n")
    
    f.write("| Comparison | Test Used | Raw p-value | Corrected p-value (Holm) | Significance (Yes/No at alpha=0.05) | Effect Size (Cliff's Delta) | Interpretation |\n")
    f.write("|------------|-----------|-------------|---------------------------|-------------------------------------|-------------------------------|----------------|\n")
    
    for _, row in df.iterrows():
        effect_size_col = 'Effect Size (Cliff\'s Delta)'
        f.write(f"| {row['Comparison']} | {row['Test Used']} | {row['Raw p-value']} | {row['Corrected p-value (Holm)']} | {row['Significance (Yes/No at alpha=0.05)']} | {row[effect_size_col]} | {row['Interpretation']} |\n")
    
    f.write("\n## Summary\n\n")
    f.write(f"- **Total comparisons**: {n_total}\n")
    f.write(f"- **Significant differences** (corrected p < 0.05): {n_significant}\n")
    f.write(f"- **Non-significant differences**: {n_total - n_significant}\n")

print(f"[OK] Markdown table saved to: {md_path}")

print("\n" + "=" * 100)
print("COMPLETE!")
print("=" * 100)
