"""
Create Statistical Significance Test Table
Following the methodology from the Persian document:
1. Friedman test for overall differences
2. Paired Wilcoxon signed-rank test for pairwise comparisons
3. Multiple testing correction (Holm-Bonferroni)
4. Effect size (Cliff's delta)
5. Proper table structure with sorting
"""

import numpy as np
import pandas as pd
from scipy import stats
from itertools import combinations

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

# Extract CV fold results from OPTIMIZATION_RESULTS_WITH_CV.md
# Using Run 1 data (lines 388-392, 412-416, 459-464, 484-488, 496-500)

cv_fold_results = {
    'Transformer + EO': [1.0000, 1.0000, 0.9362, 0.9787, 0.9362],  # Run 1, Folds 1-5 (lines 388-392)
    'Transformer + GWO': [1.0000, 1.0000, 0.9362, 0.9574, 0.9362],  # Run 1, Folds 1-5 (lines 412-416)
    'Transformer + PSO': [0.9792, 1.0000, 0.8936, 0.9787, 0.9787],  # Run 1, Folds 1-5 (lines 459-464)
    'Transformer + GA': [1.0000, 1.0000, 0.9362, 0.9787, 0.9787],  # Run 2, Folds 1-5 (lines 484-488)
    'Transformer + WOA': [1.0000, 0.9792, 0.9787, 1.0000, 1.0000]  # Run 1, Folds 1-5 (lines 496-500)
}

# Convert to numpy arrays
for alg in cv_fold_results:
    cv_fold_results[alg] = np.array(cv_fold_results[alg])

print("=" * 100)
print("STATISTICAL SIGNIFICANCE TEST TABLE")
print("=" * 100)
print("\nMethodology:")
print("1. Friedman test for overall differences")
print("2. Paired Wilcoxon signed-rank test for pairwise comparisons")
print("3. Holm-Bonferroni correction for multiple testing")
print("4. Cliff's delta for effect size")
print("\nCV Fold Accuracy Values (5 folds per algorithm):")
for alg, values in cv_fold_results.items():
    print(f"  {alg}: {values} (Mean: {np.mean(values):.4f})")

# Step 1: Friedman test for overall differences
print("\n" + "=" * 100)
print("STEP 1: FRIEDMAN TEST (Overall Differences)")
print("=" * 100)

# Prepare data for Friedman test (algorithms as columns, folds as rows)
data_matrix = np.array([cv_fold_results[alg] for alg in cv_fold_results.keys()]).T
friedman_stat, friedman_p = stats.friedmanchisquare(*data_matrix.T)

print(f"Friedman Test Statistic: {friedman_stat:.4f}")
print(f"P-Value: {friedman_p:.6f}")
if friedman_p < 0.05:
    print("[OK] Significant overall difference (p < 0.05)")
else:
    print("[NO] No significant overall difference (p >= 0.05)")

# Step 2: Pairwise comparisons
print("\n" + "=" * 100)
print("STEP 2: PAIRWISE COMPARISONS (Wilcoxon Signed-Rank Test)")
print("=" * 100)

algorithms = list(cv_fold_results.keys())
results = []

# Generate all pairwise comparisons
for alg1, alg2 in combinations(algorithms, 2):
    values1 = cv_fold_results[alg1]
    values2 = cv_fold_results[alg2]
    
    # Calculate means
    mean1 = np.mean(values1)
    mean2 = np.mean(values2)
    mean_diff = mean1 - mean2
    
    # Wilcoxon signed-rank test
    try:
        statistic, p_value = stats.wilcoxon(values1, values2, alternative='two-sided')
        test_name = 'Wilcoxon Signed-Rank'
    except Exception as e:
        # Fallback to paired t-test if Wilcoxon fails
        statistic, p_value = stats.ttest_rel(values1, values2)
        test_name = 'Paired t-test (fallback)'
    
    # Effect size (Cliff's delta)
    effect_size = cliff_delta(values1, values2)
    
    # Determine effect size category
    abs_effect = abs(effect_size)
    if abs_effect < 0.147:
        effect_category = 'Negligible'
    elif abs_effect < 0.33:
        effect_category = 'Small'
    elif abs_effect < 0.474:
        effect_category = 'Medium'
    else:
        effect_category = 'Large'
    
    results.append({
        'Algorithm 1': alg1,
        'Algorithm 2': alg2,
        'Test': test_name,
        'Statistic': f'{statistic:.4f}',
        'Raw P': f'{p_value:.6f}',
        'Mean 1': f'{mean1:.4f}',
        'Mean 2': f'{mean2:.4f}',
        'Mean Diff': f'{mean_diff:.4f}',
        'Effect Size (d)': f'{effect_size:.4f}',
        'Effect Category': effect_category
    })

# Step 3: Multiple testing correction
print("\n" + "=" * 100)
print("STEP 3: MULTIPLE TESTING CORRECTION (Holm-Bonferroni)")
print("=" * 100)

p_values = np.array([float(r['Raw P']) for r in results])
corrected_p = holm_bonferroni_correction(p_values, alpha=0.05)

# Add corrected p-values to results
for i, r in enumerate(results):
    r['Corrected P'] = f'{corrected_p[i]:.6f}'
    r['Significance'] = 'Yes' if corrected_p[i] < 0.05 else 'No'

# Step 4: Create DataFrame and sort by corrected p-value
df = pd.DataFrame(results)
df = df.sort_values('Corrected P', key=lambda x: x.astype(float))

# Step 5: Display table
print("\n" + "=" * 100)
print("STATISTICAL SIGNIFICANCE TEST RESULTS TABLE")
print("=" * 100)
print("\nNote: Results sorted by corrected p-value (lowest to highest)")
print("Significance level: alpha = 0.05")
print("Multiple testing correction: Holm-Bonferroni")
print("Effect size interpretation: |d| < 0.147 (negligible), < 0.33 (small), < 0.474 (medium), >= 0.474 (large)")
print("\n" + "-" * 100)

# Display table
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', 20)
print(df.to_string(index=False))

print("\n" + "-" * 100)

# Summary statistics
n_significant = sum(df['Significance'] == 'Yes')
n_total = len(df)

print(f"\nSummary:")
print(f"  - Total comparisons: {n_total}")
print(f"  - Significant differences (corrected p < 0.05): {n_significant}")
print(f"  - Non-significant differences: {n_total - n_significant}")

# Save to CSV
output_path = 'results/statistical_significance_table.csv'
df.to_csv(output_path, index=False)
print(f"\n[OK] Table saved to: {output_path}")

# Skip LaTeX (requires jinja2) - can be added later if needed
# latex_table = df.to_latex(index=False, escape=False, float_format="%.6f")
# latex_path = 'results/statistical_significance_table.tex'
# with open(latex_path, 'w', encoding='utf-8') as f:
#     f.write(latex_table)
# print(f"[OK] LaTeX table saved to: {latex_path}")

# Create Markdown table manually (skip pandas to_markdown which requires tabulate)
markdown_path = 'results/statistical_significance_table.md'
with open(markdown_path, 'w', encoding='utf-8') as f:
    f.write("# Statistical Significance Test Results\n\n")
    f.write("**Methodology:**\n")
    f.write("- Friedman test for overall differences\n")
    f.write("- Paired Wilcoxon signed-rank test for pairwise comparisons\n")
    f.write("- Holm-Bonferroni correction for multiple testing\n")
    f.write("- Cliff's delta for effect size\n\n")
    f.write("**Note:** Results sorted by corrected p-value (lowest to highest)\n\n")
    f.write("| Algorithm 1 | Algorithm 2 | Test | Statistic | Raw P | Mean 1 | Mean 2 | Mean Diff | Effect Size (d) | Effect Category | Corrected P | Significance |\n")
    f.write("|-------------|-------------|------|-----------|-------|--------|--------|-----------|-----------------|----------------|-------------|--------------|\n")
    for _, row in df.iterrows():
        f.write(f"| {row['Algorithm 1']} | {row['Algorithm 2']} | {row['Test']} | {row['Statistic']} | {row['Raw P']} | {row['Mean 1']} | {row['Mean 2']} | {row['Mean Diff']} | {row['Effect Size (d)']} | {row['Effect Category']} | {row['Corrected P']} | {row['Significance']} |\n")
print(f"[OK] Markdown table saved to: {markdown_path}")

# Create Matrix-Style Table (like in the image)
print("\n" + "=" * 100)
print("CREATING MATRIX-STYLE TABLE")
print("=" * 100)

# Create a lookup dictionary for p-values (both raw and corrected)
p_value_lookup = {}
raw_p_lookup = {}
for r in results:
    alg1 = r['Algorithm 1']
    alg2 = r['Algorithm 2']
    corrected_p = float(r['Corrected P'])
    raw_p = float(r['Raw P'])
    
    # Store both directions
    p_value_lookup[(alg1, alg2)] = corrected_p
    p_value_lookup[(alg2, alg1)] = corrected_p
    raw_p_lookup[(alg1, alg2)] = raw_p
    raw_p_lookup[(alg2, alg1)] = raw_p

# Define baseline algorithms (columns) - based on image: EO and PSO as baselines
baseline_algorithms = ['Transformer + EO', 'Transformer + PSO']
# All algorithms as rows (excluding baselines to avoid duplicates)
row_algorithms = [a for a in algorithms if a not in baseline_algorithms]
# Add PSO to rows if not already there (as shown in image)
if 'Transformer + PSO' not in row_algorithms:
    row_algorithms.append('Transformer + PSO')
row_algorithms = sorted(row_algorithms)

# Create matrix table in Markdown format matching the image
matrix_md = "\n## Statistical Significance Test\n\n"
matrix_md += "**Note:** This table shows corrected p-values (after Holm-Bonferroni correction).\n\n"
matrix_md += "| Algorithm | " + " | ".join(baseline_algorithms) + " |\n"
matrix_md += "|" + "---|" * (len(baseline_algorithms) + 1) + "\n"

for alg_row in row_algorithms:
    row_values = [alg_row]
    for alg_col in baseline_algorithms:
        if alg_row == alg_col:
            row_values.append("")  # Same algorithm, empty cell
        else:
            # Get p-value for this comparison
            key = (alg_row, alg_col)
            if key in p_value_lookup:
                corrected_p = p_value_lookup[key]
                raw_p = raw_p_lookup[key]
                
                # Display format: show corrected p-value
                if corrected_p < 0.05:
                    p_display = "p < 0.05"
                elif corrected_p < 0.001:
                    p_display = f"p < 0.001"
                elif corrected_p < 0.01:
                    p_display = f"p < 0.01"
                elif corrected_p < 1.0:
                    # Show actual value for non-significant
                    p_display = f"p = {corrected_p:.4f}"
                else:
                    p_display = "p > 0.05"
            else:
                p_display = ""
            row_values.append(p_display)
    matrix_md += "| " + " | ".join(row_values) + " |\n"

matrix_md += "\n**Table Notes:**\n"
matrix_md += "- Number of folds: 5 (5-fold cross-validation)\n"
matrix_md += "- Test type: Wilcoxon Signed-Rank Test (paired, non-parametric)\n"
matrix_md += "- Correction method: Holm-Bonferroni\n"
matrix_md += "- Data: Paired (same folds across algorithms)\n"
matrix_md += "- Significance level: alpha = 0.05\n"

# Save matrix table
matrix_path = 'results/statistical_significance_matrix_table.md'
with open(matrix_path, 'w', encoding='utf-8') as f:
    f.write(matrix_md)
print(f"[OK] Matrix-style table saved to: {matrix_path}")

# Also create CSV version of matrix
matrix_df.to_csv('results/statistical_significance_matrix_table.csv')
print(f"[OK] Matrix CSV saved to: results/statistical_significance_matrix_table.csv")

# Print matrix table
print("\n" + "=" * 100)
print("MATRIX-STYLE STATISTICAL SIGNIFICANCE TABLE")
print("=" * 100)
print(matrix_md)

print("\n" + "=" * 100)
print("COMPLETE!")
print("=" * 100)
