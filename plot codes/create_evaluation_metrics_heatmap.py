"""
Create Comparative Heatmap for Evaluation Metrics
Based on mean values from 3 runs of each algorithm
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Set style
sns.set_style("whitegrid")
plt.rcParams['font.size'] = 11
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['xtick.labelsize'] = 10
plt.rcParams['ytick.labelsize'] = 10

def create_evaluation_heatmap(save_path='results/figures/evaluation_metrics_heatmap.png'):
    """
    Create a comparative heatmap for evaluation metrics across algorithms.
    
    Features:
    - Algorithms as rows (Transformer + Optimizer)
    - Metrics as columns (Accuracy, F1-Score, AUC)
    - Green gradient color scheme (light to dark)
    - Min-max normalization per column
    - Sorted by average performance
    - Numerical values displayed in cells
    """
    
    # Data from 3 runs (mean values)
    # Accuracy: percentage (convert to decimal for display)
    # F1-Score: already decimal
    # AUC: percentage (convert to decimal for display)
    
    data = {
        'Transformer + EO': {
            'Accuracy': 96.08,  # percentage
            'F1 Score': 0.9609,  # decimal
            'AUC': 98.81  # percentage
        },
        'Transformer + PSO': {
            'Accuracy': 96.73,
            'F1 Score': 0.9674,
            'AUC': 99.84
        },
        'Transformer + GWO': {
            'Accuracy': 95.43,
            'F1 Score': 0.9543,
            'AUC': 97.36
        },
        'Transformer + GA': {
            'Accuracy': 90.85,
            'F1 Score': 0.9077,
            'AUC': 97.54
        },
        'Transformer + WOA': {
            'Accuracy': 96.73,
            'F1 Score': 0.9674,
            'AUC': 98.55
        }
    }
    
    # Create DataFrame
    df = pd.DataFrame(data).T
    
    # Reorder columns
    df = df[['Accuracy', 'F1 Score', 'AUC']]
    
    # Calculate average performance for sorting
    # Normalize each metric to 0-1 scale for fair averaging
    df_normalized = df.copy()
    for col in df_normalized.columns:
        if col == 'F1 Score':
            # F1-Score is already 0-1, but we'll normalize for consistency
            df_normalized[col] = (df_normalized[col] - df_normalized[col].min()) / (df_normalized[col].max() - df_normalized[col].min())
        else:
            # Accuracy and AUC are percentages, normalize to 0-1
            df_normalized[col] = (df_normalized[col] - df_normalized[col].min()) / (df_normalized[col].max() - df_normalized[col].min())
    
    # Sort by average normalized performance (descending)
    df['Avg Performance'] = df_normalized.mean(axis=1)
    df = df.sort_values('Avg Performance', ascending=False)
    df = df.drop('Avg Performance', axis=1)
    
    # Create normalized version for color mapping (per column)
    df_norm = df.copy()
    for col in df_norm.columns:
        if col == 'F1 Score':
            # F1-Score: normalize 0-1 range
            df_norm[col] = (df_norm[col] - df_norm[col].min()) / (df_norm[col].max() - df_norm[col].min())
        else:
            # Accuracy and AUC: normalize percentage range
            df_norm[col] = (df_norm[col] - df_norm[col].min()) / (df_norm[col].max() - df_norm[col].min())
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Create heatmap with green gradient
    # Using custom colormap: light green to dark green
    from matplotlib.colors import LinearSegmentedColormap
    colors = ['#e8f5e9', '#c8e6c9', '#a5d6a7', '#81c784', '#66bb6a', '#4caf50', '#388e3c', '#2e7d32', '#1b5e20']
    n_bins = 256
    cmap = LinearSegmentedColormap.from_list('green_gradient', colors, N=n_bins)
    
    # Create annotation matrix with proper formatting - all as percentages
    annot_df = df.copy()
    for col in annot_df.columns:
        if col == 'F1 Score':
            # Convert F1 Score from decimal (0-1) to percentage
            annot_df[col] = annot_df[col].apply(lambda x: f'{x * 100:.2f}')
        else:
            # Accuracy and AUC are already percentages, just format
            annot_df[col] = annot_df[col].apply(lambda x: f'{x:.2f}')
    
    # Create heatmap using normalized values for color, but display original values
    sns.heatmap(
        df_norm,
        annot=annot_df,  # Display formatted original values
        fmt='',  # Empty fmt since we're using formatted strings
        cmap=cmap,
        cbar_kws={'label': 'Normalized Performance'},
        linewidths=0.5,
        linecolor='white',
        ax=ax,
        vmin=0,
        vmax=1
    )
    
    # Customize text colors for better readability
    for i, row in enumerate(df.index):
        for j, col in enumerate(df.columns):
            text = ax.texts[i * len(df.columns) + j]
            # Make text bold and white for dark cells, black for light cells
            if df_norm.iloc[i, j] > 0.5:
                text.set_color('white')
                text.set_weight('bold')
                text.set_fontsize(11)
            else:
                text.set_color('black')
                text.set_weight('normal')
                text.set_fontsize(11)
    
    # Set labels
    ax.set_title('Evaluation Metrics', fontsize=16, fontweight='bold', pad=20)
    ax.set_xlabel('')  # Remove x-axis label
    ax.set_ylabel('Algorithms', fontsize=12, fontweight='bold')
    
    # Move x-axis labels (metric names) to the top
    ax.xaxis.set_label_position('top')
    ax.xaxis.tick_top()
    
    # Rotate y-axis labels for better readability
    plt.yticks(rotation=0)
    plt.xticks(rotation=0)
    
    # Remove footnotes - no description under the plot
    
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.1, top=0.9)
    
    # Save figure
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"[OK] Heatmap saved to: {save_path}")
    
    # Also save as high-resolution version
    save_path_hr = save_path.replace('.png', '_highres.png')
    plt.savefig(save_path_hr, dpi=600, bbox_inches='tight')
    print(f"[OK] High-resolution heatmap saved to: {save_path_hr}")
    
    plt.close()
    
    return df, df_norm

if __name__ == '__main__':
    print("=" * 80)
    print("Creating Evaluation Metrics Heatmap")
    print("=" * 80)
    
    df, df_norm = create_evaluation_heatmap()
    
    print("\nData Summary:")
    print(df)
    print("\nNormalized Data (for color mapping):")
    print(df_norm)
    print("\n[OK] Heatmap generation complete!")
