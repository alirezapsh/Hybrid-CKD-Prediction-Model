import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, confusion_matrix,
    classification_report, roc_curve, precision_recall_curve
)
from sklearn.calibration import calibration_curve
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from typing import Dict, Tuple, Optional, List
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import Patch
import seaborn as sns
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# Try to import UMAP (optional dependency)
try:
    import umap
    HAS_UMAP = True
except ImportError:
    HAS_UMAP = False
    warnings.warn("UMAP not available. Will use t-SNE for embeddings visualization.")


class ComprehensiveEvaluator:
    """
    Comprehensive evaluator for classification tasks.
    Provides all metrics required for Q1/Q2 paper publication.
    """
    
    def __init__(self, class_names: Optional[List[str]] = None):
        """
        Initialize evaluator.
        
        Args:
            class_names: List of class names (e.g., ['No CKD', 'CKD'])
        """
        self.class_names = class_names if class_names is not None else ['Class 0', 'Class 1']
    
    def evaluate(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_proba: Optional[np.ndarray] = None,
        verbose: bool = True,
        n_features: Optional[int] = None,
        fitness: Optional[float] = None,
        runtime_seconds: Optional[float] = None,
        cv_variance: Optional[Dict[str, float]] = None,
        cv_fold_results: Optional[Dict[str, List[float]]] = None,
        baseline_algorithm: str = 'EO'
    ) -> Dict:
        """
        Comprehensive evaluation of predictions.
        
        Returns all metrics needed for experiment objective:
        - Model accuracy: Accuracy, F1-score, AUC
        - Number of selected features: n_features
        - Execution time: runtime_seconds
        - Stability (variance): cv_variance (standard deviation across CV folds)
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            y_proba: Predicted probabilities (optional, for ROC-AUC)
            verbose: Whether to print results
            n_features: Number of features used (optional)
            fitness: Fitness value from optimization (optional)
            runtime_seconds: Runtime in seconds (optional)
            cv_variance: Dictionary with CV standard deviations for stability (optional)
                         Format: {'accuracy_std': 0.0190, 'f1_score_std': 0.0190, 'roc_auc_std': 0.0015}
            cv_fold_results: Dictionary with CV fold results for statistical tests (optional)
                            Format: {'EO': [1.0, 1.0, 0.9574, ...], 'GWO': [1.0, 1.0, ...], ...}
            baseline_algorithm: Algorithm name for statistical comparison (default: 'EO')
            
        Returns:
            Dictionary with ALL metrics including:
            - Performance: accuracy, f1_score, roc_auc
            - Features: n_features
            - Time: runtime_seconds
            - Fitness: fitness
            - Stability: accuracy_std, f1_score_std, roc_auc_std
            - Algorithm statistics: algorithm_statistics (dict with mean, std, min, max for each algorithm)
            - Statistical comparisons: statistical_comparisons (list), statistical_summary (dict)
        """
        # Convert to numpy arrays and ensure proper types
        if isinstance(y_true, pd.Series) or isinstance(y_true, pd.DataFrame):
            y_true = y_true.values
        y_true = np.asarray(y_true, dtype=np.int64).flatten()
        
        if isinstance(y_pred, pd.Series) or isinstance(y_pred, pd.DataFrame):
            y_pred = y_pred.values
        y_pred = np.asarray(y_pred, dtype=np.int64).flatten()
        
        if y_proba is not None:
            if isinstance(y_proba, pd.Series) or isinstance(y_proba, pd.DataFrame):
                y_proba = y_proba.values
            y_proba = np.asarray(y_proba, dtype=np.float64)
            if y_proba.ndim > 1:
                y_proba = y_proba.flatten() if y_proba.shape[1] == 1 else y_proba
        
        # Store original inputs for classification report
        self._y_true = y_true
        self._y_pred = y_pred
        
        metrics = {}
        
        # Basic classification metrics
        metrics['accuracy'] = accuracy_score(y_true, y_pred)
        metrics['precision'] = precision_score(y_true, y_pred, average='weighted', zero_division=0)
        metrics['recall'] = recall_score(y_true, y_pred, average='weighted', zero_division=0)
        metrics['f1_score'] = f1_score(y_true, y_pred, average='weighted', zero_division=0)
        
        # Per-class metrics
        metrics['precision_per_class'] = precision_score(y_true, y_pred, average=None, zero_division=0)
        metrics['recall_per_class'] = recall_score(y_true, y_pred, average=None, zero_division=0)
        metrics['f1_per_class'] = f1_score(y_true, y_pred, average=None, zero_division=0)
        
        # ROC-AUC and PR-AUC (if probabilities provided)
        if y_proba is not None:
            try:
                # Ensure y_proba is 1D for binary classification
                if y_proba.ndim > 1:
                    if y_proba.shape[1] == 2:
                        # Binary classification with 2 columns - use positive class
                        y_proba_1d = y_proba[:, 1]
                    elif y_proba.shape[1] == 1:
                        # Single column - use as is
                        y_proba_1d = y_proba.flatten()
                    else:
                        # Multi-class - use positive class (column 1)
                        y_proba_1d = y_proba[:, 1] if y_proba.shape[1] > 1 else y_proba[:, 0]
                else:
                    y_proba_1d = y_proba.flatten()
                
                # Ensure same length
                min_len = min(len(y_true), len(y_proba_1d))
                y_true_roc = y_true[:min_len]
                y_proba_roc = y_proba_1d[:min_len]
                
                metrics['roc_auc'] = roc_auc_score(y_true_roc, y_proba_roc)
                metrics['pr_auc'] = average_precision_score(y_true_roc, y_proba_roc)
            except Exception as e:
                if verbose:
                    print(f"Warning: Could not calculate ROC-AUC/PR-AUC: {e}")
                metrics['roc_auc'] = None
                metrics['pr_auc'] = None
        else:
            metrics['roc_auc'] = None
            metrics['pr_auc'] = None
        
        # Confusion matrix
        metrics['confusion_matrix'] = confusion_matrix(y_true, y_pred)
        
        # Classification report
        metrics['classification_report'] = classification_report(
            y_true, y_pred, 
            target_names=self.class_names,
            output_dict=True
        )
        
        # Additional metrics for algorithm comparison
        metrics['n_features'] = n_features
        metrics['fitness'] = fitness
        metrics['runtime_seconds'] = runtime_seconds
        
        # CV variance (stability) - standard deviation across folds
        if cv_variance is not None:
            metrics['cv_variance'] = cv_variance
            metrics['accuracy_std'] = cv_variance.get('accuracy_std', None)
            metrics['f1_score_std'] = cv_variance.get('f1_score_std', None)
            metrics['roc_auc_std'] = cv_variance.get('roc_auc_std', None)
        else:
            metrics['cv_variance'] = None
            metrics['accuracy_std'] = None
            metrics['f1_score_std'] = None
            metrics['roc_auc_std'] = None
        
        # Statistical test results for each algorithm (if CV fold results provided)
        if cv_fold_results is not None:
            try:
                # Calculate statistics for each algorithm individually
                algorithm_stats = {}
                for alg_name, fold_values in cv_fold_results.items():
                    values = np.array(fold_values)
                    algorithm_stats[alg_name] = {
                        'mean': float(np.mean(values)),
                        'std': float(np.std(values, ddof=1)) if len(values) > 1 else 0.0,
                        'min': float(np.min(values)),
                        'max': float(np.max(values)),
                        'n_folds': len(values)
                    }
                
                # Also perform pairwise comparisons if baseline_algorithm is specified
                if baseline_algorithm in cv_fold_results:
                    comparison_results = statistical_comparison(
                        cv_results=cv_fold_results,
                        baseline_algorithm=baseline_algorithm,
                        metric='accuracy',
                        alpha=0.05,
                        test_type='wilcoxon',
                        save_path=None
                    )
                    metrics['statistical_comparisons'] = comparison_results.to_dict('records')
                    metrics['statistical_summary'] = {
                        'n_significant': int(sum(comparison_results['Significant'] == 'Yes')),
                        'n_better': int(sum('significantly better' in c and baseline_algorithm in c 
                                           for c in comparison_results['Conclusion']))
                    }
                else:
                    metrics['statistical_comparisons'] = None
                    metrics['statistical_summary'] = None
                
                metrics['algorithm_statistics'] = algorithm_stats
            except Exception as e:
                if verbose:
                    print(f"Warning: Statistical tests failed: {e}")
                metrics['algorithm_statistics'] = None
                metrics['statistical_comparisons'] = None
                metrics['statistical_summary'] = None
        else:
            metrics['algorithm_statistics'] = None
            metrics['statistical_comparisons'] = None
            metrics['statistical_summary'] = None
        
        if verbose:
            self._print_metrics(metrics, baseline_algorithm)
        
        return metrics
    
    def _print_metrics(self, metrics: Dict, baseline_algorithm: str = 'EO'):
        """Print evaluation metrics."""
        print("\n" + "=" * 80)
        print("COMPREHENSIVE EVALUATION METRICS")
        print("=" * 80)
        
        print(f"\n📊 Overall Metrics:")
        print(f"   Accuracy:  {metrics['accuracy']:.4f} ({metrics['accuracy']*100:.2f}%)")
        print(f"   Precision: {metrics['precision']:.4f}")
        print(f"   Recall:    {metrics['recall']:.4f}")
        print(f"   F1-Score:  {metrics['f1_score']:.4f}")
        
        if metrics['roc_auc'] is not None:
            print(f"   ROC-AUC:   {metrics['roc_auc']:.4f}")
        if metrics['pr_auc'] is not None:
            print(f"   PR-AUC:    {metrics['pr_auc']:.4f}")
        
        # Additional metrics
        if metrics.get('n_features') is not None:
            print(f"   #Features: {metrics['n_features']}")
        if metrics.get('fitness') is not None:
            print(f"   Fitness:   {metrics['fitness']:.4f}")
        if metrics.get('runtime_seconds') is not None:
            print(f"   Runtime:   {metrics['runtime_seconds']:.1f} seconds ({metrics['runtime_seconds']/60:.2f} minutes)")
        
        # Stability (variance) - CV standard deviation
        if metrics.get('accuracy_std') is not None:
            print(f"\n📊 Stability (Variance across CV folds):")
            print(f"   Accuracy Std:  {metrics['accuracy_std']:.4f}")
            if metrics.get('f1_score_std') is not None:
                print(f"   F1-Score Std:  {metrics['f1_score_std']:.4f}")
            if metrics.get('roc_auc_std') is not None:
                print(f"   ROC-AUC Std:   {metrics['roc_auc_std']:.4f}")
            print(f"   (Lower variance = More stable algorithm)")
        
        # Algorithm statistics (individual - for each algorithm)
        if metrics.get('algorithm_statistics') is not None:
            print(f"\n📊 Algorithm Statistics (from CV folds):")
            for alg_name, stats in metrics['algorithm_statistics'].items():
                print(f"   {alg_name}:")
                print(f"      Mean: {stats['mean']:.4f}")
                print(f"      Std:  {stats['std']:.4f}")
                print(f"      Range: [{stats['min']:.4f}, {stats['max']:.4f}]")
                print(f"      Folds: {stats['n_folds']}")
        
        # Statistical test comparisons (if baseline provided)
        if metrics.get('statistical_comparisons') is not None:
            print(f"\n📊 Statistical Test Comparisons ({baseline_algorithm} vs Others):")
            for test_result in metrics['statistical_comparisons']:
                print(f"   {test_result['Comparison']}:")
                print(f"      P-Value: {test_result['P_Value']}")
                print(f"      Significant: {test_result['Significant']}")
                print(f"      Conclusion: {test_result['Conclusion']}")
            if metrics.get('statistical_summary') is not None:
                summary = metrics['statistical_summary']
                print(f"\n   Summary: {baseline_algorithm} is significantly better than {summary['n_better']} algorithm(s)")
        
        print(f"\n📋 Per-Class Metrics:")
        for i, class_name in enumerate(self.class_names):
            print(f"   {class_name}:")
            print(f"      Precision: {metrics['precision_per_class'][i]:.4f}")
            print(f"      Recall:    {metrics['recall_per_class'][i]:.4f}")
            print(f"      F1-Score:  {metrics['f1_per_class'][i]:.4f}")
        
        print(f"\n📊 Confusion Matrix:")
        print(metrics['confusion_matrix'])
        
        print(f"\n📄 Classification Report:")
        report_str = classification_report(
            self._y_true,
            self._y_pred,
            target_names=self.class_names,
            output_dict=False
        )
        print(report_str)
    
    def plot_confusion_matrix(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        save_path: Optional[str] = None,
        figsize: Tuple[int, int] = (8, 6)
    ):
        """
        Plot confusion matrix.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            save_path: Path to save figure (optional)
            figsize: Figure size
        """
        # Convert to numpy arrays
        if isinstance(y_true, pd.Series) or isinstance(y_true, pd.DataFrame):
            y_true = y_true.values
        y_true = np.asarray(y_true, dtype=np.int64).flatten()
        
        if isinstance(y_pred, pd.Series) or isinstance(y_pred, pd.DataFrame):
            y_pred = y_pred.values
        y_pred = np.asarray(y_pred, dtype=np.int64).flatten()
        
        cm = confusion_matrix(y_true, y_pred)
        
        plt.figure(figsize=figsize)
        sns.heatmap(
            cm, 
            annot=True, 
            fmt='d', 
            cmap='Blues',
            xticklabels=self.class_names,
            yticklabels=self.class_names
        )
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.title('Confusion Matrix')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Confusion matrix saved to {save_path}")
        else:
            plt.show()
        plt.close()
    
    def plot_roc_curve(
        self,
        y_true: np.ndarray,
        y_proba: np.ndarray,
        save_path: Optional[str] = None,
        figsize: Tuple[int, int] = (8, 6)
    ):
        """
        Plot ROC curve.
        
        Args:
            y_true: True labels
            y_proba: Predicted probabilities
            save_path: Path to save figure (optional)
            figsize: Figure size
        """
        # Convert to numpy arrays
        if isinstance(y_true, pd.Series) or isinstance(y_true, pd.DataFrame):
            y_true = y_true.values
        y_true = np.asarray(y_true, dtype=np.int64).flatten()
        
        if isinstance(y_proba, pd.Series) or isinstance(y_proba, pd.DataFrame):
            y_proba = y_proba.values
        y_proba = np.asarray(y_proba, dtype=np.float64)
        
        # Handle 2D probability arrays
        if y_proba.ndim > 1:
            if y_proba.shape[1] == 2:
                y_proba = y_proba[:, 1]
            elif y_proba.shape[1] == 1:
                y_proba = y_proba.flatten()
            else:
                y_proba = y_proba[:, 1] if y_proba.shape[1] > 1 else y_proba[:, 0]
        else:
            y_proba = y_proba.flatten()
        
        # Ensure same length
        min_len = min(len(y_true), len(y_proba))
        y_true = y_true[:min_len]
        y_proba = y_proba[:min_len]
        
        fpr, tpr, _ = roc_curve(y_true, y_proba)
        roc_auc = roc_auc_score(y_true, y_proba)
        
        plt.figure(figsize=figsize)
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.4f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic (ROC) Curve')
        plt.legend(loc="lower right")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"ROC curve saved to {save_path}")
        else:
            plt.show()
        plt.close()
    
    def plot_pr_curve(
        self,
        y_true: np.ndarray,
        y_proba: np.ndarray,
        save_path: Optional[str] = None,
        figsize: Tuple[int, int] = (8, 6)
    ):
        """
        Plot Precision-Recall curve.
        
        Args:
            y_true: True labels
            y_proba: Predicted probabilities
            save_path: Path to save figure (optional)
            figsize: Figure size
        """
        # Convert to numpy arrays
        if isinstance(y_true, pd.Series) or isinstance(y_true, pd.DataFrame):
            y_true = y_true.values
        y_true = np.asarray(y_true, dtype=np.int64).flatten()
        
        if isinstance(y_proba, pd.Series) or isinstance(y_proba, pd.DataFrame):
            y_proba = y_proba.values
        y_proba = np.asarray(y_proba, dtype=np.float64)
        
        # Handle 2D probability arrays
        if y_proba.ndim > 1:
            if y_proba.shape[1] == 2:
                y_proba = y_proba[:, 1]
            elif y_proba.shape[1] == 1:
                y_proba = y_proba.flatten()
            else:
                y_proba = y_proba[:, 1] if y_proba.shape[1] > 1 else y_proba[:, 0]
        else:
            y_proba = y_proba.flatten()
        
        # Ensure same length
        min_len = min(len(y_true), len(y_proba))
        y_true = y_true[:min_len]
        y_proba = y_proba[:min_len]
        
        precision, recall, _ = precision_recall_curve(y_true, y_proba)
        pr_auc = average_precision_score(y_true, y_proba)
        
        plt.figure(figsize=figsize)
        plt.plot(recall, precision, color='darkorange', lw=2, label=f'PR curve (AUC = {pr_auc:.4f})')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curve')
        plt.legend(loc="lower left")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"PR curve saved to {save_path}")
        else:
            plt.show()
        plt.close()
    
    def plot_training_history(
        self,
        history: Dict[str, List[float]],
        save_path: Optional[str] = None,
        figsize: Tuple[int, int] = (12, 5)
    ):
        """
        Plot training history (loss and accuracy curves).
        
        Args:
            history: Dictionary with 'train_loss', 'val_loss', 'train_accuracy', 'val_accuracy'
            save_path: Path to save figure (optional)
            figsize: Figure size
        """
        fig, axes = plt.subplots(1, 2, figsize=figsize)
        
        epochs = range(1, len(history.get('train_loss', [])) + 1)
        
        # Loss plot
        axes[0].plot(epochs, history.get('train_loss', []), 'b-', label='Training Loss', linewidth=2)
        if 'val_loss' in history and len(history['val_loss']) > 0:
            axes[0].plot(epochs, history.get('val_loss', []), 'r-', label='Validation Loss', linewidth=2)
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Loss')
        axes[0].set_title('Training and Validation Loss')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Accuracy plot
        if 'train_accuracy' in history:
            axes[1].plot(epochs, history['train_accuracy'], 'b-', label='Training Accuracy', linewidth=2)
        if 'val_accuracy' in history and len(history['val_accuracy']) > 0:
            axes[1].plot(epochs, history['val_accuracy'], 'r-', label='Validation Accuracy', linewidth=2)
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Accuracy (%)')
        axes[1].set_title('Training and Validation Accuracy')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Training history saved to {save_path}")
        else:
            plt.show()
        plt.close()
    
    def save_results(
        self,
        metrics: Dict,
        filepath: str,
        model_name: str = "Model"
    ):
        """
        Save evaluation results to CSV/JSON.
        
        Args:
            metrics: Evaluation metrics dictionary
            filepath: Path to save results
            model_name: Name of the model
        """
        # Prepare results DataFrame
        results = {
            'Model': [model_name],
            'Accuracy': [metrics['accuracy']],
            'Precision': [metrics['precision']],
            'Recall': [metrics['recall']],
            'F1_Score': [metrics['f1_score']],
        }
        
        if metrics['roc_auc'] is not None:
            results['ROC_AUC'] = [metrics['roc_auc']]
        if metrics['pr_auc'] is not None:
            results['PR_AUC'] = [metrics['pr_auc']]
        
        # Additional metrics
        if metrics.get('n_features') is not None:
            results['N_Features'] = [metrics['n_features']]
        if metrics.get('fitness') is not None:
            results['Fitness'] = [metrics['fitness']]
        if metrics.get('runtime_seconds') is not None:
            results['Runtime_Seconds'] = [metrics['runtime_seconds']]
        
        # Stability (variance) - CV standard deviation
        if metrics.get('accuracy_std') is not None:
            results['Accuracy_Std'] = [metrics['accuracy_std']]
        if metrics.get('f1_score_std') is not None:
            results['F1_Score_Std'] = [metrics['f1_score_std']]
        if metrics.get('roc_auc_std') is not None:
            results['ROC_AUC_Std'] = [metrics['roc_auc_std']]
        
        # Statistical test summary
        if metrics.get('statistical_summary') is not None:
            summary = metrics['statistical_summary']
            results['N_Significant_Comparisons'] = [summary['n_significant']]
            results['N_Better_Than_Others'] = [summary['n_better']]
        
        # Add per-class metrics
        for i, class_name in enumerate(self.class_names):
            results[f'Precision_{class_name}'] = [metrics['precision_per_class'][i]]
            results[f'Recall_{class_name}'] = [metrics['recall_per_class'][i]]
            results[f'F1_{class_name}'] = [metrics['f1_per_class'][i]]
        
        df = pd.DataFrame(results)
        
        # Save to CSV
        if filepath.endswith('.csv'):
            df.to_csv(filepath, index=False)
        elif filepath.endswith('.json'):
            df.to_json(filepath, indent=2, orient='records')
        else:
            df.to_csv(filepath + '.csv', index=False)
        
        print(f"Results saved to {filepath}")
    
    def plot_calibration_curve(
        self,
        y_true: np.ndarray,
        y_proba: np.ndarray,
        n_bins: int = 10,
        save_path: Optional[str] = None,
        figsize: Tuple[int, int] = (8, 6)
    ):
        """
        Plot calibration curve (reliability diagram).
        
        Args:
            y_true: True labels
            y_proba: Predicted probabilities
            n_bins: Number of bins for calibration
            save_path: Path to save figure (optional)
            figsize: Figure size
        """
        # Convert to numpy arrays
        if isinstance(y_true, pd.Series) or isinstance(y_true, pd.DataFrame):
            y_true = y_true.values
        y_true = np.asarray(y_true, dtype=np.int64).flatten()
        
        if isinstance(y_proba, pd.Series) or isinstance(y_proba, pd.DataFrame):
            y_proba = y_proba.values
        y_proba = np.asarray(y_proba, dtype=np.float64)
        
        # Handle 2D probability arrays
        if y_proba.ndim > 1:
            if y_proba.shape[1] == 2:
                y_proba = y_proba[:, 1]
            elif y_proba.shape[1] == 1:
                y_proba = y_proba.flatten()
            else:
                y_proba = y_proba[:, 1] if y_proba.shape[1] > 1 else y_proba[:, 0]
        else:
            y_proba = y_proba.flatten()
        
        # Ensure same length
        min_len = min(len(y_true), len(y_proba))
        y_true = y_true[:min_len]
        y_proba = y_proba[:min_len]
        
        fraction_of_positives, mean_predicted_value = calibration_curve(
            y_true, y_proba, n_bins=n_bins, strategy='uniform'
        )
        
        plt.figure(figsize=figsize)
        plt.plot(mean_predicted_value, fraction_of_positives, 's-', label='Model', linewidth=2, markersize=8)
        plt.plot([0, 1], [0, 1], 'k--', label='Perfectly Calibrated', linewidth=2)
        plt.xlabel('Mean Predicted Probability')
        plt.ylabel('Fraction of Positives')
        plt.title('Calibration Curve (Reliability Diagram)')
        plt.legend(loc="lower right")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Calibration curve saved to {save_path}")
        else:
            plt.show()
        plt.close()
    
    def plot_confusion_matrix_optimal(
        self,
        y_true: np.ndarray,
        y_proba: np.ndarray,
        save_path: Optional[str] = None,
        figsize: Tuple[int, int] = (8, 6),
        metric: str = 'f1'
    ):
        """
        Plot confusion matrix using optimal threshold.
        
        Args:
            y_true: True labels
            y_proba: Predicted probabilities
            save_path: Path to save figure (optional)
            figsize: Figure size
            metric: Metric to optimize ('f1', 'youden', 'precision_recall')
        """
        # Convert to numpy arrays
        if isinstance(y_true, pd.Series) or isinstance(y_true, pd.DataFrame):
            y_true = y_true.values
        y_true = np.asarray(y_true, dtype=np.int64).flatten()
        
        if isinstance(y_proba, pd.Series) or isinstance(y_proba, pd.DataFrame):
            y_proba = y_proba.values
        y_proba = np.asarray(y_proba, dtype=np.float64)
        
        # Handle 2D probability arrays
        if y_proba.ndim > 1:
            if y_proba.shape[1] == 2:
                y_proba = y_proba[:, 1]
            elif y_proba.shape[1] == 1:
                y_proba = y_proba.flatten()
            else:
                y_proba = y_proba[:, 1] if y_proba.shape[1] > 1 else y_proba[:, 0]
        else:
            y_proba = y_proba.flatten()
        
        # Ensure same length
        min_len = min(len(y_true), len(y_proba))
        y_true = y_true[:min_len]
        y_proba = y_proba[:min_len]
        
        # Find optimal threshold
        if metric == 'f1':
            thresholds = np.arange(0.1, 0.9, 0.01)
            best_threshold = 0.5
            best_f1 = 0
            for threshold in thresholds:
                y_pred_thresh = (y_proba >= threshold).astype(int)
                f1 = f1_score(y_true, y_pred_thresh, zero_division=0)
                if f1 > best_f1:
                    best_f1 = f1
                    best_threshold = threshold
        elif metric == 'youden':
            fpr, tpr, thresholds = roc_curve(y_true, y_proba)
            youden_index = np.argmax(tpr - fpr)
            best_threshold = thresholds[youden_index]
        else:  # precision_recall
            precision, recall, thresholds = precision_recall_curve(y_true, y_proba)
            f1_scores = 2 * (precision * recall) / (precision + recall + 1e-10)
            best_index = np.argmax(f1_scores)
            best_threshold = thresholds[best_index] if best_index < len(thresholds) else 0.5
        
        # Predict with optimal threshold
        y_pred_optimal = (y_proba >= best_threshold).astype(int)
        cm = confusion_matrix(y_true, y_pred_optimal)
        
        plt.figure(figsize=figsize)
        sns.heatmap(
            cm, 
            annot=True, 
            fmt='d', 
            cmap='Blues',
            xticklabels=self.class_names,
            yticklabels=self.class_names
        )
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.title(f'Confusion Matrix (Optimal Threshold: {best_threshold:.3f})')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Optimal confusion matrix saved to {save_path}")
        else:
            plt.show()
        plt.close()
        
        return best_threshold
    
    def plot_embeddings_visualization(
        self,
        embeddings: np.ndarray,
        labels: Optional[np.ndarray] = None,
        method: str = 'tsne',
        save_path: Optional[str] = None,
        figsize: Tuple[int, int] = (10, 8)
    ):
        """
        Visualize embeddings using t-SNE or UMAP.
        
        Args:
            embeddings: Embeddings to visualize (n_samples, n_features)
            labels: Optional labels for coloring (n_samples,)
            method: Visualization method ('tsne', 'umap', 'pca')
            save_path: Path to save figure (optional)
            figsize: Figure size
        """
        # Convert to numpy arrays
        if isinstance(embeddings, pd.DataFrame) or isinstance(embeddings, pd.Series):
            embeddings = embeddings.values
        embeddings = np.asarray(embeddings, dtype=np.float64)
        
        if labels is not None:
            if isinstance(labels, pd.Series) or isinstance(labels, pd.DataFrame):
                labels = labels.values
            labels = np.asarray(labels, dtype=np.int64).flatten()
            
            # Ensure same length
            min_len = min(len(embeddings), len(labels))
            embeddings = embeddings[:min_len]
            labels = labels[:min_len]
        if method.lower() == 'tsne':
            reducer = TSNE(n_components=2, random_state=42, perplexity=30)
            embeddings_2d = reducer.fit_transform(embeddings)
            method_name = 't-SNE'
        elif method.lower() == 'umap':
            if not HAS_UMAP:
                print("UMAP not available, using t-SNE instead")
                reducer = TSNE(n_components=2, random_state=42, perplexity=30)
                embeddings_2d = reducer.fit_transform(embeddings)
                method_name = 't-SNE (UMAP not available)'
            else:
                reducer = umap.UMAP(n_components=2, random_state=42)
                embeddings_2d = reducer.fit_transform(embeddings)
                method_name = 'UMAP'
        else:  # PCA
            reducer = PCA(n_components=2, random_state=42)
            embeddings_2d = reducer.fit_transform(embeddings)
            method_name = 'PCA'
        
        plt.figure(figsize=figsize)
        
        if labels is not None:
            unique_labels = np.unique(labels)
            colors = plt.cm.Set1(np.linspace(0, 1, len(unique_labels)))
            for i, label in enumerate(unique_labels):
                mask = labels == label
                label_name = self.class_names[int(label)] if int(label) < len(self.class_names) else f'Class {label}'
                plt.scatter(
                    embeddings_2d[mask, 0], 
                    embeddings_2d[mask, 1],
                    c=[colors[i]], 
                    label=label_name,
                    alpha=0.6,
                    s=50
                )
            plt.legend()
        else:
            plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], alpha=0.6, s=50)
        
        plt.xlabel(f'{method_name} Component 1')
        plt.ylabel(f'{method_name} Component 2')
        plt.title(f'Embeddings Visualization ({method_name})')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Embeddings visualization saved to {save_path}")
        else:
            plt.show()
        plt.close()
    
    def plot_feature_boxplots(
        self,
        X: np.ndarray,
        y: Optional[np.ndarray] = None,
        feature_names: Optional[List[str]] = None,
        n_features: int = 10,
        save_path: Optional[str] = None,
        figsize: Tuple[int, int] = (15, 8)
    ):
        """
        Plot boxplots for features.
        
        Args:
            X: Feature matrix (n_samples, n_features)
            y: Optional labels for grouping
            feature_names: Optional list of feature names
            n_features: Number of features to plot (top n by variance)
            save_path: Path to save figure (optional)
            figsize: Figure size
        """
        # Convert to numpy arrays
        if isinstance(X, pd.DataFrame) or isinstance(X, pd.Series):
            feature_names = X.columns.tolist() if feature_names is None and hasattr(X, 'columns') else feature_names
            X = X.values
        X = np.asarray(X, dtype=np.float64)
        
        if y is not None:
            if isinstance(y, pd.Series) or isinstance(y, pd.DataFrame):
                y = y.values
            y = np.asarray(y, dtype=np.int64).flatten()
            
            # Ensure same length
            min_len = min(len(X), len(y))
            X = X[:min_len]
            y = y[:min_len]
        
        # Select top features by variance
        n_features = min(n_features, X.shape[1])
        variances = np.var(X, axis=0)
        top_indices = np.argsort(variances)[-n_features:][::-1]
        
        if feature_names is None:
            feature_names = [f'Feature {i}' for i in range(X.shape[1])]
        
        # Prepare data for plotting
        plot_data = []
        plot_labels = []
        
        for idx in top_indices:
            feature_name = feature_names[idx] if idx < len(feature_names) else f'Feature {idx}'
            if y is not None:
                # Group by class
                for label in np.unique(y):
                    label_name = self.class_names[int(label)] if int(label) < len(self.class_names) else f'Class {label}'
                    values = X[y == label, idx]
                    if len(values) > 0:
                        plot_data.append(values)
                        plot_labels.append(f'{feature_name}\n({label_name})')
            else:
                plot_data.append(X[:, idx])
                plot_labels.append(feature_name)
        
        # Calculate grid size
        n_plots = len(plot_data)
        n_cols = min(5, n_plots)
        n_rows = (n_plots + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
        if n_plots == 1:
            axes = [axes]
        else:
            axes = axes.flatten() if n_plots > 1 else [axes]
        
        for i, (data, label) in enumerate(zip(plot_data, plot_labels)):
            if i < len(axes):
                axes[i].boxplot(data, vert=True)
                axes[i].set_title(label, fontsize=9)
                axes[i].grid(True, alpha=0.3)
        
        # Hide extra subplots
        for i in range(len(plot_data), len(axes)):
            axes[i].axis('off')
        
        plt.suptitle('Feature Distributions (Boxplots)', fontsize=14, y=1.02)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Feature boxplots saved to {save_path}")
        else:
            plt.show()
        plt.close()
    
    def plot_threshold_sweep(
        self,
        y_true: np.ndarray,
        y_proba: np.ndarray,
        save_path: Optional[str] = None,
        figsize: Tuple[int, int] = (12, 8)
    ):
        """
        Plot metrics across different classification thresholds.
        
        Args:
            y_true: True labels
            y_proba: Predicted probabilities
            save_path: Path to save figure (optional)
            figsize: Figure size
        """
        # Convert to numpy arrays
        if isinstance(y_true, pd.Series) or isinstance(y_true, pd.DataFrame):
            y_true = y_true.values
        y_true = np.asarray(y_true, dtype=np.int64).flatten()
        
        if isinstance(y_proba, pd.Series) or isinstance(y_proba, pd.DataFrame):
            y_proba = y_proba.values
        y_proba = np.asarray(y_proba, dtype=np.float64)
        
        # Handle 2D probability arrays
        if y_proba.ndim > 1:
            if y_proba.shape[1] == 2:
                y_proba = y_proba[:, 1]
            elif y_proba.shape[1] == 1:
                y_proba = y_proba.flatten()
            else:
                y_proba = y_proba[:, 1] if y_proba.shape[1] > 1 else y_proba[:, 0]
        else:
            y_proba = y_proba.flatten()
        
        # Ensure same length
        min_len = min(len(y_true), len(y_proba))
        y_true = y_true[:min_len]
        y_proba = y_proba[:min_len]
        
        thresholds = np.arange(0.1, 0.95, 0.01)
        accuracies = []
        precisions = []
        recalls = []
        f1_scores = []
        
        for threshold in thresholds:
            y_pred_thresh = (y_proba >= threshold).astype(int)
            accuracies.append(accuracy_score(y_true, y_pred_thresh))
            precisions.append(precision_score(y_true, y_pred_thresh, zero_division=0))
            recalls.append(recall_score(y_true, y_pred_thresh, zero_division=0))
            f1_scores.append(f1_score(y_true, y_pred_thresh, zero_division=0))
        
        fig, axes = plt.subplots(2, 2, figsize=figsize)
        
        # Accuracy
        axes[0, 0].plot(thresholds, accuracies, 'b-', linewidth=2)
        axes[0, 0].set_xlabel('Threshold')
        axes[0, 0].set_ylabel('Accuracy')
        axes[0, 0].set_title('Accuracy vs Threshold')
        axes[0, 0].grid(True, alpha=0.3)
        axes[0, 0].axvline(x=0.5, color='r', linestyle='--', alpha=0.5, label='Default (0.5)')
        axes[0, 0].legend()
        
        # Precision
        axes[0, 1].plot(thresholds, precisions, 'g-', linewidth=2)
        axes[0, 1].set_xlabel('Threshold')
        axes[0, 1].set_ylabel('Precision')
        axes[0, 1].set_title('Precision vs Threshold')
        axes[0, 1].grid(True, alpha=0.3)
        axes[0, 1].axvline(x=0.5, color='r', linestyle='--', alpha=0.5, label='Default (0.5)')
        axes[0, 1].legend()
        
        # Recall
        axes[1, 0].plot(thresholds, recalls, 'orange', linewidth=2)
        axes[1, 0].set_xlabel('Threshold')
        axes[1, 0].set_ylabel('Recall')
        axes[1, 0].set_title('Recall vs Threshold')
        axes[1, 0].grid(True, alpha=0.3)
        axes[1, 0].axvline(x=0.5, color='r', linestyle='--', alpha=0.5, label='Default (0.5)')
        axes[1, 0].legend()
        
        # F1-Score
        axes[1, 1].plot(thresholds, f1_scores, 'purple', linewidth=2)
        axes[1, 1].set_xlabel('Threshold')
        axes[1, 1].set_ylabel('F1-Score')
        axes[1, 1].set_title('F1-Score vs Threshold')
        axes[1, 1].grid(True, alpha=0.3)
        axes[1, 1].axvline(x=0.5, color='r', linestyle='--', alpha=0.5, label='Default (0.5)')
        best_f1_idx = np.argmax(f1_scores)
        best_threshold = thresholds[best_f1_idx]
        axes[1, 1].axvline(x=best_threshold, color='g', linestyle='--', alpha=0.7, label=f'Best F1 ({best_threshold:.3f})')
        axes[1, 1].legend()
        
        plt.suptitle('Threshold Sweep Analysis', fontsize=14, y=1.02)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Threshold sweep saved to {save_path}")
        else:
            plt.show()
        plt.close()
        
        return best_threshold


    def plot_fitness_convergence(
        self,
        optimization_history: Dict,
        algorithm_name: str = "Optimizer",
        save_path: Optional[str] = None,
        figsize: Tuple[int, int] = (12, 6)
    ):
        """
        Plot fitness convergence curve for metaheuristic optimization.
        
        Args:
            optimization_history: Dictionary containing 'fitness_history' and 'convergence_history'
            algorithm_name: Name of the optimization algorithm (e.g., 'EO', 'GWO', 'PSO')
            save_path: Path to save figure (optional)
            figsize: Figure size
        """
        fig, axes = plt.subplots(1, 2, figsize=figsize)
        
        # Extract history
        fitness_history = optimization_history.get('fitness_history', [])
        convergence_history = optimization_history.get('convergence_history', [])
        
        if len(fitness_history) == 0:
            print("Warning: No fitness history available for plotting.")
            plt.close()
            return
        
        iterations = range(1, len(fitness_history) + 1)
        
        # Plot 1: Best Fitness Convergence
        axes[0].plot(iterations, fitness_history, 'b-', linewidth=2, label='Best Fitness', marker='o', markersize=4)
        axes[0].set_xlabel('Iteration', fontsize=12)
        axes[0].set_ylabel('Best Fitness', fontsize=12)
        axes[0].set_title(f'{algorithm_name} - Best Fitness Convergence', fontsize=14, fontweight='bold')
        axes[0].grid(True, alpha=0.3)
        axes[0].legend()
        
        # Plot 2: Mean and Std Fitness (if available)
        if len(convergence_history) > 0:
            mean_fitness = [h.get('mean_fitness', 0) for h in convergence_history]
            std_fitness = [h.get('std_fitness', 0) for h in convergence_history]
            
            if len(mean_fitness) > 0:
                axes[1].plot(iterations, mean_fitness, 'g-', linewidth=2, label='Mean Fitness', marker='s', markersize=4)
                if any(std > 0 for std in std_fitness):
                    axes[1].fill_between(
                        iterations, 
                        [m - s for m, s in zip(mean_fitness, std_fitness)],
                        [m + s for m, s in zip(mean_fitness, std_fitness)],
                        alpha=0.3, color='green', label='±1 Std'
                    )
                axes[1].plot(iterations, fitness_history, 'b-', linewidth=2, label='Best Fitness', marker='o', markersize=4)
                axes[1].set_xlabel('Iteration', fontsize=12)
                axes[1].set_ylabel('Fitness', fontsize=12)
                axes[1].set_title(f'{algorithm_name} - Population Fitness Statistics', fontsize=14, fontweight='bold')
                axes[1].grid(True, alpha=0.3)
                axes[1].legend()
            else:
                axes[1].axis('off')
        else:
            axes[1].axis('off')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Fitness convergence curve saved to {save_path}")
        else:
            plt.show()
        plt.close()
    
    def plot_feature_selection_distribution(
        self,
        feature_scores: pd.DataFrame,
        selected_features: Optional[List[str]] = None,
        save_path: Optional[str] = None,
        figsize: Tuple[int, int] = (14, 8)
    ):
        """
        Plot feature selection distribution showing Mutual Information scores.
        
        Args:
            feature_scores: DataFrame with columns 'Feature' and 'MI_Score'
            selected_features: List of selected feature names (optional)
            save_path: Path to save figure (optional)
            figsize: Figure size
        """
        # Sort by MI score
        feature_scores_sorted = feature_scores.sort_values('MI_Score', ascending=True)
        
        fig, axes = plt.subplots(2, 1, figsize=figsize)
        
        # Plot 1: Horizontal bar chart
        colors = ['green' if (selected_features is None or f in selected_features) else 'red' 
                  for f in feature_scores_sorted['Feature']]
        
        axes[0].barh(range(len(feature_scores_sorted)), feature_scores_sorted['MI_Score'], color=colors, alpha=0.7)
        axes[0].set_yticks(range(len(feature_scores_sorted)))
        axes[0].set_yticklabels(feature_scores_sorted['Feature'], fontsize=9)
        axes[0].set_xlabel('Mutual Information Score', fontsize=12)
        axes[0].set_title('Feature Selection Distribution (Mutual Information)', fontsize=14, fontweight='bold')
        axes[0].grid(True, alpha=0.3, axis='x')
        axes[0].axvline(x=0, color='black', linestyle='--', linewidth=1, alpha=0.5)
        
        # Add legend
        legend_elements = [
            Patch(facecolor='green', alpha=0.7, label='Selected'),
            Patch(facecolor='red', alpha=0.7, label='Removed')
        ]
        axes[0].legend(handles=legend_elements, loc='lower right')
        
        # Plot 2: Distribution histogram
        selected_scores = feature_scores_sorted[feature_scores_sorted['Feature'].isin(selected_features)]['MI_Score'] if selected_features else feature_scores_sorted['MI_Score']
        removed_scores = feature_scores_sorted[~feature_scores_sorted['Feature'].isin(selected_features)]['MI_Score'] if selected_features else pd.Series()
        
        axes[1].hist(selected_scores, bins=20, alpha=0.7, color='green', label='Selected Features', edgecolor='black')
        if len(removed_scores) > 0:
            axes[1].hist(removed_scores, bins=20, alpha=0.7, color='red', label='Removed Features', edgecolor='black')
        axes[1].set_xlabel('Mutual Information Score', fontsize=12)
        axes[1].set_ylabel('Number of Features', fontsize=12)
        axes[1].set_title('MI Score Distribution', fontsize=14, fontweight='bold')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Feature selection distribution saved to {save_path}")
        else:
            plt.show()
        plt.close()
    
    def plot_variance_analysis(
        self,
        X: np.ndarray,
        feature_names: Optional[List[str]] = None,
        y: Optional[np.ndarray] = None,
        save_path: Optional[str] = None,
        figsize: Tuple[int, int] = (14, 8)
    ):
        """
        Plot variance analysis showing feature variances with boxplots.
        
        Args:
            X: Feature matrix (n_samples, n_features)
            feature_names: Optional list of feature names
            y: Optional labels for grouping
            save_path: Path to save figure (optional)
            figsize: Figure size
        """
        # Convert to numpy arrays
        if isinstance(X, pd.DataFrame) or isinstance(X, pd.Series):
            feature_names = X.columns.tolist() if feature_names is None and hasattr(X, 'columns') else feature_names
            X = X.values
        X = np.asarray(X, dtype=np.float64)
        
        # Calculate variances
        variances = np.var(X, axis=0)
        
        # Sort by variance
        sorted_indices = np.argsort(variances)[::-1]
        sorted_variances = variances[sorted_indices]
        
        if feature_names is None:
            feature_names = [f'Feature {i}' for i in range(X.shape[1])]
        
        sorted_feature_names = [feature_names[i] for i in sorted_indices]
        
        fig, axes = plt.subplots(2, 1, figsize=figsize)
        
        # Plot 1: Variance bar chart
        colors = plt.cm.viridis(np.linspace(0, 1, len(sorted_variances)))
        axes[0].bar(range(len(sorted_variances)), sorted_variances, color=colors, alpha=0.7, edgecolor='black')
        axes[0].set_xticks(range(len(sorted_variances)))
        axes[0].set_xticklabels(sorted_feature_names, rotation=45, ha='right', fontsize=9)
        axes[0].set_ylabel('Variance', fontsize=12)
        axes[0].set_title('Feature Variance Analysis', fontsize=14, fontweight='bold')
        axes[0].grid(True, alpha=0.3, axis='y')
        
        # Plot 2: Boxplots for top features by variance
        n_top = min(10, len(sorted_indices))
        top_indices = sorted_indices[:n_top]
        
        plot_data = []
        plot_labels = []
        
        for idx in top_indices:
            feature_name = feature_names[idx] if idx < len(feature_names) else f'Feature {idx}'
            if y is not None:
                # Group by class
                for label in np.unique(y):
                    label_name = self.class_names[int(label)] if int(label) < len(self.class_names) else f'Class {label}'
                    values = X[y == label, idx]
                    if len(values) > 0:
                        plot_data.append(values)
                        plot_labels.append(f'{feature_name}\n({label_name})')
            else:
                plot_data.append(X[:, idx])
                plot_labels.append(f'{feature_name}\n(Var={variances[idx]:.4f})')
        
        # Calculate grid size for boxplots
        n_plots = len(plot_data)
        n_cols = min(5, n_plots)
        n_rows = (n_plots + n_cols - 1) // n_cols
        
        # Create subplot for boxplots
        fig2, axes2 = plt.subplots(n_rows, n_cols, figsize=(15, 3 * n_rows))
        if n_plots == 1:
            axes2 = [axes2]
        else:
            axes2 = axes2.flatten() if n_plots > 1 else [axes2]
        
        for i, (data, label) in enumerate(zip(plot_data, plot_labels)):
            if i < len(axes2):
                axes2[i].boxplot(data, vert=True)
                axes2[i].set_title(label, fontsize=9)
                axes2[i].grid(True, alpha=0.3)
        
        # Hide extra subplots
        for i in range(len(plot_data), len(axes2)):
            axes2[i].axis('off')
        
        plt.suptitle(f'Top {n_top} Features by Variance (Boxplots)', fontsize=14, fontweight='bold', y=1.02)
        plt.tight_layout()
        
        # Combine both figures
        # Save variance bar chart
        if save_path:
            # Save variance analysis (bar chart)
            variance_path = save_path.replace('.png', '_variance.png')
            plt.figure(fig.number)
            plt.tight_layout()
            plt.savefig(variance_path, dpi=300, bbox_inches='tight')
            print(f"Variance analysis saved to {variance_path}")
            
            # Save boxplots
            boxplot_path = save_path.replace('.png', '_boxplots.png')
            plt.figure(fig2.number)
            plt.tight_layout()
            plt.savefig(boxplot_path, dpi=300, bbox_inches='tight')
            print(f"Variance boxplots saved to {boxplot_path}")
        else:
            plt.show()
        
        plt.close(fig)
        plt.close(fig2)
    
    def plot_performance_variance(
        self,
        algorithm_results: Dict[str, List[float]],
        metric_name: str = 'Accuracy',
        save_path: Optional[str] = None,
        figsize: Tuple[int, int] = (10, 6)
    ):
        """
        Plot boxplots showing performance variance across multiple runs for each algorithm.
        
        This shows how consistent each algorithm is across multiple optimization runs.
        Requires multiple runs per algorithm (at least 2, ideally 5+).
        
        Args:
            algorithm_results: Dictionary of {algorithm_name: [run1_value, run2_value, ...]}
                             Example: {'EO': [98.04, 97.50], 'GWO': [94.12, 95.20], ...}
            metric_name: Name of the metric being plotted (e.g., 'Accuracy', 'F1-Score')
            save_path: Path to save figure (optional)
            figsize: Figure size
        """
        if not algorithm_results:
            print("Warning: No algorithm results provided for variance analysis")
            return
        
        # Prepare data for boxplot
        algorithms = list(algorithm_results.keys())
        data = [algorithm_results[alg] for alg in algorithms]
        
        # Filter out algorithms with insufficient runs (need at least 2)
        valid_data = []
        valid_algorithms = []
        for alg, values in zip(algorithms, data):
            if len(values) >= 2:
                valid_data.append(values)
                valid_algorithms.append(alg)
            else:
                print(f"Warning: {alg} has only {len(values)} run(s). Need at least 2 runs for variance boxplot. Skipping.")
        
        if not valid_data:
            print("Error: No algorithms have sufficient runs (>=2) for variance boxplot")
            return
        
        # Create boxplot
        fig, ax = plt.subplots(figsize=figsize)
        
        bp = ax.boxplot(valid_data, labels=valid_algorithms, patch_artist=True, 
                       showmeans=True, meanline=True)
        
        # Color the boxes
        colors = plt.cm.Set2(np.linspace(0, 1, len(valid_data)))
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        
        # Customize
        ax.set_ylabel(f'{metric_name} (%)', fontsize=12)
        ax.set_xlabel('Algorithm', fontsize=12)
        ax.set_title(f'Performance Variance Across Multiple Runs - {metric_name}', 
                    fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')
        
        # Add statistics text
        stats_text = []
        for alg, values in zip(valid_algorithms, valid_data):
            mean_val = np.mean(values)
            std_val = np.std(values, ddof=1) if len(values) > 1 else 0
            stats_text.append(f'{alg}: {mean_val:.2f}% ± {std_val:.2f}% (n={len(values)})')
        
        # Add text box with statistics
        stats_str = '\n'.join(stats_text)
        ax.text(0.02, 0.98, stats_str, transform=ax.transAxes, 
               fontsize=9, verticalalignment='top',
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Performance variance boxplot saved to {save_path}")
        else:
            plt.show()
        
        plt.close()


def statistical_comparison(
    cv_results: Dict[str, List[float]],
    baseline_algorithm: str = 'EO',
    metric: str = 'accuracy',
    alpha: float = 0.05,
    test_type: str = 'wilcoxon',
    save_path: Optional[str] = None
) -> pd.DataFrame:
    """
    Perform statistical significance tests comparing baseline algorithm (EO) with others.
    
    Uses paired t-test or Wilcoxon signed-rank test on cross-validation fold results.
    
    Args:
        cv_results: Dictionary of {algorithm_name: [fold1_value, fold2_value, ...]}
                    Example: {'EO': [1.0, 1.0, 0.9574, ...], 'GWO': [1.0, 1.0, 0.9574, ...]}
        baseline_algorithm: Name of baseline algorithm (default: 'EO')
        metric: Metric name for display (default: 'accuracy')
        alpha: Significance level (default: 0.05)
        test_type: 'wilcoxon' or 'ttest' (default: 'wilcoxon')
        save_path: Path to save results (optional)
        
    Returns:
        DataFrame with statistical test results
    """
    if baseline_algorithm not in cv_results:
        raise ValueError(f"Baseline algorithm '{baseline_algorithm}' not found in results")
    
    baseline_values = np.array(cv_results[baseline_algorithm])
    other_algorithms = [alg for alg in cv_results.keys() if alg != baseline_algorithm]
    
    results = []
    
    for other_alg in other_algorithms:
        other_values = np.array(cv_results[other_alg])
        
        if len(baseline_values) != len(other_values):
            print(f"Warning: {baseline_algorithm} and {other_alg} have different number of folds. Skipping.")
            continue
        
        # Calculate differences (baseline - other)
        differences = baseline_values - other_values
        mean_diff = np.mean(differences)
        std_diff = np.std(differences, ddof=1) if len(differences) > 1 else 0
        
        # Perform statistical test
        if test_type.lower() == 'wilcoxon':
            # Wilcoxon signed-rank test (non-parametric, better for small samples)
            try:
                statistic, p_value = stats.wilcoxon(baseline_values, other_values, alternative='two-sided')
                test_name = 'Wilcoxon Signed-Rank'
            except Exception as e:
                print(f"Warning: Wilcoxon test failed for {other_alg}: {e}")
                # Fallback to t-test
                statistic, p_value = stats.ttest_rel(baseline_values, other_values)
                test_name = 'Paired t-test (fallback)'
        else:
            # Paired t-test (parametric)
            statistic, p_value = stats.ttest_rel(baseline_values, other_values)
            test_name = 'Paired t-test'
        
        # Determine significance
        is_significant = p_value < alpha
        
        # Determine direction (is baseline better?)
        baseline_mean = np.mean(baseline_values)
        other_mean = np.mean(other_values)
        baseline_better = baseline_mean > other_mean
        
        # Conclusion
        if is_significant and baseline_better:
            conclusion = f"{baseline_algorithm} significantly better"
        elif is_significant and not baseline_better:
            conclusion = f"{other_alg} significantly better"
        else:
            conclusion = "No significant difference"
        
        results.append({
            'Comparison': f'{baseline_algorithm} vs {other_alg}',
            'Baseline_Mean': f'{baseline_mean:.4f}',
            'Other_Mean': f'{other_mean:.4f}',
            'Mean_Difference': f'{mean_diff:.4f}',
            'Std_Difference': f'{std_diff:.4f}',
            'Test_Type': test_name,
            'Test_Statistic': f'{statistic:.4f}',
            'P_Value': f'{p_value:.6f}',
            'Significant': 'Yes' if is_significant else 'No',
            'Conclusion': conclusion
        })
    
    df = pd.DataFrame(results)
    
    # Print summary
    print("\n" + "=" * 80)
    print(f"STATISTICAL COMPARISON: {baseline_algorithm.upper()} vs Other Algorithms")
    print("=" * 80)
    print(f"Metric: {metric.upper()}")
    print(f"Test Type: {test_type.upper()}")
    print(f"Significance Level: α = {alpha}")
    print("\nResults:")
    print("-" * 80)
    print(df.to_string(index=False))
    print("-" * 80)
    
    # Summary statistics
    n_significant = sum(df['Significant'] == 'Yes')
    n_better = sum('significantly better' in c and baseline_algorithm in c for c in df['Conclusion'])
    
    print(f"\nSummary:")
    print(f"  - {baseline_algorithm} is significantly better than {n_better} algorithm(s)")
    print(f"  - {n_significant} out of {len(df)} comparisons are statistically significant (p < {alpha})")
    
    if save_path:
        df.to_csv(save_path, index=False)
        print(f"\nResults saved to: {save_path}")
    
    return df


def compare_models(
    results: Dict[str, Dict],
    save_path: Optional[str] = None
) -> pd.DataFrame:
    """
    Compare multiple models and create comparison table.
    
    Args:
        results: Dictionary of {model_name: metrics_dict}
        save_path: Path to save comparison table
        
    Returns:
        DataFrame with comparison results
    """
    comparison_data = []
    
    for model_name, metrics in results.items():
        row = {
            'Model': model_name,
            'Accuracy': metrics['accuracy'],
            'Precision': metrics['precision'],
            'Recall': metrics['recall'],
            'F1_Score': metrics['f1_score'],
        }
        
        if metrics.get('roc_auc') is not None:
            row['ROC_AUC'] = metrics['roc_auc']
        if metrics.get('pr_auc') is not None:
            row['PR_AUC'] = metrics['pr_auc']
        
        comparison_data.append(row)
    
    df = pd.DataFrame(comparison_data)
    df = df.sort_values('F1_Score', ascending=False)
    
    if save_path:
        df.to_csv(save_path, index=False)
        print(f"Comparison table saved to {save_path}")
    
    return df


if __name__ == "__main__":
    print("Comprehensive Evaluation Module for CKD Prediction")
    print("This module provides all metrics required for Q1/Q2 paper publication.")
