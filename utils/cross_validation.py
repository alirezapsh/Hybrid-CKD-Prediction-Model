import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from typing import Dict, List, Tuple, Optional, Any
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, confusion_matrix
)
import warnings
warnings.filterwarnings('ignore')

from models.sscl import train_sscl, SSCLTrainer
from models.transformer import train_transformer, TransformerTrainer, TransformerDataset


class CrossValidator:
    """
    Cross-validator for SSCL + Transformer pipeline.
    Performs k-fold cross-validation on training data.
    """
    
    def __init__(
        self,
        cv: StratifiedKFold,
        sscl_epochs: int = 50,
        transformer_epochs: int = 50,
        batch_size: int = 32,
        verbose: bool = True,
        device: Optional[torch.device] = None
    ):
        """
        Initialize cross-validator.
        
        Args:
            cv: StratifiedKFold object from preprocessing
            sscl_epochs: Number of epochs for SSCL training
            transformer_epochs: Number of epochs for Transformer training
            batch_size: Batch size for training
            verbose: Whether to print progress
            device: PyTorch device
        """
        self.cv = cv
        self.sscl_epochs = sscl_epochs
        self.transformer_epochs = transformer_epochs
        self.batch_size = batch_size
        self.verbose = verbose
        
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = device
        
        self.fold_results = []
        self.fold_histories = []
    
    def cross_validate(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None
    ) -> Dict[str, Any]:
        """
        Perform k-fold cross-validation on training data.
        
        Args:
            X_train: Training features (70% of data)
            y_train: Training labels (70% of data)
            X_val: Optional validation features (15% of data) - not used in CV
            y_val: Optional validation labels (15% of data) - not used in CV
            
        Returns:
            Dictionary containing:
            - fold_results: List of results for each fold (evaluated on internal validation)
            - aggregated_metrics: Mean ± std across folds (from internal validation)
            - all_predictions: Predictions from all folds (on internal validation)
            - all_probabilities: Probabilities from all folds (on internal validation)
        """
        # Convert to numpy if needed
        if isinstance(X_train, pd.DataFrame):
            X_train = X_train.values
        if isinstance(y_train, pd.Series):
            y_train = y_train.values
        
        X_train = np.asarray(X_train, dtype=np.float32)
        y_train = np.asarray(y_train, dtype=np.int64)
        
        n_folds = self.cv.n_splits
        fold_results = []
        fold_histories = []
        
        if self.verbose:
            print("\n" + "=" * 80)
            print(f"CROSS-VALIDATION: {n_folds}-FOLD STRATIFIED CV")
            print("=" * 80)
            print(f"Training set size: {len(X_train)} (70%)")
            print(f"Device: {self.device}")
            if X_val is not None:
                print(f"Validation set size: {len(X_val)} (15%) - used AFTER CV for hyperparameter tuning")
            print("\nNote: Each fold trains on different subset of training data (80%),")
            print("      uses internal validation (20%) for early stopping AND evaluation.")
            print("      CV results = mean ± std of internal validation evaluations (training-phase performance).")
            print("      Test set (15%) is NOT used in CV - reserved for final evaluation only.")
            print()
        
        # Perform cross-validation
        for fold_idx, (train_idx, val_idx) in enumerate(self.cv.split(X_train, y_train), 1):
            if self.verbose:
                print("\n" + "-" * 80)
                print(f"FOLD {fold_idx}/{n_folds}")
                print("-" * 80)
            
            # Split data for this fold
            X_fold_train = X_train[train_idx]  # 188 samples (80% of 70%)
            y_fold_train = y_train[train_idx]
            X_fold_internal_val = X_train[val_idx]  # 47 samples (20% of 70%) - for early stopping AND evaluation
            y_fold_internal_val = y_train[val_idx]
            
            if self.verbose:
                print(f"  Fold train size: {len(X_fold_train)} (80% of training set)")
                print(f"  Fold internal validation size: {len(X_fold_internal_val)} (20% of training set)")
                print(f"    → Used for: early stopping during training AND final evaluation")
                print(f"  Train class distribution: {np.bincount(y_fold_train)}")
                print(f"  Internal val class distribution: {np.bincount(y_fold_internal_val)}")
            
            # Step 1: Train SSCL on fold training data
            if self.verbose:
                print(f"\n  [Fold {fold_idx}] Training SSCL...")
            
            sscl_model, sscl_trainer, sscl_history = train_sscl(
                X_train=X_fold_train,
                X_val=X_fold_internal_val,  # Use internal validation for early stopping
                input_dim=X_fold_train.shape[1],
                epochs=self.sscl_epochs,
                batch_size=self.batch_size,
                verbose=False,  # Reduce verbosity for CV
                device=self.device
            )
            
            # Step 2: Extract SSCL features
            if self.verbose:
                print(f"  [Fold {fold_idx}] Extracting SSCL features...")
            
            X_fold_train_sscl = sscl_trainer.extract_features(
                X_fold_train,
                use_projection=False,
                batch_size=self.batch_size
            )
            # Extract features for internal validation (used for early stopping AND evaluation)
            X_fold_internal_val_sscl = sscl_trainer.extract_features(
                X_fold_internal_val,
                use_projection=False,
                batch_size=self.batch_size
            )
            
            # Step 3: Train Transformer on SSCL features
            if self.verbose:
                print(f"  [Fold {fold_idx}] Training Transformer...")
            
            transformer_model, transformer_trainer, transformer_history = train_transformer(
                X_train=X_fold_train_sscl,
                y_train=y_fold_train,
                X_val=X_fold_internal_val_sscl,  # Use internal validation for early stopping
                y_val=y_fold_internal_val,
                input_dim=X_fold_train_sscl.shape[1],
                epochs=self.transformer_epochs,
                batch_size=self.batch_size,
                verbose=False,  # Reduce verbosity for CV
                device=self.device
            )
            
            # Step 4: Evaluate on INTERNAL VALIDATION (47 samples) - this is where CV results come from
            if self.verbose:
                print(f"  [Fold {fold_idx}] Evaluating on INTERNAL VALIDATION (20% of training set)...")
            
            fold_metrics = self._evaluate_fold(
                transformer_trainer,
                X_fold_internal_val_sscl,  # Evaluate on INTERNAL VALIDATION
                y_fold_internal_val  # Internal validation labels
            )
            
            fold_results.append({
                'fold': fold_idx,
                'metrics': fold_metrics,
                'train_size': len(X_fold_train),
                'internal_val_size': len(X_fold_internal_val)
            })
            fold_histories.append({
                'fold': fold_idx,
                'sscl_history': sscl_history,
                'transformer_history': transformer_history
            })
            
            if self.verbose:
                print(f"  [Fold {fold_idx}] Results:")
                print(f"    Accuracy:  {fold_metrics['accuracy']:.4f}")
                print(f"    Precision: {fold_metrics['precision']:.4f}")
                print(f"    Recall:    {fold_metrics['recall']:.4f}")
                print(f"    F1-Score:  {fold_metrics['f1_score']:.4f}")
                if fold_metrics['roc_auc'] is not None:
                    print(f"    ROC-AUC:   {fold_metrics['roc_auc']:.4f}")
                if fold_metrics['pr_auc'] is not None:
                    print(f"    PR-AUC:    {fold_metrics['pr_auc']:.4f}")
            
            # Clean up to save memory
            del sscl_model, sscl_trainer, transformer_model, transformer_trainer
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
        
        # Aggregate results across folds
        aggregated_metrics = self._aggregate_results(fold_results)
        
        self.fold_results = fold_results
        self.fold_histories = fold_histories
        
        if self.verbose:
            self._print_aggregated_results(aggregated_metrics)
        
        return {
            'fold_results': fold_results,
            'aggregated_metrics': aggregated_metrics,
            'fold_histories': fold_histories
        }
    
    def _evaluate_fold(
        self,
        trainer: TransformerTrainer,
        X_val: np.ndarray,
        y_val: np.ndarray
    ) -> Dict[str, float]:
        """
        Evaluate model on fold validation set.
        
        Args:
            trainer: Trained TransformerTrainer
            X_val: Validation features
            y_val: Validation labels
            
        Returns:
            Dictionary of metrics
        """
        # Get predictions
        y_pred = trainer.predict(X_val)
        
        # Get probabilities
        trainer.model.eval()
        val_dataset = TransformerDataset(X_val)
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=0
        )
        
        y_proba = []
        with torch.no_grad():
            for batch in val_loader:
                x = batch['x'].to(self.device)
                outputs = trainer.model(x)
                probs = torch.softmax(outputs, dim=1)
                y_proba.append(probs.cpu().numpy())
        
        y_proba = np.vstack(y_proba)
        
        # Calculate metrics
        accuracy = accuracy_score(y_val, y_pred)
        precision = precision_score(y_val, y_pred, average='weighted', zero_division=0)
        recall = recall_score(y_val, y_pred, average='weighted', zero_division=0)
        f1 = f1_score(y_val, y_pred, average='weighted', zero_division=0)
        
        # ROC-AUC (binary classification)
        roc_auc = None
        if len(np.unique(y_val)) == 2:
            try:
                roc_auc = roc_auc_score(y_val, y_proba[:, 1])
            except:
                roc_auc = None
        
        # PR-AUC
        pr_auc = None
        if len(np.unique(y_val)) == 2:
            try:
                pr_auc = average_precision_score(y_val, y_proba[:, 1])
            except:
                pr_auc = None
        
        # Confusion matrix
        cm = confusion_matrix(y_val, y_pred)
        
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'roc_auc': roc_auc,
            'pr_auc': pr_auc,
            'confusion_matrix': cm
        }
    
    def _aggregate_results(
        self,
        fold_results: List[Dict[str, Any]]
    ) -> Dict[str, Dict[str, float]]:
        """
        Aggregate results across folds.
        
        Args:
            fold_results: List of results for each fold
            
        Returns:
            Dictionary with mean ± std for each metric
        """
        metrics = ['accuracy', 'precision', 'recall', 'f1_score', 'roc_auc', 'pr_auc']
        aggregated = {}
        
        for metric in metrics:
            values = []
            for fold_result in fold_results:
                value = fold_result['metrics'][metric]
                if value is not None:
                    values.append(value)
            
            if len(values) > 0:
                aggregated[metric] = {
                    'mean': np.mean(values),
                    'std': np.std(values),
                    'min': np.min(values),
                    'max': np.max(values),
                    'values': values
                }
            else:
                aggregated[metric] = {
                    'mean': None,
                    'std': None,
                    'min': None,
                    'max': None,
                    'values': []
                }
        
        # Aggregate confusion matrices
        confusion_matrices = []
        for fold_result in fold_results:
            cm = fold_result['metrics'].get('confusion_matrix')
            if cm is not None:
                confusion_matrices.append(cm)
        
        if len(confusion_matrices) > 0:
            # Average confusion matrix across folds
            aggregated['confusion_matrix'] = {
                'mean': np.mean(confusion_matrices, axis=0),
                'std': np.std(confusion_matrices, axis=0),
                'individual': confusion_matrices
            }
        
        return aggregated
    
    def _print_aggregated_results(
        self,
        aggregated_metrics: Dict[str, Dict[str, float]]
    ):
        """
        Print aggregated cross-validation results.
        
        Args:
            aggregated_metrics: Aggregated metrics dictionary
        """
        print("\n" + "=" * 80)
        print("CROSS-VALIDATION RESULTS (Mean ± Std across folds)")
        print("=" * 80)
        
        for metric_name, metric_dict in aggregated_metrics.items():
            if metric_name == 'confusion_matrix':
                continue  # Handle separately
            if metric_dict['mean'] is not None:
                mean = metric_dict['mean']
                std = metric_dict['std']
                min_val = metric_dict['min']
                max_val = metric_dict['max']
                
                print(f"\n{metric_name.upper().replace('_', '-')}:")
                print(f"  Mean ± Std: {mean:.4f} ± {std:.4f}")
                print(f"  Range: [{min_val:.4f}, {max_val:.4f}]")
                print(f"  Individual fold values: {[f'{v:.4f}' for v in metric_dict['values']]}")
        
        # Print confusion matrix
        if 'confusion_matrix' in aggregated_metrics:
            cm_mean = aggregated_metrics['confusion_matrix']['mean']
            cm_std = aggregated_metrics['confusion_matrix']['std']
            print("\nCONFUSION MATRIX (Mean across folds):")
            print("  " + str(cm_mean.astype(int)).replace('\n', '\n  '))
            print("\nCONFUSION MATRIX (Std across folds):")
            print("  " + str(cm_std.astype(int)).replace('\n', '\n  '))
            print("\nIndividual fold confusion matrices:")
            for idx, cm in enumerate(aggregated_metrics['confusion_matrix']['individual'], 1):
                print(f"  Fold {idx}:")
                print("    " + str(cm.astype(int)).replace('\n', '\n    '))
        
        print("\n" + "=" * 80)


def perform_cross_validation(
    X_train: np.ndarray,
    y_train: np.ndarray,
    cv: StratifiedKFold,
    X_val: Optional[np.ndarray] = None,
    y_val: Optional[np.ndarray] = None,
    sscl_epochs: int = 50,
    transformer_epochs: int = 50,
    batch_size: int = 32,
    verbose: bool = True,
    device: Optional[torch.device] = None
) -> Dict[str, Any]:
    """
    Convenience function to perform cross-validation.
    
    Args:
        X_train: Training features (70% of data)
        y_train: Training labels (70% of data)
        cv: StratifiedKFold object
        X_val: Optional validation features (15% of data) - not used in CV, used AFTER for hyperparameter tuning
        y_val: Optional validation labels (15% of data) - not used in CV, used AFTER for hyperparameter tuning
        sscl_epochs: Number of epochs for SSCL
        transformer_epochs: Number of epochs for Transformer
        batch_size: Batch size
        verbose: Whether to print progress
        device: PyTorch device
        
    Returns:
        Dictionary with cross-validation results (evaluated on internal validation sets)
        Note: Test set (15%) is NOT used in CV - reserved for final evaluation only
    """
    validator = CrossValidator(
        cv=cv,
        sscl_epochs=sscl_epochs,
        transformer_epochs=transformer_epochs,
        batch_size=batch_size,
        verbose=verbose,
        device=device
    )
    
    return validator.cross_validate(X_train, y_train, X_val=X_val, y_val=y_val)

