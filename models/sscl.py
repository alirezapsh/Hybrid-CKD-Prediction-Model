import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from typing import Tuple, Optional, Dict, List
import warnings
warnings.filterwarnings('ignore')


class TabularAugmentation:
    """
    Data augmentation strategies for medical tabular data.
    Handles incomplete/complex data by creating augmented views.
    """
    
    def __init__(
        self,
        noise_std: float = 0.1,
        feature_dropout: float = 0.1,
        mixup_alpha: float = 0.2,
        cutout_prob: float = 0.1
    ):
        """
        Initialize augmentation parameters.
        
        Args:
            noise_std: Standard deviation for Gaussian noise (default: 0.1)
            feature_dropout: Probability of dropping a feature (default: 0.1)
            mixup_alpha: Alpha parameter for mixup augmentation (default: 0.2)
            cutout_prob: Probability of zeroing out a feature (default: 0.1)
        """
        self.noise_std = noise_std
        self.feature_dropout_prob = feature_dropout  # Renamed to avoid conflict with method
        self.mixup_alpha = mixup_alpha
        self.cutout_prob = cutout_prob
    
    def add_gaussian_noise(self, x: np.ndarray) -> np.ndarray:
        """
        Add Gaussian noise to features.
        
        Args:
            x: Input features (n_samples, n_features)
            
        Returns:
            Augmented features with Gaussian noise
        """
        noise = np.random.normal(0, self.noise_std, size=x.shape)
        return x + noise
    
    def feature_dropout(self, x: np.ndarray) -> np.ndarray:
        """
        Randomly drop features (set to zero).
        
        Args:
            x: Input features (n_samples, n_features)
            
        Returns:
            Augmented features with dropped features
        """
        x_aug = x.copy()
        mask = np.random.binomial(1, self.feature_dropout_prob, size=x.shape)
        x_aug[mask == 1] = 0
        return x_aug
    
    def cutout(self, x: np.ndarray) -> np.ndarray:
        """
        Randomly zero out features (cutout).
        
        Args:
            x: Input features (n_samples, n_features)
            
        Returns:
            Augmented features with cutout
        """
        x_aug = x.copy()
        mask = np.random.binomial(1, self.cutout_prob, size=x.shape)
        x_aug[mask == 1] = 0
        return x_aug
    
    def mixup(self, x: np.ndarray, y: Optional[np.ndarray] = None) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Mixup augmentation: mix two samples.
        
        Args:
            x: Input features (n_samples, n_features)
            y: Optional labels (n_samples,)
            
        Returns:
            Mixed features and labels
        """
        if len(x) < 2:
            return x, y
        
        # Random permutation
        indices = np.random.permutation(len(x))
        x_shuffled = x[indices]
        
        # Sample lambda from Beta distribution
        lam = np.random.beta(self.mixup_alpha, self.mixup_alpha)
        
        # Mix samples
        x_mixed = lam * x + (1 - lam) * x_shuffled
        
        if y is not None:
            y_shuffled = y[indices]
            y_mixed = lam * y + (1 - lam) * y_shuffled
            return x_mixed, y_mixed
        
        return x_mixed, None
    
    def augment(self, x: np.ndarray, method: str = 'combined') -> np.ndarray:
        """
        Apply augmentation method.
        
        Args:
            x: Input features (n_samples, n_features)
            method: Augmentation method ('noise', 'dropout', 'cutout', 'mixup', 'combined')
            
        Returns:
            Augmented features
        """
        if method == 'noise':
            return self.add_gaussian_noise(x)
        elif method == 'dropout':
            return self.feature_dropout(x)
        elif method == 'cutout':
            return self.cutout(x)
        elif method == 'mixup':
            x_aug, _ = self.mixup(x)
            return x_aug
        elif method == 'combined':
            # Randomly select one augmentation method
            methods = ['noise', 'dropout']
            selected = np.random.choice(methods)
            return self.augment(x, method=selected)
        else:
            raise ValueError(f"Unknown augmentation method: {method}")


class Encoder(nn.Module):
    """
    Encoder network for learning rich semantic representations.
    Multi-layer perceptron with residual connections.
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dims: List[int] = [128, 256, 128],
        dropout: float = 0.2,
        use_batch_norm: bool = True,
        activation: str = 'relu'
    ):
        """
        Initialize encoder.
        
        Args:
            input_dim: Input feature dimension
            hidden_dims: List of hidden layer dimensions
            dropout: Dropout probability
            use_batch_norm: Whether to use batch normalization
            activation: Activation function ('relu', 'gelu', 'tanh')
        """
        super(Encoder, self).__init__()
        
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.dropout = dropout
        self.use_batch_norm = use_batch_norm
        
        # Activation function
        if activation == 'relu':
            self.activation = nn.ReLU()
        elif activation == 'gelu':
            self.activation = nn.GELU()
        elif activation == 'tanh':
            self.activation = nn.Tanh()
        else:
            raise ValueError(f"Unknown activation: {activation}")
        
        # Build layers
        layers = []
        prev_dim = input_dim
        
        for i, hidden_dim in enumerate(hidden_dims):
            layers.append(nn.Linear(prev_dim, hidden_dim))
            
            if use_batch_norm:
                layers.append(nn.BatchNorm1d(hidden_dim))
            
            layers.append(self.activation)
            
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            
            prev_dim = hidden_dim
        
        self.encoder = nn.Sequential(*layers)
        self.output_dim = hidden_dims[-1]
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through encoder.
        
        Args:
            x: Input tensor (batch_size, input_dim)
            
        Returns:
            Encoded representation (batch_size, output_dim)
        """
        return self.encoder(x)


class ProjectionHead(nn.Module):
    """
    Projection head for contrastive learning.
    Maps encoder output to a lower-dimensional space for contrastive loss.
    """
    
    def __init__(
        self,
        input_dim: int,
        projection_dim: int = 64,
        hidden_dim: int = 128,
        dropout: float = 0.1
    ):
        """
        Initialize projection head.
        
        Args:
            input_dim: Input dimension (encoder output dimension)
            projection_dim: Output projection dimension
            hidden_dim: Hidden layer dimension
            dropout: Dropout probability
        """
        super(ProjectionHead, self).__init__()
        
        self.projection = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, projection_dim),
            nn.BatchNorm1d(projection_dim, affine=False)  # No affine transformation for final layer
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through projection head.
        
        Args:
            x: Input tensor (batch_size, input_dim)
            
        Returns:
            Projected representation (batch_size, projection_dim)
        """
        return self.projection(x)


class SSCLModel(nn.Module):
    """
    Complete SSCL model: Encoder + Projection Head.
    """
    
    def __init__(
        self,
        input_dim: int,
        encoder_hidden_dims: List[int] = [128, 256, 128],
        projection_dim: int = 64,
        projection_hidden_dim: int = 128,
        encoder_dropout: float = 0.2,
        projection_dropout: float = 0.1,
        use_batch_norm: bool = True,
        activation: str = 'relu'
    ):
        """
        Initialize SSCL model.
        
        Args:
            input_dim: Input feature dimension
            encoder_hidden_dims: Encoder hidden layer dimensions
            projection_dim: Projection output dimension
            projection_hidden_dim: Projection hidden layer dimension
            encoder_dropout: Encoder dropout probability
            projection_dropout: Projection dropout probability
            use_batch_norm: Whether to use batch normalization
            activation: Activation function
        """
        super(SSCLModel, self).__init__()
        
        self.encoder = Encoder(
            input_dim=input_dim,
            hidden_dims=encoder_hidden_dims,
            dropout=encoder_dropout,
            use_batch_norm=use_batch_norm,
            activation=activation
        )
        
        self.projection_head = ProjectionHead(
            input_dim=self.encoder.output_dim,
            projection_dim=projection_dim,
            hidden_dim=projection_hidden_dim,
            dropout=projection_dropout
        )
    
    def forward(self, x: torch.Tensor, return_projection: bool = True) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input tensor (batch_size, input_dim)
            return_projection: If True, return projection; else return encoder output
            
        Returns:
            Encoded or projected representation
        """
        encoded = self.encoder(x)
        if return_projection:
            return self.projection_head(encoded)
        return encoded
    
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """
        Get encoder representation (without projection).
        
        Args:
            x: Input tensor (batch_size, input_dim)
            
        Returns:
            Encoded representation (batch_size, encoder_output_dim)
        """
        return self.encoder(x)


class ContrastiveLoss(nn.Module):
    """
    Contrastive loss (InfoNCE) for self-supervised learning.
    """
    
    def __init__(self, temperature: float = 0.07):
        """
        Initialize contrastive loss.
        
        Args:
            temperature: Temperature parameter for softmax (default: 0.07)
        """
        super(ContrastiveLoss, self).__init__()
        self.temperature = temperature
    
    def forward(self, z1: torch.Tensor, z2: torch.Tensor) -> torch.Tensor:
        """
        Compute contrastive loss between two augmented views.
        
        Args:
            z1: Projected representation of first view (batch_size, projection_dim)
            z2: Projected representation of second view (batch_size, projection_dim)
            
        Returns:
            Contrastive loss value
        """
        batch_size = z1.size(0)
        
        # Normalize projections
        z1 = F.normalize(z1, dim=1)
        z2 = F.normalize(z2, dim=1)
        
        # Concatenate all projections
        all_projections = torch.cat([z1, z2], dim=0)  # (2*batch_size, projection_dim)
        
        # Compute similarity matrix
        similarity_matrix = torch.matmul(all_projections, all_projections.T) / self.temperature
        
        # Create labels: positive pairs are (i, i+batch_size) and (i+batch_size, i)
        labels = torch.arange(batch_size, device=z1.device)
        labels = torch.cat([labels + batch_size, labels], dim=0)  # (2*batch_size,)
        
        # Mask to remove self-similarity
        mask = torch.eye(2 * batch_size, device=z1.device).bool()
        similarity_matrix = similarity_matrix.masked_fill(mask, float('-inf'))
        
        # Compute cross-entropy loss
        loss = F.cross_entropy(similarity_matrix, labels)
        
        return loss


class TabularDataset(Dataset):
    """
    Dataset class for tabular data with augmentation.
    Accepts both pandas DataFrames/Series and numpy arrays.
    """
    
    def __init__(
        self,
        X,
        y = None,
        augment: bool = False,
        augmentation: Optional[TabularAugmentation] = None
    ):
        """
        Initialize dataset.
        
        Args:
            X: Features - can be pandas DataFrame, Series, or numpy array (n_samples, n_features)
            y: Optional labels - can be pandas Series or numpy array (n_samples,)
            augment: Whether to apply augmentation
            augmentation: Augmentation object
        """
        # Convert pandas to numpy if needed
        if isinstance(X, pd.DataFrame) or isinstance(X, pd.Series):
            X = X.values
        self.X = np.asarray(X, dtype=np.float32)
        
        if y is not None:
            if isinstance(y, pd.Series) or isinstance(y, pd.DataFrame):
                y = y.values
            self.y = np.asarray(y, dtype=np.float32)
        else:
            self.y = None
        
        self.augment = augment
        self.augmentation = augmentation if augmentation is not None else TabularAugmentation()
    
    def __len__(self) -> int:
        return len(self.X)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get item with optional augmentation.
        
        Args:
            idx: Sample index
            
        Returns:
            Dictionary with 'x' (and optionally 'x_aug', 'y')
        """
        x = self.X[idx]
        
        result = {'x': torch.tensor(x, dtype=torch.float32)}
        
        if self.augment:
            x_aug = self.augmentation.augment(x.reshape(1, -1)).flatten()
            result['x_aug'] = torch.tensor(x_aug, dtype=torch.float32)
        
        if self.y is not None:
            result['y'] = torch.tensor(self.y[idx], dtype=torch.float32)
        
        return result


class SSCLTrainer:
    """
    Trainer for Self-Supervised Contrastive Learning.
    """
    
    def __init__(
        self,
        model: SSCLModel,
        device: Optional[torch.device] = None,
        learning_rate: float = 0.001,
        weight_decay: float = 1e-4,
        temperature: float = 0.07
    ):
        """
        Initialize trainer.
        
        Args:
            model: SSCL model
            device: PyTorch device (default: auto-detect)
            learning_rate: Learning rate
            weight_decay: Weight decay for optimizer
            temperature: Temperature for contrastive loss
        """
        self.model = model
        
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = device
        
        self.model.to(self.device)
        
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )
        
        self.criterion = ContrastiveLoss(temperature=temperature)
        
        self.train_losses = []
        self.val_losses = []
    
    def train_epoch(
        self,
        dataloader: DataLoader,
        verbose: bool = True
    ) -> float:
        """
        Train for one epoch.
        
        Args:
            dataloader: Training dataloader
            verbose: Whether to print progress
            
        Returns:
            Average training loss
        """
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        
        for batch in dataloader:
            x = batch['x'].to(self.device)
            x_aug = batch['x_aug'].to(self.device)
            
            # Forward pass
            z1 = self.model(x)
            z2 = self.model(x_aug)
            
            # Compute loss
            loss = self.criterion(z1, z2)
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
        
        avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
        self.train_losses.append(avg_loss)
        
        if verbose:
            print(f"  Training Loss: {avg_loss:.4f}")
        
        return avg_loss
    
    def validate(
        self,
        dataloader: DataLoader,
        verbose: bool = True
    ) -> float:
        """
        Validate on validation set.
        
        Args:
            dataloader: Validation dataloader
            verbose: Whether to print progress
            
        Returns:
            Average validation loss
        """
        self.model.eval()
        total_loss = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for batch in dataloader:
                x = batch['x'].to(self.device)
                x_aug = batch['x_aug'].to(self.device)
                
                # Forward pass
                z1 = self.model(x)
                z2 = self.model(x_aug)
                
                # Compute loss
                loss = self.criterion(z1, z2)
                
                total_loss += loss.item()
                num_batches += 1
        
        avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
        self.val_losses.append(avg_loss)
        
        if verbose:
            print(f"  Validation Loss: {avg_loss:.4f}")
        
        return avg_loss
    
    def fit(
        self,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        epochs: int = 50,
        verbose: bool = True,
        early_stopping_patience: Optional[int] = None
    ) -> Dict[str, List[float]]:
        """
        Train the model.
        
        Args:
            train_loader: Training dataloader
            val_loader: Optional validation dataloader
            epochs: Number of training epochs
            verbose: Whether to print progress
            early_stopping_patience: Early stopping patience (None = disabled)
            
        Returns:
            Dictionary with training history
        """
        if verbose:
            print("=" * 80)
            print("SSCL TRAINING")
            print("=" * 80)
            print(f"Device: {self.device}")
            print(f"Epochs: {epochs}")
            print(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
            print("=" * 80)
        
        best_val_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(epochs):
            if verbose:
                print(f"\nEpoch {epoch + 1}/{epochs}")
            
            # Train
            train_loss = self.train_epoch(train_loader, verbose=verbose)
            
            # Validate
            if val_loader is not None:
                val_loss = self.validate(val_loader, verbose=verbose)
                
                # Early stopping
                if early_stopping_patience is not None:
                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
                        patience_counter = 0
                        # Save best model
                        self.best_model_state = self.model.state_dict().copy()
                    else:
                        patience_counter += 1
                        if patience_counter >= early_stopping_patience:
                            if verbose:
                                print(f"\nEarly stopping at epoch {epoch + 1}")
                            if hasattr(self, 'best_model_state'):
                                self.model.load_state_dict(self.best_model_state)
                            break
        
        history = {
            'train_loss': self.train_losses,
            'val_loss': self.val_losses
        }
        
        if verbose:
            print("\n" + "=" * 80)
            print("TRAINING COMPLETE!")
            print("=" * 80)
        
        return history
    
    def extract_features(
        self,
        X,
        use_projection: bool = False,
        batch_size: int = 32,
        return_pandas: bool = False
    ) -> np.ndarray:
        """
        Extract learned representations.
        
        Args:
            X: Input data - pandas DataFrame/Series or numpy array, or DataLoader
            use_projection: If True, use projection head; else use encoder only
            batch_size: Batch size (ignored if X is DataLoader)
            return_pandas: If True and X is pandas, return pandas DataFrame
            
        Returns:
            Extracted features (n_samples, feature_dim) as numpy array or pandas DataFrame
        """
        self.model.eval()
        
        # Handle DataLoader input (backward compatibility)
        if isinstance(X, DataLoader):
            features = []
            with torch.no_grad():
                for batch in X:
                    x = batch['x'].to(self.device)
                    if use_projection:
                        z = self.model(x, return_projection=True)
                    else:
                        z = self.model.encode(x)
                    features.append(z.cpu().numpy())
            return np.vstack(features)
        
        # Handle pandas/numpy input
        if isinstance(X, pd.DataFrame) or isinstance(X, pd.Series):
            X_np = X.values
            is_pandas = True
        else:
            X_np = np.asarray(X, dtype=np.float32)
            is_pandas = False
        
        # Create dataset and dataloader
        dataset = TabularDataset(X_np, augment=False)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
        
        features = []
        with torch.no_grad():
            for batch in dataloader:
                x = batch['x'].to(self.device)
                if use_projection:
                    z = self.model(x, return_projection=True)
                else:
                    z = self.model.encode(x)
                features.append(z.cpu().numpy())
        
        result = np.vstack(features)
        
        # Return as pandas DataFrame if requested and input was pandas
        if return_pandas and is_pandas:
            if isinstance(X, pd.DataFrame):
                return pd.DataFrame(result, index=X.index)
            else:
                return pd.DataFrame(result, index=X.index if hasattr(X, 'index') else None)
        
        return result


def train_sscl(
    X_train,
    X_val = None,
    input_dim: Optional[int] = None,
    encoder_hidden_dims: List[int] = [128, 256, 128],
    projection_dim: int = 64,
    projection_hidden_dim: int = 128,
    encoder_dropout: float = 0.2,
    projection_dropout: float = 0.1,
    batch_size: int = 32,
    epochs: int = 50,
    learning_rate: float = 0.001,
    weight_decay: float = 1e-4,
    temperature: float = 0.07,
    noise_std: float = 0.1,
    feature_dropout: float = 0.1,
    early_stopping_patience: Optional[int] = 10,
    verbose: bool = True,
    device: Optional[torch.device] = None
) -> Tuple[SSCLModel, SSCLTrainer, Dict[str, List[float]]]:
    """
    High-level function to train SSCL model.
    
    Accepts output directly from preprocessing.py (pandas DataFrames/Series).
    
    Args:
        X_train: Training features - pandas DataFrame/Series or numpy array (n_samples, n_features)
        X_val: Optional validation features - pandas DataFrame/Series or numpy array (n_samples, n_features)
        input_dim: Input dimension (auto-detected if None)
        encoder_hidden_dims: Encoder hidden dimensions
        projection_dim: Projection dimension
        projection_hidden_dim: Projection hidden dimension
        encoder_dropout: Encoder dropout
        projection_dropout: Projection dropout
        batch_size: Batch size
        epochs: Number of epochs
        learning_rate: Learning rate
        weight_decay: Weight decay
        temperature: Contrastive loss temperature
        noise_std: Augmentation noise std
        feature_dropout: Augmentation feature dropout
        early_stopping_patience: Early stopping patience
        verbose: Whether to print progress
        device: PyTorch device
        
    Returns:
        Tuple of (model, trainer, history)
    """
    # Convert pandas to numpy if needed
    if isinstance(X_train, pd.DataFrame) or isinstance(X_train, pd.Series):
        X_train = X_train.values
    X_train = np.asarray(X_train, dtype=np.float32)
    
    if X_val is not None:
        if isinstance(X_val, pd.DataFrame) or isinstance(X_val, pd.Series):
            X_val = X_val.values
        X_val = np.asarray(X_val, dtype=np.float32)
    
    # Auto-detect input dimension
    if input_dim is None:
        input_dim = X_train.shape[1]
    
    # Create augmentation
    augmentation = TabularAugmentation(
        noise_std=noise_std,
        feature_dropout=feature_dropout
    )
    
    # Create datasets
    train_dataset = TabularDataset(
        X_train,
        augment=True,
        augmentation=augmentation
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0  # Set to 0 for Windows compatibility
    )
    
    val_loader = None
    if X_val is not None:
        val_dataset = TabularDataset(
            X_val,
            augment=True,
            augmentation=augmentation
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=0
        )
    
    # Create model
    model = SSCLModel(
        input_dim=input_dim,
        encoder_hidden_dims=encoder_hidden_dims,
        projection_dim=projection_dim,
        projection_hidden_dim=projection_hidden_dim,
        encoder_dropout=encoder_dropout,
        projection_dropout=projection_dropout
    )
    
    # Create trainer
    trainer = SSCLTrainer(
        model=model,
        device=device,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        temperature=temperature
    )
    
    # Train
    history = trainer.fit(
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=epochs,
        verbose=verbose,
        early_stopping_patience=early_stopping_patience
    )
    
    return model, trainer, history


if __name__ == "__main__":
    print("SSCL Module for CKD Prediction")
    print("This module implements Self-Supervised Contrastive Learning for learning")
    print("rich semantic representations from medical tabular data.")
