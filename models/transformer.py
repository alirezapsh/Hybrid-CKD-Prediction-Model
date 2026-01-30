import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from typing import Tuple, Optional, Dict, List
import math
import warnings
warnings.filterwarnings('ignore')


class PositionalEncoding(nn.Module):
    """
    Positional encoding for feature positions in tabular data.
    Since tabular data doesn't have natural sequence order, we use learnable
    or fixed positional encodings to help the model understand feature relationships.
    """
    
    def __init__(self, d_model: int, max_len: int = 5000, learnable: bool = True):
        """
        Initialize positional encoding.
        
        Args:
            d_model: Model dimension (feature embedding dimension)
            max_len: Maximum sequence length (number of features)
            learnable: If True, use learnable positional encoding; else use fixed sinusoidal
        """
        super(PositionalEncoding, self).__init__()
        self.d_model = d_model
        self.learnable = learnable
        
        if learnable:
            # Learnable positional encoding
            self.pos_encoding = nn.Parameter(torch.randn(max_len, d_model))
        else:
            # Fixed sinusoidal positional encoding
            pe = torch.zeros(max_len, d_model)
            position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
            div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
            pe[:, 0::2] = torch.sin(position * div_term)
            pe[:, 1::2] = torch.cos(position * div_term)
            pe = pe.unsqueeze(0)  # (1, max_len, d_model)
            self.register_buffer('pe', pe)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Add positional encoding to input.
        
        Args:
            x: Input tensor (batch_size, seq_len, d_model)
            
        Returns:
            Tensor with positional encoding added
        """
        if self.learnable:
            seq_len = x.size(1)
            return x + self.pos_encoding[:seq_len, :].unsqueeze(0)
        else:
            seq_len = x.size(1)
            return x + self.pe[:, :seq_len, :]


class MultiHeadAttention(nn.Module):
    """
    Multi-head self-attention mechanism for learning feature relationships.
    """
    
    def __init__(
        self,
        d_model: int,
        n_heads: int = 8,
        dropout: float = 0.1
    ):
        """
        Initialize multi-head attention.
        
        Args:
            d_model: Model dimension
            n_heads: Number of attention heads
            dropout: Dropout probability
        """
        super(MultiHeadAttention, self).__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
        self.scale = math.sqrt(self.d_k)
    
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass through multi-head attention.
        
        Args:
            x: Input tensor (batch_size, seq_len, d_model)
            mask: Optional attention mask
            
        Returns:
            Output tensor (batch_size, seq_len, d_model)
        """
        batch_size, seq_len, d_model = x.size()
        
        # Linear projections
        Q = self.W_q(x)  # (batch_size, seq_len, d_model)
        K = self.W_k(x)
        V = self.W_v(x)
        
        # Reshape for multi-head attention
        Q = Q.view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)  # (batch_size, n_heads, seq_len, d_k)
        K = K.view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        V = V.view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        
        # Scaled dot-product attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale  # (batch_size, n_heads, seq_len, seq_len)
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        # Apply attention to values
        context = torch.matmul(attention_weights, V)  # (batch_size, n_heads, seq_len, d_k)
        
        # Concatenate heads
        context = context.transpose(1, 2).contiguous().view(batch_size, seq_len, d_model)
        
        # Output projection
        output = self.W_o(context)
        
        return output


class FeedForward(nn.Module):
    """
    Position-wise feed-forward network.
    """
    
    def __init__(
        self,
        d_model: int,
        d_ff: int,
        dropout: float = 0.1,
        activation: str = 'gelu'
    ):
        """
        Initialize feed-forward network.
        
        Args:
            d_model: Model dimension
            d_ff: Feed-forward hidden dimension
            dropout: Dropout probability
            activation: Activation function ('relu', 'gelu', 'swish')
        """
        super(FeedForward, self).__init__()
        
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        
        if activation == 'relu':
            self.activation = nn.ReLU()
        elif activation == 'gelu':
            self.activation = nn.GELU()
        elif activation == 'swish':
            self.activation = nn.SiLU()
        else:
            raise ValueError(f"Unknown activation: {activation}")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through feed-forward network.
        
        Args:
            x: Input tensor (batch_size, seq_len, d_model)
            
        Returns:
            Output tensor (batch_size, seq_len, d_model)
        """
        return self.linear2(self.dropout(self.activation(self.linear1(x))))


class TransformerBlock(nn.Module):
    """
    Transformer encoder block with self-attention and feed-forward.
    """
    
    def __init__(
        self,
        d_model: int,
        n_heads: int = 8,
        d_ff: int = None,
        dropout: float = 0.1,
        activation: str = 'gelu'
    ):
        """
        Initialize transformer block.
        
        Args:
            d_model: Model dimension
            n_heads: Number of attention heads
            d_ff: Feed-forward hidden dimension (default: 4 * d_model)
            dropout: Dropout probability
            activation: Activation function
        """
        super(TransformerBlock, self).__init__()
        
        if d_ff is None:
            d_ff = 4 * d_model
        
        self.attention = MultiHeadAttention(d_model, n_heads, dropout)
        self.feed_forward = FeedForward(d_model, d_ff, dropout, activation)
        
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass through transformer block.
        
        Args:
            x: Input tensor (batch_size, seq_len, d_model)
            mask: Optional attention mask
            
        Returns:
            Output tensor (batch_size, seq_len, d_model)
        """
        # Self-attention with residual connection
        attn_output = self.attention(x, mask)
        x = self.norm1(x + self.dropout1(attn_output))
        
        # Feed-forward with residual connection
        ff_output = self.feed_forward(x)
        x = self.norm2(x + self.dropout2(ff_output))
        
        return x


class FeatureEmbedding(nn.Module):
    """
    Embedding layer for features.
    For tabular data, we treat each feature as a token and embed it.
    """
    
    def __init__(
        self,
        input_dim: int,
        d_model: int,
        dropout: float = 0.1
    ):
        """
        Initialize feature embedding.
        
        Args:
            input_dim: Input feature dimension (from SSCL)
            d_model: Model dimension (embedding dimension)
            dropout: Dropout probability
        """
        super(FeatureEmbedding, self).__init__()
        
        # For tabular data, we can either:
        # 1. Treat the entire feature vector as a single token
        # 2. Split features into multiple tokens
        # We'll use option 1: project input features to d_model
        
        self.linear = nn.Linear(input_dim, d_model)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Embed input features.
        
        Args:
            x: Input tensor (batch_size, input_dim)
            
        Returns:
            Embedded tensor (batch_size, 1, d_model) - single token per sample
        """
        # Project to d_model and add sequence dimension
        embedded = self.linear(x)  # (batch_size, d_model)
        embedded = embedded.unsqueeze(1)  # (batch_size, 1, d_model)
        return self.dropout(embedded)


class TabularTransformer(nn.Module):
    """
    Transformer model for tabular data.
    Accepts SSCL features and learns long-range dependencies between features.
    """
    
    def __init__(
        self,
        input_dim: int,
        d_model: int = 128,
        n_heads: int = 8,
        n_layers: int = 4,
        d_ff: Optional[int] = None,
        dropout: float = 0.1,
        activation: str = 'gelu',
        num_classes: int = 2,
        use_classification_head: bool = True,
        pooling: str = 'mean'  # 'mean', 'max', 'cls', 'last'
    ):
        """
        Initialize tabular transformer.
        
        Args:
            input_dim: Input feature dimension (from SSCL output)
            d_model: Model dimension (embedding dimension)
            n_heads: Number of attention heads
            n_layers: Number of transformer layers
            d_ff: Feed-forward hidden dimension (default: 4 * d_model)
            dropout: Dropout probability
            activation: Activation function
            num_classes: Number of output classes
            use_classification_head: Whether to include classification head
            pooling: Pooling strategy ('mean', 'max', 'cls', 'last')
        """
        super(TabularTransformer, self).__init__()
        
        self.input_dim = input_dim
        self.d_model = d_model
        self.num_classes = num_classes
        self.use_classification_head = use_classification_head
        self.pooling = pooling
        
        # Feature embedding
        self.embedding = FeatureEmbedding(input_dim, d_model, dropout)
        
        # Positional encoding (for single token per sample, this helps with batch processing)
        self.pos_encoding = PositionalEncoding(d_model, max_len=1, learnable=True)
        
        # Transformer blocks
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(d_model, n_heads, d_ff, dropout, activation)
            for _ in range(n_layers)
        ])
        
        # Classification head
        if use_classification_head:
            if pooling == 'cls':
                # Add CLS token
                self.cls_token = nn.Parameter(torch.randn(1, 1, d_model))
                self.classifier = nn.Sequential(
                    nn.LayerNorm(d_model),
                    nn.Linear(d_model, d_model // 2),
                    nn.GELU(),
                    nn.Dropout(dropout),
                    nn.Linear(d_model // 2, num_classes)
                )
            else:
                self.classifier = nn.Sequential(
                    nn.LayerNorm(d_model),
                    nn.Linear(d_model, d_model // 2),
                    nn.GELU(),
                    nn.Dropout(dropout),
                    nn.Linear(d_model // 2, num_classes)
                )
    
    def forward(self, x: torch.Tensor, return_features: bool = False) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input tensor (batch_size, input_dim) - SSCL features
            return_features: If True, return features before classification
            
        Returns:
            Output tensor (batch_size, num_classes) or (batch_size, d_model) if return_features
        """
        # Embed features
        x = self.embedding(x)  # (batch_size, 1, d_model)
        
        # Add positional encoding
        x = self.pos_encoding(x)
        
        # Add CLS token if using CLS pooling
        if self.pooling == 'cls' and self.use_classification_head:
            cls_tokens = self.cls_token.expand(x.size(0), -1, -1)
            x = torch.cat([cls_tokens, x], dim=1)  # (batch_size, 2, d_model)
        
        # Pass through transformer blocks
        for transformer_block in self.transformer_blocks:
            x = transformer_block(x)
        
        # Pooling
        if self.pooling == 'cls' and self.use_classification_head:
            # Use CLS token
            pooled = x[:, 0, :]  # (batch_size, d_model)
        elif self.pooling == 'mean':
            pooled = x.mean(dim=1)  # (batch_size, d_model)
        elif self.pooling == 'max':
            pooled = x.max(dim=1)[0]  # (batch_size, d_model)
        elif self.pooling == 'last':
            pooled = x[:, -1, :]  # (batch_size, d_model)
        else:
            pooled = x.mean(dim=1)
        
        if return_features:
            return pooled
        
        if self.use_classification_head:
            # Classification
            output = self.classifier(pooled)
            return output
        else:
            return pooled


class TransformerDataset(Dataset):
    """
    Dataset class for transformer training.
    Accepts numpy arrays from SSCL or pandas DataFrames.
    """
    
    def __init__(
        self,
        X,
        y = None
    ):
        """
        Initialize dataset.
        
        Args:
            X: Features - numpy array or pandas DataFrame (n_samples, n_features)
            y: Optional labels - numpy array or pandas Series (n_samples,)
        """
        # Convert pandas to numpy if needed
        if isinstance(X, pd.DataFrame) or isinstance(X, pd.Series):
            X = X.values
        self.X = np.asarray(X, dtype=np.float32)
        
        if y is not None:
            if isinstance(y, pd.Series) or isinstance(y, pd.DataFrame):
                y = y.values
            self.y = np.asarray(y, dtype=np.int64)
        else:
            self.y = None
    
    def __len__(self) -> int:
        return len(self.X)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get item.
        
        Args:
            idx: Sample index
            
        Returns:
            Dictionary with 'x' (and optionally 'y')
        """
        result = {'x': torch.tensor(self.X[idx], dtype=torch.float32)}
        
        if self.y is not None:
            result['y'] = torch.tensor(self.y[idx], dtype=torch.long)
        
        return result


class TransformerTrainer:
    """
    Trainer for Transformer model.
    """
    
    def __init__(
        self,
        model: TabularTransformer,
        device: Optional[torch.device] = None,
        learning_rate: float = 0.0001,
        weight_decay: float = 1e-4
    ):
        """
        Initialize trainer.
        
        Args:
            model: Transformer model
            device: PyTorch device (default: auto-detect)
            learning_rate: Learning rate
            weight_decay: Weight decay for optimizer
        """
        self.model = model
        
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = device
        
        self.model.to(self.device)
        
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )
        
        self.criterion = nn.CrossEntropyLoss()
        
        self.train_losses = []
        self.val_losses = []
        self.train_accuracies = []
        self.val_accuracies = []
    
    def train_epoch(
        self,
        dataloader: DataLoader,
        verbose: bool = True
    ) -> Tuple[float, float]:
        """
        Train for one epoch.
        
        Args:
            dataloader: Training dataloader
            verbose: Whether to print progress
            
        Returns:
            Tuple of (average_loss, accuracy)
        """
        self.model.train()
        total_loss = 0.0
        correct = 0
        total = 0
        num_batches = 0
        
        for batch in dataloader:
            x = batch['x'].to(self.device)
            y = batch['y'].to(self.device)
            
            # Forward pass
            outputs = self.model(x)
            loss = self.criterion(outputs, y)
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            # Calculate accuracy
            _, predicted = torch.max(outputs.data, 1)
            total += y.size(0)
            correct += (predicted == y).sum().item()
            
            total_loss += loss.item()
            num_batches += 1
        
        avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
        accuracy = 100.0 * correct / total if total > 0 else 0.0
        
        self.train_losses.append(avg_loss)
        self.train_accuracies.append(accuracy)
        
        if verbose:
            print(f"  Training Loss: {avg_loss:.4f}, Accuracy: {accuracy:.2f}%")
        
        return avg_loss, accuracy
    
    def validate(
        self,
        dataloader: DataLoader,
        verbose: bool = True
    ) -> Tuple[float, float]:
        """
        Validate on validation set.
        
        Args:
            dataloader: Validation dataloader
            verbose: Whether to print progress
            
        Returns:
            Tuple of (average_loss, accuracy)
        """
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0
        num_batches = 0
        
        with torch.no_grad():
            for batch in dataloader:
                x = batch['x'].to(self.device)
                y = batch['y'].to(self.device)
                
                # Forward pass
                outputs = self.model(x)
                loss = self.criterion(outputs, y)
                
                # Calculate accuracy
                _, predicted = torch.max(outputs.data, 1)
                total += y.size(0)
                correct += (predicted == y).sum().item()
                
                total_loss += loss.item()
                num_batches += 1
        
        avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
        accuracy = 100.0 * correct / total if total > 0 else 0.0
        
        self.val_losses.append(avg_loss)
        self.val_accuracies.append(accuracy)
        
        if verbose:
            print(f"  Validation Loss: {avg_loss:.4f}, Accuracy: {accuracy:.2f}%")
        
        return avg_loss, accuracy
    
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
            print("TRANSFORMER TRAINING")
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
            train_loss, train_acc = self.train_epoch(train_loader, verbose=verbose)
            
            # Validate
            if val_loader is not None:
                val_loss, val_acc = self.validate(val_loader, verbose=verbose)
                
                # Early stopping
                if early_stopping_patience is not None:
                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
                        patience_counter = 0
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
            'val_loss': self.val_losses,
            'train_accuracy': self.train_accuracies,
            'val_accuracy': self.val_accuracies
        }
        
        if verbose:
            print("\n" + "=" * 80)
            print("TRAINING COMPLETE!")
            print("=" * 80)
        
        return history
    
    def predict(self, X, batch_size: int = 32) -> np.ndarray:
        """
        Make predictions.
        
        Args:
            X: Input data - numpy array or pandas DataFrame
            batch_size: Batch size
            
        Returns:
            Predictions (n_samples,)
        """
        self.model.eval()
        
        # Convert pandas to numpy if needed
        if isinstance(X, pd.DataFrame) or isinstance(X, pd.Series):
            X = X.values
        X = np.asarray(X, dtype=np.float32)
        
        # Create dataset and dataloader
        dataset = TransformerDataset(X)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
        
        predictions = []
        with torch.no_grad():
            for batch in dataloader:
                x = batch['x'].to(self.device)
                outputs = self.model(x)
                _, predicted = torch.max(outputs, 1)
                predictions.append(predicted.cpu().numpy())
        
        return np.concatenate(predictions)
    
    def extract_features(
        self,
        X,
        batch_size: int = 32
    ) -> np.ndarray:
        """
        Extract learned features from transformer.
        
        Args:
            X: Input data - numpy array or pandas DataFrame
            batch_size: Batch size
            
        Returns:
            Extracted features (n_samples, d_model)
        """
        self.model.eval()
        
        # Convert pandas to numpy if needed
        if isinstance(X, pd.DataFrame) or isinstance(X, pd.Series):
            X = X.values
        X = np.asarray(X, dtype=np.float32)
        
        # Create dataset and dataloader
        dataset = TransformerDataset(X)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
        
        features = []
        with torch.no_grad():
            for batch in dataloader:
                x = batch['x'].to(self.device)
                feat = self.model(x, return_features=True)
                features.append(feat.cpu().numpy())
        
        return np.vstack(features)


def train_transformer(
    X_train,
    y_train,
    X_val = None,
    y_val = None,
    input_dim: Optional[int] = None,
    d_model: int = 128,
    n_heads: int = 8,
    n_layers: int = 4,
    d_ff: Optional[int] = None,
    dropout: float = 0.1,
    activation: str = 'gelu',
    num_classes: int = 2,
    batch_size: int = 32,
    epochs: int = 50,
    learning_rate: float = 0.0001,
    weight_decay: float = 1e-4,
    early_stopping_patience: Optional[int] = 10,
    verbose: bool = True,
    device: Optional[torch.device] = None
) -> Tuple[TabularTransformer, TransformerTrainer, Dict[str, List[float]]]:
    """
    High-level function to train Transformer model.
    
    Accepts SSCL features directly (numpy arrays).
    
    Args:
        X_train: Training features - numpy array from SSCL (n_samples, n_features)
        y_train: Training labels - numpy array or pandas Series (n_samples,)
        X_val: Optional validation features
        y_val: Optional validation labels
        input_dim: Input dimension (auto-detected if None)
        d_model: Model dimension
        n_heads: Number of attention heads
        n_layers: Number of transformer layers
        d_ff: Feed-forward hidden dimension
        dropout: Dropout probability
        activation: Activation function
        num_classes: Number of output classes
        batch_size: Batch size
        epochs: Number of epochs
        learning_rate: Learning rate
        weight_decay: Weight decay
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
    
    if isinstance(y_train, pd.Series) or isinstance(y_train, pd.DataFrame):
        y_train = y_train.values
    y_train = np.asarray(y_train, dtype=np.int64)
    
    if X_val is not None:
        if isinstance(X_val, pd.DataFrame) or isinstance(X_val, pd.Series):
            X_val = X_val.values
        X_val = np.asarray(X_val, dtype=np.float32)
    
    if y_val is not None:
        if isinstance(y_val, pd.Series) or isinstance(y_val, pd.DataFrame):
            y_val = y_val.values
        y_val = np.asarray(y_val, dtype=np.int64)
    
    # Auto-detect input dimension
    if input_dim is None:
        input_dim = X_train.shape[1]
    
    # Create datasets
    train_dataset = TransformerDataset(X_train, y_train)
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0
    )
    
    val_loader = None
    if X_val is not None and y_val is not None:
        val_dataset = TransformerDataset(X_val, y_val)
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=0
        )
    
    # Create model
    model = TabularTransformer(
        input_dim=input_dim,
        d_model=d_model,
        n_heads=n_heads,
        n_layers=n_layers,
        d_ff=d_ff,
        dropout=dropout,
        activation=activation,
        num_classes=num_classes
    )
    
    # Create trainer
    trainer = TransformerTrainer(
        model=model,
        device=device,
        learning_rate=learning_rate,
        weight_decay=weight_decay
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
    print("Transformer Module for CKD Prediction")
    print("This module implements Transformer architecture to learn long-range")
    print("dependencies and relationships between features from SSCL representations.")
