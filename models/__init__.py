"""
Models module for CKD prediction.
Contains Transformer, SSCL, and EO implementations.
"""

from .sscl import (
    SSCLModel,
    Encoder,
    ProjectionHead,
    ContrastiveLoss,
    TabularAugmentation,
    TabularDataset,
    SSCLTrainer,
    train_sscl
)

from .transformer import (
    TabularTransformer,
    TransformerBlock,
    MultiHeadAttention,
    FeedForward,
    PositionalEncoding,
    FeatureEmbedding,
    TransformerDataset,
    TransformerTrainer,
    train_transformer
)

from .equilibrium_optimizer import (
    EquilibriumOptimizer,
    HyperparameterOptimizer
)

__all__ = [
    # SSCL
    'SSCLModel',
    'Encoder',
    'ProjectionHead',
    'ContrastiveLoss',
    'TabularAugmentation',
    'TabularDataset',
    'SSCLTrainer',
    'train_sscl',
    # Transformer
    'TabularTransformer',
    'TransformerBlock',
    'MultiHeadAttention',
    'FeedForward',
    'PositionalEncoding',
    'FeatureEmbedding',
    'TransformerDataset',
    'TransformerTrainer',
    'train_transformer',
    # Equilibrium Optimizer
    'EquilibriumOptimizer',
    'HyperparameterOptimizer'
]

