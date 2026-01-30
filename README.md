
# Hybrid Framework for CKD Prediction (SSCL-Transformer-Metaheuristic Algorithms)

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.10+-ee4c2c.svg)](https://pytorch.org/)

## Project Overview
This repository contains the source code for a hybrid deep learning framework designed for the effective prediction of Chronic Kidney Disease (CKD) from tabular medical data.

Addressing the challenges of small sample sizes and missing values in medical datasets, our framework integrates three key components:
1.  **Layer 1: Self-Supervised Contrastive Learning (SSCL):** Learns robust semantic representations from unlabeled data using tabular augmentation and InfoNCE loss.
2.  **Layer 2: Transformer Architecture:** Captures complex, non-linear, and long-range dependencies between clinical features using Multi-Head Attention.
3.  **Layer 3: Metaheuristic Algortithm:** Meta-heuristic algorithms used to automatically fine-tune the hyperparameters of the deep learning models.

## Dataset

The dataset used is the Chronic Kidney Disease dataset from UCI Machine Learning Repository:
- **File**: `data/chronic_kidney_disease_full.arff`
- **Instances**: 400 (250 CKD, 150 notckd)
- **Attributes**: 24 features + 1 class attribute

## Preprocessing Pipeline

The preprocessing module (`preprocessing.py`) implements the following steps:

1. **Cleaning**: Handle missing values using:
   - Moving Average (with configurable window size)
   - K-Nearest Neighbors (KNN) imputation

2. **Outlier Removal**: Apply the Interquartile Range (IQR) method

3. **Normalization**: Use Min-Max Scaling to standardize features (range: 0-1)

4. **Feature Selection**: Conduct initial selection using Mutual Information

5. **Data Splitting**: 
   - 70% Training
   - 15% Validation
   - 15% Testing

## Installation

1. Install required packages:
```bash
pip install -r requirements.txt
```
