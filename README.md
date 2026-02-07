# Federated Credit Card Fraud Detection

## Overview
This project implements a federated learning framework for credit card fraud detection,
focusing on privacy preservation, non-IID data, and communication-efficient training.

## Dataset

This project uses the public **Credit Card Fraud Detection** dataset by ULB (Kaggle).

Due to file size and repository best practices, the dataset is not included in this GitHub repository.

### How to obtain
1. Download the dataset from Kaggle:
https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud
2. Place `creditcard.csv` in this folder:
`data/creditcard.csv`

## Key Features
- Federated vs Centralized comparison
- Non-IID client partitioning
- Differential Privacy (DP)
- Top-K sparsification
- Evaluation using Recall, FNR, FPR, AUC

## Tech Stack
- Python
- NumPy / Pandas
- Scikit-learn / PyTorch (if used)
- Federated Learning simulation

## How to Run
1. Install dependencies
2. Prepare dataset (see Dataset section)
3. Run experiment scripts

## Results
