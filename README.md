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
<img width="2670" height="1466" alt="F<img width="2670" height="1466" alt="Figure_6_X_Runwise_FNR_Comparison_DP_0_4" src="https://github.com/user-attachments/assets/229e70ba-3519-4a26-88f8-dadc55e53c5f" />
igure_6_X_Runwise_FNR_DP_0_6" src="https://github.com/user-attachments/assets/187e419a-3512-4bb6-9aee-cb131e2a979f" />
