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

<img width="2970" height="1466" alt="Figure_4_X_Centralized_vs_Federated_Metric_Stability_exp1" src="https://github.com/user-attachments/assets/a50b1ec3-67ff-4db9-90a8-83b4834b3b31" />
<img width="2970" height="1466" alt="Figure_4_X_Centralized_vs_Federated_Metric_Stability_exp2" src="https://github.com/user-attachments/assets/dabaa155-b827-4cf0-b4ac-532fa1f30753" />
<img width="2970" height="1466" alt="Figure_6_X_FNR_Stability_FullFL_vs_TopKFL_NoDP" src="https://github.com/user-attachments/assets/27779160-4545-4b2d-88b3-b0c569f12d08" />
<img width="2670" height="1466" alt="Figure_6_X_Runwise_FNR_Comparison_DP_0_4" src="https://github.com/user-attachments/assets/6e0e6b3e-3e77-438b-a61a-5d2cbbd6a81e" />
<img width="2670" height="1466" alt="Figure_6_X_Runwise_FNR_DP_0_6" src="https://github.com/user-attachments/assets/c82710e0-40ac-4184-ae9c-9521a2474588" />

