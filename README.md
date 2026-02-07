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
<img width="2670" height="1466" alt="Figure_6_X_Runwise_FNR_Comparison_DP_0_4" src="https://github.com/user-attachments/assets/18980571-9cf8-438f-a340-789e2225acc4" />
<img width="2970" height="1466" alt="Figure_6_X_FNR_Stability_FullFL_vs_TopKFL_NoDP" src="https://github.com/user-attachments/assets/fd4226fb-6562-4f43-a23b-260f927ea572" />
<img width="3270" height="1465" alt="Figure_5_X_FullFL_vs_TopKFL_Performance_and_Communication_NonIID_0 4" src="https://github.com/user-attachments/assets/551185cc-de14-4b61-a46b-a2d17f062a6f" />
<img width="3270" height="1465" alt="Figure_5_X_FullFL_vs_TopKFL_Performance_and_Communication_0 6" src="https://github.com/user-attachments/assets/0b54d91e-8e92-42e2-8f87-4891e396d24f" />
<img width="3270" height="1465" alt="Figure_5_X_FullFL_vs_TopKFL_Performance_and_Communication" src="https://github.com/user-attachments/assets/9917e595-3efe-4476-9490-3d60b96f3a4c" />
<img width="2970" height="1466" alt="Figure_4_X_Centralized_vs_Federated_Metric_Stability_exp2" src="https://github.com/user-attachments/assets/8407e4b8-01f2-45bc-8edc-deed9f2920c3" />
<img width="2970" height="1466" alt="Figure_4_X_Centralized_vs_Federated_Metric_Stability_exp1" src="https://github.com/user-attachments/assets/e49d8a8c-11de-40cb-9705-3b2dd17f1a24" />

