# Federated Credit Card Fraud Detection

## Overview
This project implements a federated learning framework for credit card fraud detection,
focusing on privacy preservation, non-IID data, and communication-efficient training.

-exp1.py / exp2.py / exp3.py = training

-dv_exp*.py = plotting

-data_partition.py = non-IID client split

-exported_models/ = saved models

-eval_demo_app.py = demo app

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
<img width="3270" height="1465" alt="Figure_5_X_FullFL_vs_TopKFL_Performance_and_Communication" src="https://github.com/user-attachments/assets/9c57a757-d1ea-4305-be0a-60363e746e1b" />
<img width="3270" height="1465" alt="Figure_5_X_FullFL_vs_TopKFL_Performance_and_Communication_NonIID_0 4" src="https://github.com/user-attachments/assets/28d48976-6e02-4e8c-a654-dee14f4d3a83" />
<img width="3270" height="1465" alt="Figure_5_X_FullFL_vs_TopKFL_Performance_and_Communication_0 6" src="https://github.com/user-attachments/assets/1a4b68af-c129-49da-ac79-b2aa9e129438" />
