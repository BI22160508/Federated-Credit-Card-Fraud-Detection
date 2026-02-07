import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from imblearn.over_sampling import SMOTE

RANDOM_STATE = 30
CENTRAL_SMOTE_RATIO = 0.1  # use 0.1 instead of default 1.0 (50/50)

def label_hour(hour):
    if 0 <= hour <= 5:
        return "Midnight"
    elif 6 <= hour <= 11:
        return "Morning"
    elif 12 <= hour <= 17:
        return "Afternoon"
    else:
        return "Evening"

# Load dataset
df = pd.read_csv("creditcard.csv")
X = df.drop(columns=["Class"]).copy()
y = df["Class"].copy()

# 1) Split FIRST (prevents leakage)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
)

# 2) Fit scaler on TRAIN only, transform both
scaler = MinMaxScaler()
X_train[["Time", "Amount"]] = scaler.fit_transform(X_train[["Time", "Amount"]])
X_test[["Time", "Amount"]] = scaler.transform(X_test[["Time", "Amount"]])

# Reset index for safe iloc use later
X_test = X_test.reset_index(drop=True)
y_test = y_test.reset_index(drop=True)
X_train = X_train.reset_index(drop=True)
y_train = y_train.reset_index(drop=True)

# 3) Feature engineering (same logic for train & test)
def add_time_features(df_):
    df_ = df_.copy()
    df_["Hour"] = (df_["Time"] * 24).astype(int) % 24
    df_["TimePeriod"] = df_["Hour"].apply(label_hour)
    df_.drop(columns=["Hour"], inplace=True)
    return df_

X_train = add_time_features(X_train)
X_test = add_time_features(X_test)

# 4) AmountBin: compute bin edges from TRAIN only, apply to both using pd.cut
q_edges = X_train["Amount"].quantile([0.0, 1/3, 2/3, 1.0]).values
q_edges = np.unique(q_edges)
if len(q_edges) < 4:
    q_edges = np.linspace(X_train["Amount"].min(), X_train["Amount"].max(), 4)

labels = ["Low", "Medium", "High"]
X_train["AmountBin"] = pd.cut(X_train["Amount"], bins=q_edges, labels=labels, include_lowest=True)
X_test["AmountBin"] = pd.cut(X_test["Amount"], bins=q_edges, labels=labels, include_lowest=True)

# 5) One-hot encode and ALIGN columns
X_train = pd.get_dummies(X_train, columns=["TimePeriod", "AmountBin"])
X_test = pd.get_dummies(X_test, columns=["TimePeriod", "AmountBin"])

X_train, X_test = X_train.align(X_test, join="left", axis=1, fill_value=0)

# Convert any bools to int
bool_cols = X_train.select_dtypes(include=["bool"]).columns
X_train[bool_cols] = X_train[bool_cols].astype(int)
X_test[bool_cols] = X_test[bool_cols].astype(int)

# Save IMBALANCED processed train (for federated partitioning)
train_imbal = X_train.copy()
train_imbal["Class"] = y_train.values
train_imbal.to_csv("train_imbalanced_processed.csv", index=False)

# 6) Centralized SMOTE (baseline training file) ✅ using 0.1 ratio
smote = SMOTE(
    sampling_strategy=CENTRAL_SMOTE_RATIO,
    random_state=RANDOM_STATE
)
X_train_res, y_train_res = smote.fit_resample(X_train, y_train)

train_res = X_train_res.copy()
train_res["Class"] = y_train_res.values
train_res.to_csv("train_resampled_processed.csv", index=False)

print(
    "[INFO] Centralized SMOTE applied | "
    f"ratio={CENTRAL_SMOTE_RATIO} | "
    f"total={len(y_train_res)}, fraud={int(y_train_res.sum())}, fraud_rate={y_train_res.mean():.6f}"
)

# 7) Create unbalanced + balanced test sets (processed)
X_test_unbalanced = X_test.copy()
y_test_unbalanced = y_test.copy()

fraud_idx = y_test[y_test == 1].index
nonfraud_idx = y_test[y_test == 0].sample(n=len(fraud_idx), random_state=RANDOM_STATE).index
balanced_idx = sorted(list(fraud_idx) + list(nonfraud_idx))

X_test_balanced = X_test.iloc[balanced_idx].reset_index(drop=True)
y_test_balanced = y_test.iloc[balanced_idx].reset_index(drop=True)

# Save processed tests
test_unbal = X_test_unbalanced.copy()
test_unbal["Class"] = y_test_unbalanced.values
test_unbal.to_csv("test_unbalanced_processed.csv", index=False)

test_bal = X_test_balanced.copy()
test_bal["Class"] = y_test_balanced.values
test_bal.to_csv("test_balanced_processed.csv", index=False)

print("Saved:")
print("- train_imbalanced_processed.csv (for FL partitioning)")
print("- train_resampled_processed.csv (centralized baseline with SMOTE=0.1)")
print("- test_unbalanced_processed.csv (real-world eval)")
print("- test_balanced_processed.csv (diagnostic)")
