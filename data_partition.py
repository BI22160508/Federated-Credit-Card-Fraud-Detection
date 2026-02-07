import os
import pandas as pd
import numpy as np
from imblearn.over_sampling import SMOTE

# -----------------------------
# Config
# -----------------------------
N_CLIENTS = 5
RANDOM_STATE = 30

CLIENT_SIZE_SPLITS = [0.40, 0.25, 0.15, 0.10, 0.10]
FRAUD_SHARE_SPLITS = [0.40, 0.25, 0.15, 0.10, 0.10]

USE_SHARED_BALANCED_TEST = True
APPLY_LOCAL_SMOTE = True
LOCAL_SMOTE_RATIO = 0.1   

OUTPUT_DIR = "clients_out"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# -----------------------------
# Load datasets (processed + imbalanced train)
# -----------------------------
train_df = pd.read_csv("train_imbalanced_processed.csv")
test_df = pd.read_csv("test_balanced_processed.csv")
test_unbalanced_df = pd.read_csv("test_unbalanced_processed.csv")

# Shuffle
train_df = train_df.sample(frac=1, random_state=RANDOM_STATE).reset_index(drop=True)
test_df = test_df.sample(frac=1, random_state=RANDOM_STATE).reset_index(drop=True)
test_unbalanced_df = test_unbalanced_df.sample(frac=1, random_state=RANDOM_STATE).reset_index(drop=True)

# -----------------------------
# Pools
# -----------------------------
fraud_df = train_df[train_df["Class"] == 1].copy().reset_index(drop=True)
nonfraud_df = train_df[train_df["Class"] == 0].copy().reset_index(drop=True)

n_total_train = len(train_df)
n_total_fraud = len(fraud_df)
n_total_nonfraud = len(nonfraud_df)

print(f"[INFO] Train total={n_total_train}, fraud={n_total_fraud}, nonfraud={n_total_nonfraud}")
print(f"[INFO] Global fraud rate = {n_total_fraud/n_total_train:.6f}")

# -----------------------------
# Client sizes (Option A)
# -----------------------------
client_sizes = [int(p * n_total_train) for p in CLIENT_SIZE_SPLITS]
client_sizes[-1] += n_total_train - sum(client_sizes)
print("[INFO] Client total sizes:", client_sizes, "sum=", sum(client_sizes))

# -----------------------------
# Fraud counts by share
# -----------------------------
fraud_counts = [int(p * n_total_fraud) for p in FRAUD_SHARE_SPLITS]
fraud_counts[-1] += n_total_fraud - sum(fraud_counts)
print("[INFO] Fraud counts per client:", fraud_counts, "sum=", sum(fraud_counts))

# Basic feasibility checks
for i in range(N_CLIENTS):
    if fraud_counts[i] > client_sizes[i]:
        raise ValueError(f"[ERROR] client_{i+1}: fraud_count {fraud_counts[i]} > total_size {client_sizes[i]}")

# Total nonfraud needed across clients
nonfraud_needed = sum([client_sizes[i] - fraud_counts[i] for i in range(N_CLIENTS)])
if nonfraud_needed > n_total_nonfraud:
    raise ValueError(
        f"[ERROR] Not enough nonfraud to fill clients. "
        f"Needed={nonfraud_needed}, Available={n_total_nonfraud}"
    )

# -----------------------------
# Split unbalanced test into unequal chunks
# -----------------------------
test_unbal_sizes = [int(p * len(test_unbalanced_df)) for p in CLIENT_SIZE_SPLITS]
test_unbal_sizes[-1] += len(test_unbalanced_df) - sum(test_unbal_sizes)

def split_by_sizes(df, sizes):
    out, start = [], 0
    for sz in sizes:
        out.append(df.iloc[start:start+sz].reset_index(drop=True))
        start += sz
    return out

test_unbal_chunks = split_by_sizes(test_unbalanced_df, test_unbal_sizes)

# -----------------------------
# Build client datasets
# -----------------------------
client_data = {}
summary_rows = []

fraud_pool = fraud_df.sample(frac=1, random_state=RANDOM_STATE).reset_index(drop=True)
nonfraud_pool = nonfraud_df.sample(frac=1, random_state=RANDOM_STATE).reset_index(drop=True)

fraud_start = 0
nonfraud_start = 0

for i in range(N_CLIENTS):
    client_name = f"client_{i+1}"
    total_i = client_sizes[i]
    fraud_i = fraud_counts[i]
    nonfraud_i = total_i - fraud_i

    client_fraud = fraud_pool.iloc[fraud_start:fraud_start+fraud_i]
    fraud_start += fraud_i

    client_nonfraud = nonfraud_pool.iloc[nonfraud_start:nonfraud_start+nonfraud_i]
    nonfraud_start += nonfraud_i

    client_train = pd.concat([client_fraud, client_nonfraud], ignore_index=True)
    client_train = client_train.sample(frac=1, random_state=RANDOM_STATE+i).reset_index(drop=True)

    # BEFORE SMOTE stats
    before_total = len(client_train)
    before_fraud = int(client_train["Class"].sum())
    before_rate = before_fraud / before_total if before_total else 0.0

    # Local SMOTE (training only)
    if APPLY_LOCAL_SMOTE:
        X_local = client_train.drop(columns=["Class"])
        y_local = client_train["Class"]
        fraud_local = int(y_local.sum())

        if fraud_local < 2:
            print(f"[WARN] {client_name}: fraud<{2}, skip SMOTE.")
            X_res, y_res = X_local, y_local
        else:
            k = min(5, fraud_local - 1)

            # Use different random_state per client to avoid identical synthetic samples
            smote = SMOTE(
                sampling_strategy=LOCAL_SMOTE_RATIO,
                random_state=RANDOM_STATE + i + 1,
                k_neighbors=k
            )
            X_res, y_res = smote.fit_resample(X_local, y_local)

        client_train = pd.concat([X_res, y_res.rename("Class")], axis=1)
        client_train = client_train.sample(frac=1, random_state=RANDOM_STATE+i+99).reset_index(drop=True)

    # AFTER SMOTE stats
    after_total = len(client_train)
    after_fraud = int(client_train["Class"].sum())
    after_rate = after_fraud / after_total if after_total else 0.0

    print(
        f"{client_name}: "
        f"BeforeSMOTE total={before_total}, fraud={before_fraud}, rate={before_rate:.6f} | "
        f"AfterSMOTE total={after_total}, fraud={after_fraud}, rate={after_rate:.6f}"
    )

    # Attach tests
    client_test_balanced = test_df if USE_SHARED_BALANCED_TEST else None
    client_test_unbalanced = test_unbal_chunks[i]

    client_data[client_name] = {
        "X_train": client_train.drop(columns=["Class"]),
        "y_train": client_train["Class"],
        "X_test_balanced": client_test_balanced.drop(columns=["Class"]) if USE_SHARED_BALANCED_TEST else None,
        "y_test_balanced": client_test_balanced["Class"] if USE_SHARED_BALANCED_TEST else None,
        "X_test_unbalanced": client_test_unbalanced.drop(columns=["Class"]),
        "y_test_unbalanced": client_test_unbalanced["Class"],
    }

    # Summary row for report table
    summary_rows.append({
        "client": client_name,
        "train_total_before": before_total,
        "train_fraud_before": before_fraud,
        "train_rate_before": before_rate,
        "train_total_after": after_total,
        "train_fraud_after": after_fraud,
        "train_rate_after": after_rate,
        "test_unbal_total": len(client_test_unbalanced),
        "test_unbal_fraud": int(client_test_unbalanced["Class"].sum()),
        "test_unbal_rate": float(client_test_unbalanced["Class"].mean())
    })

print("[INFO] Clients created:", list(client_data.keys()))

# -----------------------------
# Export CSVs
# -----------------------------
for i in range(N_CLIENTS):
    client_name = f"client_{i+1}"
    c = client_data[client_name]

    c["X_train"].to_csv(os.path.join(OUTPUT_DIR, f"{client_name}_X_train.csv"), index=False)
    c["y_train"].to_frame(name="Class").to_csv(os.path.join(OUTPUT_DIR, f"{client_name}_y_train.csv"), index=False)

    if USE_SHARED_BALANCED_TEST:
        c["X_test_balanced"].to_csv(os.path.join(OUTPUT_DIR, f"{client_name}_X_test_balanced.csv"), index=False)
        c["y_test_balanced"].to_frame(name="Class").to_csv(os.path.join(OUTPUT_DIR, f"{client_name}_y_test_balanced.csv"), index=False)

    c["X_test_unbalanced"].to_csv(os.path.join(OUTPUT_DIR, f"{client_name}_X_test_unbalanced.csv"), index=False)
    c["y_test_unbalanced"].to_frame(name="Class").to_csv(os.path.join(OUTPUT_DIR, f"{client_name}_y_test_unbalanced.csv"), index=False)

summary_df = pd.DataFrame(summary_rows)
summary_df.to_csv(os.path.join(OUTPUT_DIR, "client_summary.csv"), index=False)

print(f"[INFO] Exported client datasets to: {OUTPUT_DIR}/")
print(f"[INFO] Saved client summary table: {OUTPUT_DIR}/client_summary.csv")

# Remaining pool checks
print(f"[INFO] Remaining fraud in pool (unused): {len(fraud_pool) - fraud_start}")
print(f"[INFO] Remaining nonfraud in pool (unused): {len(nonfraud_pool) - nonfraud_start}")

# -----------------------------
# Print test statistics per client
# -----------------------------
print("\n[INFO] Test set statistics per client")
for row in summary_rows:
    print(
        f"{row['client']} | "
        f"Test(Bal): total={len(test_df)}, fraud={int(test_df['Class'].sum())}, rate={test_df['Class'].mean():.6f} | "
        f"Test(Unbal): total={row['test_unbal_total']}, fraud={row['test_unbal_fraud']}, rate={row['test_unbal_rate']:.6f}"
    )
