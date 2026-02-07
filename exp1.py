# fl_fedavg_full_with_all_metrics.py
# - Centralized baseline
# - Federated Learning (standard FedAvg only)
# - Detailed metrics: Accuracy, Precision, Recall, F1, AUC-ROC, FNR, FPR

import os
import logging
from typing import List, Dict

import numpy as np
import pandas as pd

import flwr as fl
from flwr.server.strategy import FedAvg

from fl_model_io import save_mlp_npz

# Create folder once at script start
os.makedirs("exported_models", exist_ok=True)

try:
    from flwr.common import parameters_to_ndarrays
except Exception:
    from flwr.common import ndarray_from_bytes
    def parameters_to_ndarrays(parameters) -> List[np.ndarray]:
        return [ndarray_from_bytes(t) for t in parameters.tensors]

from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (
    classification_report,
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    confusion_matrix,
)

# ---------------------------
# CONFIG
# ---------------------------
N_CLIENTS = 5
ROUNDS = 50
LOCAL_EPOCHS = 1
RAY_OBJECT_STORE_MB = 128
HIDDEN_UNITS = 64
RANDOM_STATE = 30

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
log = logging.getLogger("fl_fedavg_full")

# ---------------------------
# Metric helpers
# ---------------------------
def compute_rates(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    fnr = fn / (tp + fn) if (tp + fn) > 0 else np.nan
    fpr = fp / (fp + tn) if (fp + tn) > 0 else np.nan
    return {"fnr": float(fnr), "fpr": float(fpr)}

def safe_auc(y_true, y_prob):
    try:
        if y_prob is None:
            return float("nan")
        return float(roc_auc_score(y_true, y_prob))
    except Exception:
        return float("nan")

def summarize_metrics(y_true, y_pred, y_prob):
    acc = accuracy_score(y_true, y_pred)
    pre = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1  = f1_score(y_true, y_pred, zero_division=0)
    auc = safe_auc(y_true, y_prob)
    rates = compute_rates(y_true, y_pred)
    return {
        "acc": acc, "precision": pre, "recall": rec, "f1": f1, "auc": auc,
        "fnr": rates["fnr"], "fpr": rates["fpr"]
    }

def print_metrics_block(title, y_true, y_pred, y_prob):
    m = summarize_metrics(y_true, y_pred, y_prob)
    print(f"\n=== {title} ===")
    print(classification_report(y_true, y_pred, digits=4))
    print(
        f"Accuracy: {m['acc']:.4f}  Precision: {m['precision']:.4f}  Recall: {m['recall']:.4f}  "
        f"F1: {m['f1']:.4f}  AUC-ROC: {m['auc'] if not np.isnan(m['auc']) else 'nan':.4f}  "
        f"FNR: {m['fnr']:.4f}  FPR: {m['fpr']:.4f}"
    )
    return m

# ---------------------------
# Centralized baseline
# ---------------------------
def run_centralized_mlp(train_csv="train_resampled_processed.csv", test_csv="test_unbalanced_processed.csv"):
    train_df = pd.read_csv(train_csv)
    test_df = pd.read_csv(test_csv)
    X_train = train_df.drop(columns=["Class"]).values
    y_train = train_df["Class"].values
    X_test = test_df.drop(columns=["Class"]).values
    y_test = test_df["Class"].values

    clf = MLPClassifier(
        hidden_layer_sizes=(HIDDEN_UNITS,),
        batch_size=32,
        learning_rate_init=1e-3,
        solver="adam",
        max_iter=300,
        random_state=RANDOM_STATE,
    )
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    try:
        y_prob = clf.predict_proba(X_test)[:, 1]
    except Exception:
        y_prob = None
    metrics = print_metrics_block("Centralized MLP Baseline", y_test, y_pred, y_prob)
    return metrics, (X_test, y_test)

# ---------------------------
# Client data loading
# ---------------------------
def load_client_data_list(n_clients):
    clients = []
    for i in range(1, n_clients + 1):
        Xtr = pd.read_csv(f"client_{i}_X_train.csv").values
        ytr = pd.read_csv(f"client_{i}_y_train.csv").squeeze().to_numpy()
        Xte = pd.read_csv(f"client_{i}_X_test_unbalanced.csv").values
        yte = pd.read_csv(f"client_{i}_y_test_unbalanced.csv").squeeze().to_numpy()
        clients.append({"X_train": Xtr, "y_train": ytr, "X_test": Xte, "y_test": yte})
    return clients

def compute_all_classes(client_data_list):
    ys = [d["y_train"] for d in client_data_list]
    return np.unique(np.concatenate(ys)).astype(int)

# ---------------------------
# Model init and parameter helpers
# ---------------------------
def init_mlp(n_features, all_classes):
    mlp = MLPClassifier(
        hidden_layer_sizes=(HIDDEN_UNITS,),
        batch_size=32,
        learning_rate_init=1e-3,
        solver="adam",
        warm_start=True,
        max_iter=1,
        random_state=RANDOM_STATE,
    )
    X_seed = np.zeros((len(all_classes), n_features))
    y_seed = all_classes.copy()
    mlp.partial_fit(X_seed, y_seed, classes=all_classes)
    return mlp

def get_params(model):
    return [w.copy() for w in model.coefs_] + [b.copy() for b in model.intercepts_]

def set_params(model, params):
    n_layers = len(model.coefs_)
    model.coefs_ = [p.copy() for p in params[:n_layers]]
    model.intercepts_ = [p.copy() for p in params[n_layers:]]

# ---------------------------
# Flower Client (Full FedAvg)
# ---------------------------
class FraudClient(fl.client.NumPyClient):
    def __init__(self, Xtr, ytr, Xte, yte, all_classes):
        self.Xtr, self.ytr, self.Xte, self.yte = Xtr, ytr, Xte, yte
        self.ALL_CLASSES = all_classes
        self.model = init_mlp(Xtr.shape[1], all_classes)

    def get_parameters(self, config=None): return get_params(self.model)
    def set_parameters(self, parameters): set_params(self.model, parameters)

    def fit(self, parameters, config=None):
        self.set_parameters(parameters)
        local_epochs = int(config.get("local_epochs", 1)) if config else 1
        for _ in range(local_epochs):
            self.model.partial_fit(self.Xtr, self.ytr, classes=self.ALL_CLASSES)
        return get_params(self.model), len(self.Xtr), {}

    def evaluate(self, parameters, config=None):
        self.set_parameters(parameters)
        y_pred = self.model.predict(self.Xte)
        try:
            y_prob = self.model.predict_proba(self.Xte)[:, 1]
        except Exception:
            y_prob = None
        m = summarize_metrics(self.yte, y_pred, y_prob)
        loss = 1.0 - m["acc"]
        return loss, len(self.Xte), m

# ---------------------------
# FedAvg strategy that stores final weights
# ---------------------------
class FedAvgWithStore(FedAvg):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.latest_params = None
    def aggregate_fit(self, server_round, results, failures):
        aggregated, metrics = super().aggregate_fit(server_round, results, failures)
        if aggregated is not None:
            self.latest_params = aggregated
        return aggregated, metrics

# ---------------------------
# Run FL simulation and evaluate global model
# ---------------------------
def run_federated_sim(client_data_list, all_classes, rounds, local_epochs, ray_mem):
    import ray
    ray.shutdown()
    ray_tmp = os.path.join(os.getcwd(), "ray_tmp"); os.makedirs(ray_tmp, exist_ok=True)
    ray.init(local_mode=True, ignore_reinit_error=True,
             _temp_dir=ray_tmp, object_store_memory=ray_mem * 1024 * 1024,
             include_dashboard=False)
    log.info(f"Ray initialized (local_mode, {ray_mem}MB object store)")

    n = len(client_data_list)
    strategy = FedAvgWithStore(
        fraction_fit=1.0, min_fit_clients=n, min_available_clients=n,
        on_fit_config_fn=lambda rnd: {"local_epochs": local_epochs},
    )
    fl.simulation.start_simulation(
        client_fn=lambda cid: FraudClient(
            client_data_list[int(cid)]["X_train"],
            client_data_list[int(cid)]["y_train"],
            client_data_list[int(cid)]["X_test"],
            client_data_list[int(cid)]["y_test"],
            all_classes,
        ).to_client(),
        num_clients=n,
        config=fl.server.ServerConfig(num_rounds=rounds),
        strategy=strategy,
        client_resources={"num_cpus": 1},
    )

    X_test = np.vstack([d["X_test"] for d in client_data_list])
    y_test = np.concatenate([d["y_test"] for d in client_data_list])
    model = init_mlp(X_test.shape[1], all_classes)
    params = parameters_to_ndarrays(strategy.latest_params)
    set_params(model, params)
    
    # EXPORT GLOBAL MODEL (NPZ)
    save_mlp_npz(
        path="exported_models/exp1_fullfl_global_model.npz",
        params=params,
        n_features=X_test.shape[1],
        classes=all_classes,
        hidden_units=HIDDEN_UNITS
    )
    
    y_pred = model.predict(X_test)
    try: y_prob = model.predict_proba(X_test)[:, 1]
    except Exception: y_prob = None
    return print_metrics_block("Federated Global Model", y_test, y_pred, y_prob)

# ---------------------------
# Main
# ---------------------------
def main():
    central_metrics, _ = run_centralized_mlp()
    client_data_list = load_client_data_list(N_CLIENTS)
    all_classes = compute_all_classes(client_data_list)
    log.info(f"ALL_CLASSES: {all_classes.tolist()}")
    fed_metrics = run_federated_sim(client_data_list, all_classes, ROUNDS, LOCAL_EPOCHS, RAY_OBJECT_STORE_MB)

    print("\n=== Quick Compare (Centralized vs Federated) ===")
    def fmt(m): return (f"Acc:{m['acc']:.4f} Prec:{m['precision']:.4f} Rec:{m['recall']:.4f} "
                        f"F1:{m['f1']:.4f} AUC:{m['auc'] if not np.isnan(m['auc']) else 'nan':4f} "
                        f"FNR:{m['fnr']:.4f} FPR:{m['fpr']:.4f}")
    print("Centralized ->", fmt(central_metrics))
    print("Federated   ->", fmt(fed_metrics))

if __name__ == "__main__":
    main()
