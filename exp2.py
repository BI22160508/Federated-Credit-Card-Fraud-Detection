# exp2.py
# - Centralized baseline
# - Federated Learning (standard FedAvg only)
# - Detailed metrics: Accuracy, Precision, Recall, F1, AUC-ROC, FNR, FPR
# - Multi-seed loop + Mean±Std + CSV exports

import os
import shutil
import logging
import random
from typing import List

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

# Experiment 1 (fixed seed): set FIXED_SEED to an int, e.g. 30
# Experiment 2 (robustness): set FIXED_SEED = None and provide SEEDS list
FIXED_SEED = None
SEEDS = [1, 7, 13, 21, 30, 42, 56, 73, 89, 101]

# Data files
CENT_TRAIN_CSV = "train_resampled_processed.csv"
CENT_TEST_CSV = "test_unbalanced_processed.csv"

CLIENT_X_TRAIN_FMT = "client_{i}_X_train.csv"
CLIENT_Y_TRAIN_FMT = "client_{i}_y_train.csv"
CLIENT_X_TEST_FMT = "client_{i}_X_test_unbalanced.csv"
CLIENT_Y_TEST_FMT = "client_{i}_y_test_unbalanced.csv"

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
log = logging.getLogger("fl_fedavg_full")


# ---------------------------
# Reproducibility helpers
# ---------------------------
def set_global_seed(seed: int) -> None:
    """Control common RNG sources."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)


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
    f1 = f1_score(y_true, y_pred, zero_division=0)
    auc = safe_auc(y_true, y_prob)
    rates = compute_rates(y_true, y_pred)
    return {
        "acc": float(acc),
        "precision": float(pre),
        "recall": float(rec),
        "f1": float(f1),
        "auc": float(auc),
        "fnr": float(rates["fnr"]),
        "fpr": float(rates["fpr"]),
    }


def print_metrics_block(title, y_true, y_pred, y_prob):
    m = summarize_metrics(y_true, y_pred, y_prob)
    print(f"\n=== {title} ===")
    print(classification_report(y_true, y_pred, digits=4))
    auc_val = m["auc"]
    print(
        f"Accuracy: {m['acc']:.4f}  Precision: {m['precision']:.4f}  Recall: {m['recall']:.4f}  "
        f"F1: {m['f1']:.4f}  AUC-ROC: {auc_val:.4f}  "
        f"FNR: {m['fnr']:.4f}  FPR: {m['fpr']:.4f}"
    )
    return m


def quick_fmt(m):
    auc_val = m["auc"]
    return (
        f"Acc:{m['acc']:.4f} Prec:{m['precision']:.4f} Rec:{m['recall']:.4f} "
        f"F1:{m['f1']:.4f} AUC:{auc_val:.4f} FNR:{m['fnr']:.4f} FPR:{m['fpr']:.4f}"
    )


# ---------------------------
# Centralized baseline
# ---------------------------
def run_centralized_mlp(seed: int, train_csv=CENT_TRAIN_CSV, test_csv=CENT_TEST_CSV):
    train_df = pd.read_csv(train_csv)
    test_df = pd.read_csv(test_csv)

    X_train = train_df.drop(columns=["Class"]).values
    y_train = train_df["Class"].values.astype(int)

    X_test = test_df.drop(columns=["Class"]).values
    y_test = test_df["Class"].values.astype(int)

    clf = MLPClassifier(
        hidden_layer_sizes=(HIDDEN_UNITS,),
        batch_size=32,
        learning_rate_init=1e-3,
        solver="adam",
        max_iter=300,
        random_state=seed,
    )
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    try:
        y_prob = clf.predict_proba(X_test)[:, 1]
    except Exception:
        y_prob = None

    metrics = print_metrics_block(f"Centralized MLP Baseline (seed={seed})", y_test, y_pred, y_prob)
    return metrics


# ---------------------------
# Client data loading
# ---------------------------
def load_client_data_list(n_clients: int):
    clients = []
    for i in range(1, n_clients + 1):
        Xtr = pd.read_csv(CLIENT_X_TRAIN_FMT.format(i=i)).values
        ytr = pd.read_csv(CLIENT_Y_TRAIN_FMT.format(i=i)).squeeze().to_numpy().astype(int)

        Xte = pd.read_csv(CLIENT_X_TEST_FMT.format(i=i)).values
        yte = pd.read_csv(CLIENT_Y_TEST_FMT.format(i=i)).squeeze().to_numpy().astype(int)

        clients.append({"X_train": Xtr, "y_train": ytr, "X_test": Xte, "y_test": yte})
    return clients


def compute_all_classes(client_data_list):
    ys = [d["y_train"] for d in client_data_list]
    return np.unique(np.concatenate(ys)).astype(int)


# ---------------------------
# Model init and parameter helpers
# ---------------------------
def init_mlp(n_features: int, all_classes: np.ndarray, seed: int):
    """
    Initialize MLP and create internal weights using partial_fit on a small dummy batch.
    IMPORTANT: seed must be passed here for reproducibility across runs.
    """
    mlp = MLPClassifier(
        hidden_layer_sizes=(HIDDEN_UNITS,),
        batch_size=32,
        learning_rate_init=1e-3,
        solver="adam",
        warm_start=True,
        max_iter=1,
        random_state=seed,
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
    def __init__(self, Xtr, ytr, Xte, yte, all_classes, seed: int):
        self.Xtr, self.ytr, self.Xte, self.yte = Xtr, ytr, Xte, yte
        self.ALL_CLASSES = all_classes
        self.seed = seed
        self.model = init_mlp(Xtr.shape[1], all_classes, seed)

    def get_parameters(self, config=None):
        return get_params(self.model)

    def set_parameters(self, parameters):
        set_params(self.model, parameters)

    def fit(self, parameters, config=None):
        self.set_parameters(parameters)
        local_epochs = int(config.get("local_epochs", 1)) if config else 1

        # NOTE: MLPClassifier partial_fit uses NumPy RNG internally; global seed is set per run.
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
def run_federated_sim(client_data_list, all_classes, rounds: int, local_epochs: int, ray_mem: int, seed: int):
    import ray

    ray.shutdown()
    set_global_seed(seed)  # control NumPy RNG per run

    ray_tmp = os.path.join(os.getcwd(), "ray_tmp")
    os.makedirs(ray_tmp, exist_ok=True)

    ray.init(
        local_mode=True,
        ignore_reinit_error=True,
        _temp_dir=ray_tmp,
        object_store_memory=ray_mem * 1024 * 1024,
        include_dashboard=False,
    )
    log.info(f"Ray initialized (local_mode, {ray_mem}MB object store) | seed={seed}")

    n = len(client_data_list)

    strategy = FedAvgWithStore(
        fraction_fit=1.0,
        min_fit_clients=n,
        min_available_clients=n,
        on_fit_config_fn=lambda rnd: {"local_epochs": local_epochs},
    )

    fl.simulation.start_simulation(
        client_fn=lambda cid: FraudClient(
            client_data_list[int(cid)]["X_train"],
            client_data_list[int(cid)]["y_train"],
            client_data_list[int(cid)]["X_test"],
            client_data_list[int(cid)]["y_test"],
            all_classes,
            seed,
        ).to_client(),
        num_clients=n,
        config=fl.server.ServerConfig(num_rounds=rounds),
        strategy=strategy,
        client_resources={"num_cpus": 1},
    )

    # Global evaluation set (concatenate client test sets)
    X_test = np.vstack([d["X_test"] for d in client_data_list])
    y_test = np.concatenate([d["y_test"] for d in client_data_list]).astype(int)

    # Rebuild model and load final aggregated weights
    model = init_mlp(X_test.shape[1], all_classes, seed)
    if strategy.latest_params is None:
        raise RuntimeError("No aggregated parameters were stored (strategy.latest_params is None).")

    params = parameters_to_ndarrays(strategy.latest_params)
    set_params(model, params)
    
    # EXPORT GLOBAL MODEL (NPZ)
    model_path = f"exported_models/exp2_federated_seed_{seed}.npz"
    save_mlp_npz(path=model_path, 
                 params=params, 
                 n_features=X_test.shape[1], 
                 classes=all_classes, 
                 hidden_units=HIDDEN_UNITS)

    y_pred = model.predict(X_test)
    try:
        y_prob = model.predict_proba(X_test)[:, 1]
    except Exception:
        y_prob = None

    return print_metrics_block(f"Federated Global Model (seed={seed})", y_test, y_pred, y_prob)


# ---------------------------
# Main (Multi-seed runner)
# ---------------------------
def main():
    # Decide seeds
    seeds_to_run = [FIXED_SEED] if isinstance(FIXED_SEED, int) else SEEDS

    # Load client data once (if you want each seed to re-partition clients, that should happen elsewhere)
    client_data_list = load_client_data_list(N_CLIENTS)
    all_classes = compute_all_classes(client_data_list)
    log.info(f"ALL_CLASSES: {all_classes.tolist()}")

    rows = []

    for run_idx, seed in enumerate(seeds_to_run, start=1):
        set_global_seed(seed)
        log.info(f"===== RUN {run_idx}/{len(seeds_to_run)} | seed={seed} =====")

        central_metrics = run_centralized_mlp(seed=seed)
        fed_metrics = run_federated_sim(
            client_data_list=client_data_list,
            all_classes=all_classes,
            rounds=ROUNDS,
            local_epochs=LOCAL_EPOCHS,
            ray_mem=RAY_OBJECT_STORE_MB,
            seed=seed,
        )

        print("\n=== Quick Compare (Centralized vs Federated) ===")
        print(f"[Run {run_idx} | seed={seed}] Centralized -> {quick_fmt(central_metrics)}")
        print(f"[Run {run_idx} | seed={seed}] Federated   -> {quick_fmt(fed_metrics)}")

        def add_row(model_name: str, m: dict):
            rows.append(
                {
                    "run": run_idx,
                    "seed": seed,
                    "model": model_name,
                    "acc": m["acc"],
                    "precision": m["precision"],
                    "recall": m["recall"],
                    "f1": m["f1"],
                    "auc": m["auc"],
                    "fnr": m["fnr"],
                    "fpr": m["fpr"],
                }
            )

        add_row("Centralized", central_metrics)
        add_row("Federated", fed_metrics)

    df = pd.DataFrame(rows)

    # Mean ± Std summary (per model)
    metric_cols = ["acc", "precision", "recall", "f1", "auc", "fnr", "fpr"]
    summary = df.groupby("model")[metric_cols].agg(["mean", "std"])

    print("\n=== Summary (Mean ± Std across runs) ===")
    print(summary)

    # Save outputs for thesis appendix / plotting
    out_all = "exp2_all_runs_metrics.csv" if FIXED_SEED is None else "exp1_single_seed_metrics.csv"
    out_summary = "exp2_mean_std_metrics.csv" if FIXED_SEED is None else "exp1_mean_std_metrics.csv"
    df.to_csv(out_all, index=False)
    summary.to_csv(out_summary)

    print(f"\nSaved per-run metrics to: {out_all}")
    print(f"Saved mean±std summary to: {out_summary}")
    
    # =========================
    # Auto-select best seed and copy best models for demo (EXP2)
    # =========================

    PRIMARY = "recall"    # for fraud detection (missed fraud is costly)    
    TIEBREAK_1 = "fnr"    # lower is better
    TIEBREAK_2 = "auc"    # higher is better

    def pick_best_row(df: pd.DataFrame, model_name: str) -> pd.Series:
        d = df[df["model"] == model_name].copy()
        if d.empty:
            raise ValueError(f"No rows found for model={model_name}")

        d = d.sort_values(
            by=[PRIMARY, TIEBREAK_1, TIEBREAK_2],
            ascending=[False, True, False]
        )
        return d.iloc[0]

    # Pick best seed (Federated is usually the demo focus)
    best_fed = pick_best_row(df, "Federated")

    print("\n=== EXP2 BEST SEED SELECTION ===")
    print(
        f"[Best Federated] seed={int(best_fed['seed'])}  "
        f"recall={best_fed['recall']:.4f}  fnr={best_fed['fnr']:.4f}  auc={best_fed['auc']:.4f}"
    )

    fed_src = f"exported_models/exp2_federated_seed_{int(best_fed['seed'])}.npz"
    fed_dst = "exported_models/exp2_best_federated.npz"

    if not os.path.exists(fed_src):
        print(f"[WARN] Federated model file not found: {fed_src}")
    else:
        shutil.copy2(fed_src, fed_dst)
        print(f"[OK] Copied Best Federated model → {fed_dst}")


if __name__ == "__main__":
    main()
