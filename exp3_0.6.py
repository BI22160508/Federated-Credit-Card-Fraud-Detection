# fl_full_vs_topk_with_all_metrics_multiseed_dp.py
# - Federated Learning (Full FedAvg vs Top-K sparsified)
# - Metrics: Accuracy, Precision, Recall, F1, AUC-ROC, FNR, FPR
# - Communication tracking per round (bytes sent, dense-bytes, efficiency)
# - DP-style simulation: L2 clipping + Gaussian noise on client updates
# - Master-level: Multi-seed execution + Mean±Std reporting + CSV outputs
#
# Based on your script (with fixes + multi-seed + DP on both modes).  :contentReference[oaicite:1]{index=1}

import os
import shutil
import logging
import random
from typing import List, Dict, Tuple, Optional

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

    def parameters_to_ndarrays(parameters):
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
TOPK_FRACTION = 0.20

# ---- Multi-seed ----
SEEDS = [1, 7, 13, 21, 30, 42, 56, 73, 89, 101]

# ---- DP simulation (applied to BOTH Full and Top-K) ----
ENABLE_DP = True # true to enable it
CLIP_NORM = 1.0
NOISE_MULTIPLIER = 0.6  # std = NOISE_MULTIPLIER * CLIP_NORM

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
log = logging.getLogger("fl_topk_vs_full_multiseed_dp")


# ---------------------------
# Reproducibility helpers
# ---------------------------
def set_global_seed(seed: int) -> None:
    """Control common RNG sources (note: distributed scheduling can still add nondeterminism)."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)


# ---------------------------
# Metric helpers
# ---------------------------
def compute_rates(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    fnr = fn / (tp + fn) if (tp + fn) > 0 else np.nan
    fpr = fp / (fp + tn) if (fp + tn) > 0 else np.nan
    return {"fnr": float(fnr), "fpr": float(fpr)}


def safe_auc(y_true: np.ndarray, y_prob: Optional[np.ndarray]) -> float:
    try:
        if y_prob is None:
            return float("nan")
        return float(roc_auc_score(y_true, y_prob))
    except Exception:
        return float("nan")


def summarize_metrics(y_true, y_pred, y_prob) -> Dict[str, float]:
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
    auc_txt = m["auc"]
    print(
        f"Accuracy: {m['acc']:.4f}  Precision: {m['precision']:.4f}  Recall: {m['recall']:.4f}  "
        f"F1: {m['f1']:.4f}  AUC-ROC: {auc_txt:.4f}  "
        f"FNR: {m['fnr']:.4f}  FPR: {m['fpr']:.4f}"
    )
    return m


# ---------------------------
# Client data loading
# ---------------------------
def load_client_data_list(n_clients: int) -> List[Dict[str, np.ndarray]]:
    clients = []
    for i in range(1, n_clients + 1):
        Xtr = pd.read_csv(f"client_{i}_X_train.csv").values
        ytr = pd.read_csv(f"client_{i}_y_train.csv").squeeze().to_numpy().astype(int)
        Xte = pd.read_csv(f"client_{i}_X_test_unbalanced.csv").values
        yte = pd.read_csv(f"client_{i}_y_test_unbalanced.csv").squeeze().to_numpy().astype(int)
        clients.append({"X_train": Xtr, "y_train": ytr, "X_test": Xte, "y_test": yte})
    return clients


def compute_all_classes(client_data_list):
    ys = [d["y_train"] for d in client_data_list]
    uniq = np.unique(np.concatenate(ys))
    return uniq.astype(int)


# ---------------------------
# MLP init and params helpers
# ---------------------------
def init_mlp(n_features: int, all_classes: np.ndarray, seed: int) -> MLPClassifier:
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


def get_params(model: MLPClassifier) -> List[np.ndarray]:
    return [w.copy() for w in model.coefs_] + [b.copy() for b in model.intercepts_]


def set_params(model: MLPClassifier, params: List[np.ndarray]) -> None:
    n_layers = len(model.coefs_)
    model.coefs_ = [p.copy() for p in params[:n_layers]]
    model.intercepts_ = [p.copy() for p in params[n_layers:]]


# ---------------------------
# Top-K utilities
# ---------------------------
def sparsify_topk(delta_list: List[np.ndarray], frac: float) -> List[np.ndarray]:
    if not (0.0 < frac < 1.0):
        return [d.copy() for d in delta_list]
    out = []
    for d in delta_list:
        flat = d.ravel()
        n = flat.size
        k = max(1, int(frac * n))
        if k >= n:
            out.append(d.copy())
            continue
        thr = np.partition(np.abs(flat), -k)[-k]
        mask = (np.abs(d) >= thr)
        out.append(d * mask)
    return out


# ---------------------------
# DP utilities (clip + noise)
# ---------------------------
def l2_clip(delta_list: List[np.ndarray], clip_norm: float) -> List[np.ndarray]:
    flats = [d.ravel() for d in delta_list]
    concat = np.concatenate(flats) if flats else np.array([], dtype=float)
    norm = np.linalg.norm(concat, ord=2)
    if norm == 0:
        return [d.copy() for d in delta_list]
    scale = min(1.0, clip_norm / norm)
    return [d * scale for d in delta_list]


def add_gaussian_noise(
    delta_list: List[np.ndarray],
    std: float,
    rng: np.random.Generator,
    mask_like: Optional[List[np.ndarray]] = None,
) -> List[np.ndarray]:
    """
    Add Gaussian noise.
    If mask_like provided (sparse), only add noise where mask!=0 (preserve sparsity).
    """
    out = []
    for i, d in enumerate(delta_list):
        if mask_like is not None:
            mask = (mask_like[i] != 0)
            noise = np.zeros_like(d)
            if mask.any():
                noise_vals = rng.normal(0.0, std, size=int(mask.sum()))
                noise[mask] = noise_vals
            out.append(d + noise)
        else:
            out.append(d + rng.normal(0.0, std, size=d.shape))
    return out


# ---------------------------
# Communication accounting
# ---------------------------
def count_bytes_dense(param_list: List[np.ndarray]) -> int:
    return int(sum(p.size for p in param_list) * 8)  # float64 => 8 bytes


def count_bytes_sparse(param_list: List[np.ndarray]) -> Tuple[int, int, int]:
    """
    Return (dense_bytes_possible, nnz, payload_bytes_assuming_values_only).
    NOTE: payload ignores indices for simplicity (thesis: call this an approximation).
    """
    dense_bytes = count_bytes_dense(param_list)
    nnz = int(sum((p != 0).sum() for p in param_list))
    eff_bytes = nnz * 8
    return dense_bytes, nnz, eff_bytes


# ---------------------------
# Flower Client
# ---------------------------
class FraudClient(fl.client.NumPyClient):
    def __init__(
        self,
        Xtr: np.ndarray,
        ytr: np.ndarray,
        Xte: np.ndarray,
        yte: np.ndarray,
        all_classes: np.ndarray,
        base_seed: int,
        cid: int,
        mode_tag: str,
        topk_frac: Optional[float] = None,
    ):
        self.Xtr, self.ytr, self.Xte, self.yte = Xtr, ytr, Xte, yte
        self.ALL_CLASSES = all_classes
        self.topk_frac = topk_frac
        self.base_seed = int(base_seed)
        self.cid = int(cid)
        self.mode_tag = str(mode_tag)  # "full" or "topk"
        self.model = init_mlp(n_features=Xtr.shape[1], all_classes=all_classes, seed=base_seed)

    def get_parameters(self, config=None):
        return get_params(self.model)

    def set_parameters(self, parameters):
        set_params(self.model, parameters)

    def fit(self, parameters, config=None):
        old_params = [p.copy() for p in parameters]
        self.set_parameters(parameters)

        local_epochs = int(config.get("local_epochs", 1)) if config else 1
        server_round = int(config.get("server_round", 0)) if config else 0

        # Train locally
        for _ in range(local_epochs):
            self.model.partial_fit(self.Xtr, self.ytr, classes=self.ALL_CLASSES)

        new_params = get_params(self.model)

        # Compute update Δ
        delta = [n - o for n, o in zip(new_params, old_params)]

        # DP simulation: clip + noise (deterministic per seed/round/cid/mode)
        # This makes your DP noise repeatable across multi-seed runs (important for thesis).
        if ENABLE_DP:
            delta = l2_clip(delta, CLIP_NORM)
            std = float(NOISE_MULTIPLIER * CLIP_NORM)

            mode_bit = 0 if self.mode_tag == "full" else 1
            noise_seed = self.base_seed + 1000 * server_round + 10 * self.cid + mode_bit
            rng = np.random.default_rng(noise_seed)

        if self.topk_frac is None:
            # Full: add dense noise then send dense outgoing
            if ENABLE_DP:
                delta = add_gaussian_noise(delta, std=std, rng=rng, mask_like=None)
            outgoing = [o + d for o, d in zip(old_params, delta)]

            dense_bytes = count_bytes_dense(outgoing)
            metrics = {
                "mode": "full",
                "bytes": int(dense_bytes),
                "nnz": int(sum(p.size for p in outgoing)),
                "dense_bytes": int(dense_bytes),
            }

            self.set_parameters(outgoing)
            return outgoing, len(self.Xtr), metrics

        # Top-K: sparsify then add noise ONLY on non-zero entries, send sparse outgoing
        delta_k = sparsify_topk(delta, self.topk_frac)
        if ENABLE_DP:
            delta_k = add_gaussian_noise(delta_k, std=std, rng=rng, mask_like=delta_k)

        outgoing = [o + d for o, d in zip(old_params, delta_k)]

        dense_possible, nnz, eff_bytes = count_bytes_sparse(delta_k)
        metrics = {
            "mode": "topk",
            "topk_frac": float(self.topk_frac),
            "bytes": int(eff_bytes),                 # approx payload (values only)
            "nnz": int(nnz),
            "dense_bytes": int(dense_possible),
        }

        self.set_parameters(outgoing)
        return outgoing, len(self.Xtr), metrics

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
# Strategy that stores final weights + comm/eval history
# ---------------------------
class FedAvgWithStore(FedAvg):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.latest_params = None
        self.comm_history: List[Dict[str, float]] = []
        self.eval_history: List[Dict[str, float]] = []

    def aggregate_fit(self, server_round, results, failures):
        aggregated, metrics = super().aggregate_fit(server_round, results, failures)
        if aggregated is not None:
            self.latest_params = aggregated

        # Aggregate comm metrics from clients
        round_bytes = 0
        dense_possible = 0
        for _, fit_res in results:
            m = fit_res.metrics or {}
            round_bytes += int(m.get("bytes", 0))
            dense_possible += int(m.get("dense_bytes", 0))

        # Fallback if dense_possible not provided
        if dense_possible == 0 and self.latest_params is not None:
            lst = parameters_to_ndarrays(self.latest_params)
            dense_possible = count_bytes_dense(lst)

        efficiency = 1.0 - (round_bytes / dense_possible) if dense_possible > 0 else 0.0
        self.comm_history.append(
            {
                "round": float(server_round),
                "bytes": float(round_bytes),
                "dense_possible": float(dense_possible),
                "efficiency": float(efficiency),
            }
        )
        return aggregated, metrics

    def aggregate_evaluate(self, server_round, results, failures):
        aggregated, metrics = super().aggregate_evaluate(server_round, results, failures)

        # FIX: client returns metric key "acc" (not "accuracy")
        accs, weights = [], []
        for _, ev in results:
            if ev.metrics and "acc" in ev.metrics:
                accs.append(float(ev.metrics["acc"]))
                weights.append(ev.num_examples)

        wacc = float(np.average(accs, weights=weights)) if accs else float("nan")
        self.eval_history.append({"round": float(server_round), "acc": wacc})
        return aggregated, metrics


# ---------------------------
# Per-round table printer
# ---------------------------
def print_round_table(title: str, comm_hist: List[Dict[str, float]], eval_hist: List[Dict[str, float]]):
    acc_by_round = {int(e["round"]): e.get("acc", np.nan) for e in eval_hist}
    print(f"\n--- Per-round Metrics: {title} ---")
    print(f"{'Round':>5}  {'Acc':>9}  {'Bytes Sent':>12}  {'Dense Bytes':>12}  {'Efficiency':>11}")
    for r in comm_hist:
        rnd = int(r["round"])
        acc = acc_by_round.get(rnd, np.nan)
        bytes_sent = int(r["bytes"])
        dense_b = int(r["dense_possible"])
        eff = float(r["efficiency"])
        acc_txt = f"{acc:.4f}" if not np.isnan(acc) else "  nan  "
        print(f"{rnd:5d}  {acc_txt:>9}  {bytes_sent:12d}  {dense_b:12d}  {eff:11.4f}")


# ---------------------------
# Run FL simulation and evaluate global model
# ---------------------------
def run_federated_sim(
    client_data_list: List[Dict[str, np.ndarray]],
    all_classes: np.ndarray,
    rounds: int,
    local_epochs: int,
    topk_frac: Optional[float],
    ray_mem: int,
    seed: int,
    mode_tag: str,
):
    import ray

    ray.shutdown()
    set_global_seed(seed)

    ray_tmp = os.path.join(os.getcwd(), "ray_tmp")
    os.makedirs(ray_tmp, exist_ok=True)
    ray.init(
        local_mode=True,
        ignore_reinit_error=True,
        _temp_dir=ray_tmp,
        object_store_memory=ray_mem * 1024 * 1024,
        include_dashboard=False,
    )
    log.info(f"Ray initialized (local_mode, {ray_mem}MB object store) | seed={seed} | mode={mode_tag}")

    n = len(client_data_list)

    strategy = FedAvgWithStore(
        fraction_fit=1.0,
        min_fit_clients=n,
        min_available_clients=n,
        on_fit_config_fn=lambda rnd: {"local_epochs": local_epochs, "server_round": int(rnd)},
    )

    fl.simulation.start_simulation(
        client_fn=lambda cid: FraudClient(
            client_data_list[int(cid)]["X_train"],
            client_data_list[int(cid)]["y_train"],
            client_data_list[int(cid)]["X_test"],
            client_data_list[int(cid)]["y_test"],
            all_classes,
            base_seed=seed,
            cid=int(cid),
            mode_tag=mode_tag,
            topk_frac=topk_frac,
        ).to_client(),
        num_clients=n,
        config=fl.server.ServerConfig(num_rounds=rounds),
        strategy=strategy,
        client_resources={"num_cpus": 1},
    )

    # Evaluate global model (concat all client test sets)
    X_test = np.vstack([d["X_test"] for d in client_data_list])
    y_test = np.concatenate([d["y_test"] for d in client_data_list]).astype(int)

    if strategy.latest_params is None:
        raise RuntimeError("No aggregated parameters stored: strategy.latest_params is None")

    model = init_mlp(n_features=X_test.shape[1], all_classes=all_classes, seed=seed)
    params = parameters_to_ndarrays(strategy.latest_params)
    set_params(model, params)
    
    model_path = f"exported_models/exp3_{mode_tag}_seed_{seed}_0.6.npz"

    save_mlp_npz(
        path=model_path,
        params=params,
        n_features=X_test.shape[1],
        classes=all_classes,
        hidden_units=HIDDEN_UNITS
    )

    log.info(f"[MODEL EXPORT] Saved global model to {model_path}")

    y_pred = model.predict(X_test)
    try:
        y_prob = model.predict_proba(X_test)[:, 1]
    except Exception:
        y_prob = None

    title = f"{mode_tag.upper()} Federated Model (seed={seed})"
    metrics = print_metrics_block(title, y_test, y_pred, y_prob)

    # Communication summary
    total_bytes = float(sum(r["bytes"] for r in strategy.comm_history))
    total_dense_possible = float(sum(r["dense_possible"] for r in strategy.comm_history))
    avg_eff = float(np.mean([r["efficiency"] for r in strategy.comm_history])) if strategy.comm_history else 0.0
    print(
        f"\n[COMM] {title}: bytes_sent={total_bytes:.0f}  "
        f"bytes_dense_possible={total_dense_possible:.0f}  avg_efficiency={avg_eff:.4f}"
    )

    # Per-round table
    print_round_table(title, strategy.comm_history, strategy.eval_history)

    if ENABLE_DP:
        print(f"[DP] ENABLED: clip_norm={CLIP_NORM}, noise_multiplier={NOISE_MULTIPLIER} (std={NOISE_MULTIPLIER * CLIP_NORM:.4f})")
    else:
        print("[DP] DISABLED")

    return metrics, strategy.comm_history, strategy.eval_history, {
        "total_bytes": total_bytes,
        "total_dense_possible": total_dense_possible,
        "avg_efficiency": avg_eff,
    }


# ---------------------------
# Main (Multi-seed + DP on both modes)
# ---------------------------
def main():
    client_data_list = load_client_data_list(N_CLIENTS)
    all_classes = compute_all_classes(client_data_list)
    log.info(f"ALL_CLASSES: {all_classes.tolist()}")

    final_rows: List[Dict[str, float]] = []
    round_rows: List[Dict[str, float]] = []

    for run_idx, seed in enumerate(SEEDS, start=1):
        log.info(f"===== RUN {run_idx}/{len(SEEDS)} | seed={seed} =====")

        # FULL
        full_metrics, full_comm, full_eval, full_comm_sum = run_federated_sim(
            client_data_list, all_classes, ROUNDS, LOCAL_EPOCHS, None, RAY_OBJECT_STORE_MB, seed, mode_tag="full"
        )

        # TOPK
        topk_metrics, topk_comm, topk_eval, topk_comm_sum = run_federated_sim(
            client_data_list, all_classes, ROUNDS, LOCAL_EPOCHS, TOPK_FRACTION, RAY_OBJECT_STORE_MB, seed, mode_tag="topk"
        )

        def add_final(model_name: str, m: Dict[str, float], comm_sum: Dict[str, float]):
            final_rows.append({
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
                "total_bytes": comm_sum["total_bytes"],
                "total_dense_possible": comm_sum["total_dense_possible"],
                "avg_efficiency": comm_sum["avg_efficiency"],
                "dp_enabled": int(ENABLE_DP),
                "clip_norm": float(CLIP_NORM),
                "noise_multiplier": float(NOISE_MULTIPLIER),
                "topk_fraction": float(TOPK_FRACTION) if model_name == "TopK-FL" else float("nan"),
            })

        add_final("Full-FL", full_metrics, full_comm_sum)
        add_final("TopK-FL", topk_metrics, topk_comm_sum)

        # Optional: store per-round logs for plotting
        for r in full_comm:
            round_rows.append({
                "run": run_idx, "seed": seed, "model": "Full-FL",
                "round": int(r["round"]), "bytes": float(r["bytes"]),
                "dense_possible": float(r["dense_possible"]), "efficiency": float(r["efficiency"])
            })
        for e in full_eval:
            round_rows.append({
                "run": run_idx, "seed": seed, "model": "Full-FL",
                "round": int(e["round"]), "acc_round": float(e.get("acc", np.nan))
            })
        for r in topk_comm:
            round_rows.append({
                "run": run_idx, "seed": seed, "model": "TopK-FL",
                "round": int(r["round"]), "bytes": float(r["bytes"]),
                "dense_possible": float(r["dense_possible"]), "efficiency": float(r["efficiency"])
            })
        for e in topk_eval:
            round_rows.append({
                "run": run_idx, "seed": seed, "model": "TopK-FL",
                "round": int(e["round"]), "acc_round": float(e.get("acc", np.nan))
            })

    df_final = pd.DataFrame(final_rows)
    metric_cols = ["acc", "precision", "recall", "f1", "auc", "fnr", "fpr",
                   "total_bytes", "total_dense_possible", "avg_efficiency"]
    df_summary = df_final.groupby("model")[metric_cols].agg(["mean", "std"])

    df_final.to_csv("exp3_full_vs_topk_dp_all_runs_dp_0.6.csv", index=False)
    df_summary.to_csv("exp3_full_vs_topk_dp_mean_std_dp_0.6.csv")

    # per-round export
    if round_rows:
        pd.DataFrame(round_rows).to_csv("exp3_full_vs_topk_dp_per_round_logs_dp_0.6.csv", index=False)

    print("\n=== FINAL (Per-run) saved to: exp3_full_vs_topk_dp_all_runs.csv ===")
    print("=== SUMMARY (Mean±Std) saved to: exp3_full_vs_topk_dp_mean_std.csv ===")
    print("\n=== Mean ± Std===")
    print(df_summary)

    print("\n=== DP Simulation ===")
    if ENABLE_DP:
        print(f"ENABLED (clip_norm={CLIP_NORM}, noise_multiplier={NOISE_MULTIPLIER}, std={NOISE_MULTIPLIER * CLIP_NORM:.4f})")
        print("NOTE: This is a DP-style simulation (clip + noise). A formal ε,δ requires accounting.")
    else:
        print("DISABLED (set ENABLE_DP=True to enable clipping + Gaussian noise)")    
    # =========================
    # Auto-select best seed and copy best models for demo
    # =========================

    # Choose what "best" means:
    # Recommended for fraud detection: maximize recall, tie-break by minimizing FNR, then maximize AUC.
    PRIMARY = "recall"        # or "f1" / "auc"
    TIEBREAK_1 = "fnr"        # lower is better
    TIEBREAK_2 = "auc"        # higher is better

    def pick_best_row(df: pd.DataFrame, model_name: str) -> pd.Series:
        d = df[df["model"] == model_name].copy()
        if d.empty:
            raise ValueError(f"No rows found for model={model_name}")

        # sort: primary desc, fnr asc, auc desc
        d = d.sort_values(
            by=[PRIMARY, TIEBREAK_1, TIEBREAK_2],
            ascending=[False, True, False]
        )
        return d.iloc[0]
    
    # Pick best seed for each model
    best_full = pick_best_row(df_final, "Full-FL")
    best_topk = pick_best_row(df_final, "TopK-FL")

    print("\n=== BEST SEED SELECTION ===")
    print(f"[Best Full-FL] seed={int(best_full['seed'])}  recall={best_full['recall']:.4f}  fnr={best_full['fnr']:.4f}  auc={best_full['auc']:.4f}")
    print(f"[Best TopK-FL] seed={int(best_topk['seed'])}  recall={best_topk['recall']:.4f}  fnr={best_topk['fnr']:.4f}  auc={best_topk['auc']:.4f}")

    # Copy the exported NPZ models into clean “demo” filenames
    full_src = f"exported_models/exp3_full_seed_{int(best_full['seed'])}_0.6.npz"
    topk_src = f"exported_models/exp3_topk_seed_{int(best_topk['seed'])}_0.6.npz"

    full_dst = "exported_models/exp3_best_fullfl_0.6.npz"
    topk_dst = "exported_models/exp3_best_topkfl_0.6.npz"
    
    print(f"[DEBUG] full_src={full_src}")
    print(f"[DEBUG] topk_src={topk_src}")

    if not os.path.exists(full_src):
        print(f"[WARN] Full-FL model file not found: {full_src}")
    else:
        shutil.copy2(full_src, full_dst)
        print(f"[OK] Copied Best Full-FL model → {full_dst}")

    if not os.path.exists(topk_src):
        print(f"[WARN] TopK-FL model file not found: {topk_src}")
    else:
        shutil.copy2(topk_src, topk_dst)
        print(f"[OK] Copied Best TopK-FL model → {topk_dst}")



if __name__ == "__main__":
    main()
