import numpy as np
import matplotlib.pyplot as plt

# =========================
# Metrics (Mean ± Std)
# =========================

metrics = [
    "Accuracy",
    "Precision",
    "Recall",
    "F1-score",
    "AUC-ROC",
    "FNR",
    "FPR"
]

centralized_mean = np.array([
    0.8776,
    1.0000,
    0.7551,
    0.8605,
    0.9394,
    0.2449,
    0.0000
])

centralized_std = np.array([
    0.0000,
    0.0000,
    0.0000,
    0.0000,
    0.0000,
    0.0000,
    0.0000
])

federated_mean = np.array([
    0.8830,
    1.0000,
    0.7674,
    0.8680,
    0.9381,
    0.2326,
    0.0000
])

federated_std = np.array([
    0.0037,
    0.0000,
    0.0067,
    0.0047,
    0.0017,
    0.0067,
    0.0000
])

# =========================
# Plot
# =========================

x = np.arange(len(metrics))
width = 0.35

plt.figure(figsize=(10, 5))

plt.bar(
    x - width / 2,
    centralized_mean,
    width,
    yerr=centralized_std,
    capsize=5,
    label="Centralized"
)

plt.bar(
    x + width / 2,
    federated_mean,
    width,
    yerr=federated_std,
    capsize=5,
    label="Federated"
)

plt.ylabel("Score")
plt.title("Centralized vs Federated Learning Performance Stability (Mean ± Std)")
plt.xticks(x, metrics, rotation=30)
plt.ylim(0.20, 1.02)
plt.legend()
plt.grid(axis="y", linestyle="--", alpha=0.6)
plt.tight_layout()

# Save figure (high resolution)
plt.savefig(
    "Figure_4_X_Centralized_vs_Federated_Metric_Stability_exp1.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()


# =========================
# Line plot across runs
# =========================
runs = np.arange(1, 11)

Cen_fnr = np.array([
    0.2449, 0.2449, 0.2449, 0.2449, 0.2449,
    0.2449, 0.2449, 0.2449, 0.2449, 0.2449
])

Fed_fnr = np.array([
    0.2347, 0.2347, 0.2347, 0.2245, 0.2449,
    0.2347, 0.2347, 0.2347, 0.2347, 0.2245
])

# =========================
# Plot
# =========================
plt.figure(figsize=(10, 5))

plt.plot(runs, Cen_fnr, marker='o', linewidth=2, label="Full-FL")
plt.plot(runs, Fed_fnr, marker='s', linestyle='--', linewidth=2, label="TopK-FL")

# Reference threshold
plt.axhline(
    y=0.25,
    linestyle=':',
    linewidth=2,
    label="Reference FNR Threshold (0.25)"
)

plt.xlabel("Execution Run")
plt.ylabel("False Negative Rate (FNR)")
plt.title("FNR Stability Across Runs (Full-FL vs TopK-FL, No DP)")
plt.xticks(runs)
plt.ylim(0.18, 0.30)
plt.legend()
plt.grid(True, linestyle="--", alpha=0.6)
plt.tight_layout()

plt.savefig(
    "Figure_6_X_FNR_Stability_FullFL_vs_TopKFL_NoDP.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()