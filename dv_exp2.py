import numpy as np
import matplotlib.pyplot as plt

# =========================
# Metrics (Mean ± Std) - Next Experiment
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
    0.9993,
    0.7948,
    0.7689,
    0.7807,
    0.9449,
    0.2311,
    0.0004
])

centralized_std = np.array([
    0.00006,
    0.0372,
    0.0147,
    0.0153,
    0.0043,
    0.0147,
    0.00010
])

federated_mean = np.array([
    0.9994,
    0.8338,
    0.7704,
    0.8018,
    0.9470,
    0.2296,
    0.0003
])

federated_std = np.array([
    0.00006,
    0.0353,
    0.0105,
    0.0176,
    0.0060,
    0.0105,
    0.00007
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
plt.ylim(0.00, 1.02)
plt.legend()
plt.grid(axis="y", linestyle="--", alpha=0.6)
plt.tight_layout()

# Save figure (high resolution)
plt.savefig(
    "Figure_4_X_Centralized_vs_Federated_Metric_Stability_exp2.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()


# =========================
# Line plot across runs
# =========================
runs = np.arange(1, 11)

Cen_fnr = np.array([
    0.2143, 0.2245, 0.2449, 0.2449, 0.2449,
    0.2449, 0.2143, 0.2347, 0.2449, 0.2347
])

Fed_fnr = np.array([
    0.2347, 0.2347, 0.2143, 0.2347, 0.2449,
    0.2143, 0.2347, 0.2245, 0.2449, 0.2347
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
