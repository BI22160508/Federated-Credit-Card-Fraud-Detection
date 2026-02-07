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
    "FPR",
    "Comm. Efficiency"
]

fullfl_mean = np.array([
    0.9794,
    0.1193,
    0.6469,
    0.1660,
    0.8791,
    0.3531,
    0.0200,
    0.0000
])

fullfl_std = np.array([
    0.0159,
    0.1625,
    0.1336,
    0.1538,
    0.0440,
    0.1336,
    0.0159,
    0.0000
])

topk_mean = np.array([
    0.9976,
    0.4392,
    0.5684,
    0.4736,
    0.9112,
    0.4316,
    0.0017,
    0.8006
])

topk_std = np.array([
    0.0015,
    0.1929,
    0.1742,
    0.1773,
    0.0269,
    0.1742,
    0.0014,
    0.0000
])

# =========================
# Plot
# =========================

x = np.arange(len(metrics))
width = 0.35

plt.figure(figsize=(11, 5))

plt.bar(
    x - width / 2,
    fullfl_mean,
    width,
    yerr=fullfl_std,
    capsize=5,
    label="Full-FL"
)

plt.bar(
    x + width / 2,
    topk_mean,
    width,
    yerr=topk_std,
    capsize=5,
    label="TopK-FL"
)

plt.ylabel("Score / Ratio")
plt.title("Full-FL vs TopK-FL Performance Under Non-IID Conditions (Mean ± Std)")
plt.xticks(x, metrics, rotation=30)
plt.ylim(0.00, 1.05)
plt.legend()
plt.grid(axis="y", linestyle="--", alpha=0.6)
plt.tight_layout()

# Save figure (high resolution)
plt.savefig(
    "Figure_5_X_FullFL_vs_TopKFL_Performance_and_Communication_NonIID_0.4.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()


# =========================
# Line plot across runs
# =========================
runs = np.arange(1, 11)

# =========================
# FNR values (Noise = 0.4)
# =========================
fnr_fullfl = np.array([
    0.2347, 0.6224, 0.2551, 0.2449, 0.3571,
    0.2245, 0.4898, 0.2755, 0.3673, 0.4592
])

fnr_topkfl = np.array([
    0.5816, 0.6633, 0.3878, 0.3571, 0.2755,
    0.3776, 0.3163, 0.7653, 0.2755, 0.3163
])

# =========================
# Plot
# =========================
plt.figure(figsize=(9, 5))

plt.plot(runs, fnr_fullfl, marker='o', linewidth=2, label="Full-FL (DP = 0.4)")
plt.plot(runs, fnr_topkfl, marker='s', linewidth=2, label="TopK-FL (DP = 0.4)")

plt.xlabel("Execution Run")
plt.ylabel("False Negative Rate (FNR)")
plt.title("Run-wise False Negative Rate under Differential Privacy (Noise = 0.4)")
plt.xticks(runs)
plt.ylim(0, 1.0)
plt.grid(True, linestyle="--", alpha=0.6)
plt.legend()
plt.tight_layout()

# Save high-resolution figure
plt.savefig(
    "Figure_6_X_Runwise_FNR_Comparison_DP_0_4.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()

