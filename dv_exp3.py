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
    0.9994,
    0.8436,
    0.7684,
    0.8036,
    0.9467,
    0.2316,
    0.0003,
    0.0000
])

fullfl_std = np.array([
    0.0001,
    0.0418,
    0.0137,
    0.0162,
    0.0052,
    0.0137,
    0.0001,
    0.0000
])

topk_mean = np.array([
    0.9994,
    0.8548,
    0.7592,
    0.8037,
    0.9520,
    0.2408,
    0.0002,
    0.8006
])

topk_std = np.array([
    0.0001,
    0.0371,
    0.0099,
    0.0148,
    0.0037,
    0.0099,
    0.0001,
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
plt.title("Full-FL vs TopK-FL Performance and Communication Efficiency (Mean ± Std)")
plt.xticks(x, metrics, rotation=30)
plt.ylim(0.00, 1.02)
plt.legend()
plt.grid(axis="y", linestyle="--", alpha=0.6)
plt.tight_layout()

# Save figure (high resolution)
plt.savefig(
    "Figure_5_X_FullFL_vs_TopKFL_Performance_and_Communication.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()

# =========================
# Line plot across runs
# =========================
runs = np.arange(1, 11)

fullfl_fnr = np.array([
    0.2347, 0.2245, 0.2449, 0.2143, 0.2245,
    0.2143, 0.2347, 0.2245, 0.2449, 0.2551
])

topk_fnr = np.array([
    0.2245, 0.2347, 0.2449, 0.2551, 0.2449,
    0.2551, 0.2449, 0.2347, 0.2347, 0.2347
])

# =========================
# Plot
# =========================
plt.figure(figsize=(10, 5))

plt.plot(runs, fullfl_fnr, marker='o', linewidth=2, label="Full-FL")
plt.plot(runs, topk_fnr, marker='s', linestyle='--', linewidth=2, label="TopK-FL")

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
