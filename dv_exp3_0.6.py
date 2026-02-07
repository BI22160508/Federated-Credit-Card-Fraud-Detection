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
    0.9566,
    0.0674,
    0.6500,
    0.1002,
    0.8521,
    0.3500,
    0.0429,
    0.0000
])

fullfl_std = np.array([
    0.0317,
    0.1112,
    0.1472,
    0.1267,
    0.0698,
    0.1472,
    0.0317,
    0.0000
])

topk_mean = np.array([
    0.9744,
    0.0567,
    0.5776,
    0.0984,
    0.8645,
    0.4224,
    0.0249,
    0.8006
])

topk_std = np.array([
    0.0201,
    0.0384,
    0.1927,
    0.0622,
    0.0661,
    0.1927,
    0.0202,
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
    "Figure_5_X_FullFL_vs_TopKFL_Performance_and_Communication_0.6.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()

import numpy as np
import matplotlib.pyplot as plt

# =========================
# Line plot across runs
# =========================
runs = np.arange(1, 11)

# =========================
# FNR values (Noise = 0.6)
# =========================
fnr_fullfl = np.array([
    0.2551, 0.6939, 0.2143, 0.2551, 0.2959,
    0.2245, 0.4286, 0.2959, 0.3776, 0.4592
])

fnr_topkfl = np.array([
    0.9184, 0.3265, 0.4286, 0.2755, 0.5000,
    0.3367, 0.2551, 0.4082, 0.4694, 0.3061
])

# =========================
# Plot
# =========================
plt.figure(figsize=(9, 5))

plt.plot(runs, fnr_fullfl, marker='o', linewidth=2, label="Full-FL (DP = 0.6)")
plt.plot(runs, fnr_topkfl, marker='s', linewidth=2, label="TopK-FL (DP = 0.6)")

plt.xlabel("Execution Run")
plt.ylabel("False Negative Rate (FNR)")
plt.title("Run-wise False Negative Rate under Strong Differential Privacy (Noise = 0.6)")
plt.xticks(runs)
plt.ylim(0, 1.0)
plt.grid(True, linestyle="--", alpha=0.6)
plt.legend()
plt.tight_layout()

# Save figure
plt.savefig(
    "Figure_6_X_Runwise_FNR_DP_0_6.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()
