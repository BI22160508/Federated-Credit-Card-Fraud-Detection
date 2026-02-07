import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, roc_curve
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

from fl_model_io import load_mlp_npz, save_mlp_npz


# ======================================================
# Page config
# ======================================================
st.set_page_config(page_title="Federated Fraud Model UI", layout="wide")
st.title("Federated Fraud Detection — Train / Evaluate / Predict UI")
st.caption("Examiner demo: upload CSV → (optional) load exported FL model → evaluate or flag fraud")

os.makedirs("exported_models", exist_ok=True)


# ======================================================
# Modes
# ======================================================
mode = st.radio(
    "Select Mode",
    [
        "Train model (demo)",
        "Evaluate model (with labels)",
        "Predict only (no labels)"
    ],
    horizontal=True
)


# ======================================================
# Helpers
# ======================================================
def detect_label_column(df: pd.DataFrame):
    for c in ["Class", "class", "label", "target", "y"]:
        if c in df.columns:
            return c
    return None

def safe_predict_proba(model: MLPClassifier, X: np.ndarray):
    try:
        return model.predict_proba(X)[:, 1]
    except Exception:
        return None

def compute_fnr_fpr(y_true: np.ndarray, y_pred: np.ndarray):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    fnr = fn / (tp + fn) if (tp + fn) > 0 else np.nan
    fpr = fp / (fp + tn) if (fp + tn) > 0 else np.nan
    return fnr, fpr, (tn, fp, fn, tp)

def build_model_from_npz(model_file):
    """
    Load .npz and return a ready sklearn MLPClassifier with weights set.
    """
    params, n_features, all_classes, hidden_units = load_mlp_npz(model_file)

    model = MLPClassifier(
        hidden_layer_sizes=(int(hidden_units),),
        batch_size=32,
        learning_rate_init=1e-3,
        solver="adam",
        warm_start=True,
        max_iter=1,
        random_state=30,
    )

    # Seed-fit so sklearn creates correct arrays
    X_seed = np.zeros((len(all_classes), n_features))
    y_seed = all_classes.copy()
    model.partial_fit(X_seed, y_seed, classes=all_classes)

    # Set weights
    n_layers = len(model.coefs_)
    model.coefs_ = [p.copy() for p in params[:n_layers]]
    model.intercepts_ = [p.copy() for p in params[n_layers:]]

    return model, n_features, all_classes, hidden_units

def show_eval(y_true: np.ndarray, y_pred: np.ndarray, y_prob=None):
    acc = accuracy_score(y_true, y_pred)
    pre = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)

    auc = np.nan
    if y_prob is not None:
        try:
            auc = roc_auc_score(y_true, y_prob)
        except Exception:
            pass

    fnr, fpr, (tn, fp, fn, tp) = compute_fnr_fpr(y_true, y_pred)

    st.subheader("Evaluation Metrics")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Accuracy", f"{acc:.4f}")
    c2.metric("Precision", f"{pre:.4f}")
    c3.metric("Recall", f"{rec:.4f}")
    c4.metric("F1-score", f"{f1:.4f}")

    c5, c6, c7 = st.columns(3)
    c5.metric("AUC-ROC", f"{auc:.4f}" if not np.isnan(auc) else "N/A")
    c6.metric("FNR", f"{fnr:.4f}")
    c7.metric("FPR", f"{fpr:.4f}")

    st.caption(f"Confusion Matrix: TN={tn}, FP={fp}, FN={fn}, TP={tp}")

    st.subheader("Confusion Matrix")
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    fig, ax = plt.subplots(figsize=(4.6, 4.6))
    ax.imshow(cm)
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["Legit", "Fraud"])
    ax.set_yticklabels(["Legit", "Fraud"])
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    for (i, j), v in np.ndenumerate(cm):
        ax.text(j, i, str(v), ha="center", va="center")
    st.pyplot(fig)

    st.subheader("ROC Curve")
    if y_prob is None:
        st.info("ROC curve unavailable (no predict_proba).")
    else:
        fpr_c, tpr_c, _ = roc_curve(y_true, y_prob)
        fig, ax = plt.subplots(figsize=(6.0, 4.2))
        ax.plot(fpr_c, tpr_c)
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.set_title("ROC Curve")
        ax.grid(True, linestyle="--", alpha=0.6)
        st.pyplot(fig)


# ======================================================
# Sidebar inputs
# ======================================================
st.sidebar.header("Inputs")
csv_file = st.sidebar.file_uploader("Upload CSV (features + optional label)", type=["csv"])
threshold = st.sidebar.slider("Fraud threshold", 0.01, 0.99, 0.50, 0.01)
preview_rows = st.sidebar.number_input("Preview rows", 5, 200, 30, 5)

model_file = None
if mode in ["Evaluate model (with labels)", "Predict only (no labels)"]:
    st.sidebar.header("Exported Model")
    model_file = st.sidebar.file_uploader("Upload exported model (.npz)", type=["npz"])

if csv_file is None:
    st.info("Upload a CSV file to proceed.")
    st.stop()

df = pd.read_csv(csv_file)
label_col = detect_label_column(df)


# ======================================================
# MODE 1: Train model (demo) — Option 3 (spinner + progress)
# ======================================================
if mode == "Train model (demo)":
    st.subheader("Train Model (Demo)")
    st.caption(
        "This demo trains a local MLP to show an end-to-end pipeline. "
        "Your thesis results should still use exported FL global models."
    )

    if label_col is None:
        st.error("Training requires a label column (e.g., 'Class').")
        st.stop()

    X = df.drop(columns=[label_col]).values
    y = df[label_col].values.astype(int)

    test_size = st.sidebar.slider("Test split", 0.1, 0.5, 0.2, 0.05)
    hidden_units = st.sidebar.selectbox("Hidden units", [32, 64, 128], index=1)
    epochs = st.sidebar.slider("Training epochs", 1, 50, 10, 1)
    lr = st.sidebar.selectbox("Learning rate", [1e-4, 5e-4, 1e-3, 2e-3], index=2)
    seed = st.sidebar.number_input("Random seed", 1, 9999, 30, 1)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=float(test_size), random_state=int(seed), stratify=y
    )

    scaler = MinMaxScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    if st.button("Train Now"):
        with st.spinner("Training model..."):
            progress = st.progress(0)
            status = st.empty()

            model = MLPClassifier(
                hidden_layer_sizes=(int(hidden_units),),
                batch_size=32,
                learning_rate_init=float(lr),
                solver="adam",
                max_iter=1,
                warm_start=True,
                random_state=int(seed),
            )

            # ✅ SAFE initialization: guarantee BOTH classes exist
            X_init = np.vstack([X_train[0], X_train[0]])
            y_init = np.array([0, 1])
            model.partial_fit(X_init, y_init, classes=np.array([0, 1]))

            for i in range(int(epochs)):
                model.partial_fit(X_train, y_train)
                progress.progress((i + 1) / int(epochs))
                status.text(f"Training epoch {i + 1}/{epochs}")

        status.success("Training completed successfully ✅")
        progress.empty()

        y_prob = safe_predict_proba(model, X_test)
        y_pred = model.predict(X_test) if y_prob is None else (y_prob >= threshold).astype(int)

        show_eval(y_test, y_pred, y_prob)

        st.subheader("Export Demo-Trained Model (Optional)")
        export_name = st.text_input("Export filename", value="exported_models/demo_trained_mlp.npz")
        if st.button("Export Demo Model"):
            params = model.coefs_ + model.intercepts_
            save_mlp_npz(
                path=export_name,
                params=params,
                n_features=X.shape[1],
                classes=np.array([0, 1]),
                hidden_units=int(hidden_units)
            )
            st.success(f"Saved: {export_name}")

    st.stop()


# ======================================================
# MODE 2: Evaluate model (with labels)
# ======================================================
if mode == "Evaluate model (with labels)":
    if model_file is None:
        st.info("Upload an exported model (.npz) in the sidebar.")
        st.stop()

    if label_col is None:
        st.error("Evaluation requires a label column (e.g., 'Class').")
        st.stop()

    X = df.drop(columns=[label_col]).values
    y_true = df[label_col].values.astype(int)

    model, n_features, _, _ = build_model_from_npz(model_file)

    if X.shape[1] != n_features:
        st.error(f"Feature mismatch: CSV has {X.shape[1]} but model expects {n_features}")
        st.stop()

    y_prob = safe_predict_proba(model, X)
    y_pred = model.predict(X) if y_prob is None else (y_prob >= threshold).astype(int)

    st.subheader("Evaluation Results (Exported Model)")
    show_eval(y_true, y_pred, y_prob)

    out = df.copy()
    out["Pred_Label"] = y_pred
    if y_prob is not None:
        out["Pred_Prob_Fraud"] = y_prob

    st.download_button(
        "Download evaluated predictions CSV",
        out.to_csv(index=False).encode("utf-8"),
        "evaluation_predictions.csv",
        "text/csv"
    )

    st.stop()


# ======================================================
# MODE 3: Predict only (no labels) — show possible fraud only
# ======================================================
if mode == "Predict only (no labels)":
    if model_file is None:
        st.info("Upload an exported model (.npz) in the sidebar.")
        st.stop()

    X = df.values if label_col is None else df.drop(columns=[label_col]).values

    model, n_features, _, _ = build_model_from_npz(model_file)

    if X.shape[1] != n_features:
        st.error(f"Feature mismatch: CSV has {X.shape[1]} but model expects {n_features}")
        st.stop()

    y_prob = safe_predict_proba(model, X)
    y_pred = model.predict(X) if y_prob is None else (y_prob >= threshold).astype(int)

    out = df.copy()
    if y_prob is not None:
        out["Pred_Prob_Fraud"] = y_prob
    out["Pred_Label"] = y_pred

    st.subheader("Possible Fraud (Flagged Only)")
    flagged = out[out["Pred_Label"] == 1].copy()

    st.caption(f"Total records: {len(out)} | Flagged fraud: {len(flagged)}")
    st.dataframe(flagged.head(preview_rows), use_container_width=True)

    st.download_button(
        "Download flagged fraud records (CSV)",
        flagged.to_csv(index=False).encode("utf-8"),
        "possible_fraud_only.csv",
        "text/csv"
    )

    st.stop()
