"""
BITS Pilani M.Tech AIML - ML Assignment 2
Streamlit App: Heart Disease Classification
"""

import os
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import streamlit as st

from sklearn.metrics import (
    accuracy_score, roc_auc_score, precision_score,
    recall_score, f1_score, matthews_corrcoef,
    confusion_matrix, classification_report
)

warnings.filterwarnings("ignore")

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Heart Disease Classifier",
    page_icon="🫀",
    layout="wide",
)

# ── Constants ──────────────────────────────────────────────────────────────────
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR  = os.path.join(BASE_DIR, "model")

MODEL_OPTIONS = {
    "Logistic Regression":    "lr_model.joblib",
    "Decision Tree":          "dt_model.joblib",
    "K-Nearest Neighbor":     "knn_model.joblib",
    "Naive Bayes":            "nb_model.joblib",
    "Support Vector Machine": "svm_model.joblib",
}

TARGET_CANDIDATES = ["target", "condition", "num", "diagnosis", "output"]


# ── Helpers ────────────────────────────────────────────────────────────────────
@st.cache_resource
def load_scaler():
    path = os.path.join(MODEL_DIR, "scaler.joblib")
    if not os.path.exists(path):
        st.error("scaler.joblib not found. Run train_models.py first.")
        st.stop()
    return joblib.load(path)


@st.cache_resource
def load_model(model_name: str):
    filename = MODEL_OPTIONS[model_name]
    path = os.path.join(MODEL_DIR, filename)
    if not os.path.exists(path):
        st.error(f"{filename} not found. Run train_models.py first.")
        st.stop()
    return joblib.load(path)


def get_target_col(df: pd.DataFrame) -> str:
    for col in TARGET_CANDIDATES:
        if col in df.columns:
            return col
    return df.columns[-1]


def compute_metrics(y_true, y_pred, y_prob) -> dict:
    return {
        "Accuracy":  round(accuracy_score(y_true, y_pred), 4),
        "AUC Score": round(roc_auc_score(y_true, y_prob), 4),
        "Precision": round(precision_score(y_true, y_pred, zero_division=0), 4),
        "Recall":    round(recall_score(y_true, y_pred, zero_division=0), 4),
        "F1 Score":  round(f1_score(y_true, y_pred, zero_division=0), 4),
        "MCC":       round(matthews_corrcoef(y_true, y_pred), 4),
    }


def plot_confusion_matrix(y_true, y_pred, model_name: str):
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=["No Disease", "Disease"],
        yticklabels=["No Disease", "Disease"],
        ax=ax, linewidths=0.5
    )
    ax.set_title(f"Confusion Matrix — {model_name}", fontsize=12, fontweight="bold")
    ax.set_xlabel("Predicted Label", fontsize=10)
    ax.set_ylabel("True Label", fontsize=10)
    plt.tight_layout()
    return fig


def metric_color(value: float, metric: str) -> str:
    """Return a colour class string based on metric value."""
    if metric == "MCC":
        if value >= 0.6:  return "🟢"
        if value >= 0.3:  return "🟡"
        return "🔴"
    if value >= 0.85:  return "🟢"
    if value >= 0.70:  return "🟡"
    return "🔴"


# ── App Layout ─────────────────────────────────────────────────────────────────
st.title("🫀 Heart Disease Classification")
st.markdown(
    "**BITS Pilani M.Tech AIML — ML Assignment 2**  \n"
    "Upload your test CSV, select a model, and evaluate performance."
)
st.divider()

# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Controls")

    st.markdown("### 1. Upload Test Data (CSV)")
    uploaded_file = st.file_uploader(
        "Upload test_data.csv",
        type=["csv"],
        help="CSV must contain all 13 feature columns and the target column.",
    )

    st.markdown("### 2. Select Model")
    selected_model = st.selectbox(
        "Choose a classifier",
        list(MODEL_OPTIONS.keys()),
        help="All models were trained on the same dataset.",
    )

    run_button = st.button("▶ Run Evaluation", type="primary", use_container_width=True)

    st.divider()
    st.markdown("#### About")
    st.markdown(
        "Dataset: Heart Disease (UCI/Cleveland)  \n"
        "Features: 13  |  Instances: 1,025  \n"
        "Task: Binary Classification  \n"
        "Target: 0 = No Disease, 1 = Disease"
    )

# ── Main Panel ─────────────────────────────────────────────────────────────────
if not uploaded_file:
    st.info(
        "👈 Upload `test_data.csv` from the sidebar and click **Run Evaluation** to begin.  \n\n"
        "You can download `test_data.csv` from the GitHub repository."
    )
    st.stop()

# Load and validate the uploaded CSV
try:
    test_df = pd.read_csv(uploaded_file)
except Exception as e:
    st.error(f"Could not read CSV: {e}")
    st.stop()

target_col = get_target_col(test_df)

if test_df.shape[1] < 2:
    st.error("CSV must have at least feature columns and a target column.")
    st.stop()

# Drop non-numeric, impute NaNs, binarise target
test_df.replace("?", np.nan, inplace=True)
for col in test_df.columns:
    test_df[col] = pd.to_numeric(test_df[col], errors="coerce")
    if test_df[col].isnull().any():
        test_df[col].fillna(test_df[col].mode()[0], inplace=True)

test_df[target_col] = (test_df[target_col] > 0).astype(int)

X_test = test_df.drop(columns=[target_col])
y_test = test_df[target_col]

# ── Data Preview ───────────────────────────────────────────────────────────────
with st.expander("📋 Uploaded Data Preview", expanded=False):
    col1, col2, col3 = st.columns(3)
    col1.metric("Rows",     test_df.shape[0])
    col2.metric("Features", X_test.shape[1])
    col3.metric("Target: Disease %", f"{y_test.mean()*100:.1f}%")
    st.dataframe(test_df.head(10), use_container_width=True)

if not run_button:
    st.info("Click **▶ Run Evaluation** in the sidebar to see results.")
    st.stop()

# ── Load model & scaler ────────────────────────────────────────────────────────
with st.spinner(f"Loading {selected_model} …"):
    scaler = load_scaler()
    model  = load_model(selected_model)

X_test_sc = scaler.transform(X_test)
y_pred    = model.predict(X_test_sc)
y_prob    = model.predict_proba(X_test_sc)[:, 1]

metrics = compute_metrics(y_test, y_pred, y_prob)

# ── Results ────────────────────────────────────────────────────────────────────
st.subheader(f"📊 Results — {selected_model}")

# Metric cards
cols = st.columns(6)
metric_names = list(metrics.keys())
for i, (col, mname) in enumerate(zip(cols, metric_names)):
    val = metrics[mname]
    icon = metric_color(val, mname)
    col.metric(label=f"{icon} {mname}", value=f"{val:.4f}")

st.divider()

# Confusion matrix + classification report side by side
col_cm, col_cr = st.columns([1, 1])

with col_cm:
    st.markdown("#### Confusion Matrix")
    fig = plot_confusion_matrix(y_test, y_pred, selected_model)
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)

with col_cr:
    st.markdown("#### Classification Report")
    report = classification_report(
        y_test, y_pred,
        target_names=["No Disease", "Disease"],
        output_dict=True
    )
    report_df = pd.DataFrame(report).T.round(4)
    st.dataframe(report_df, use_container_width=True)

st.divider()

# ── All-models comparison (if model_metrics.csv exists) ───────────────────────
metrics_path = os.path.join(BASE_DIR, "model_metrics.csv")
if os.path.exists(metrics_path):
    st.subheader("📈 All Models Comparison")
    all_metrics = pd.read_csv(metrics_path)

    # Highlight current model
    def highlight_selected(row):
        if row["ML Model Name"] == selected_model:
            return ["background-color: #d4edda"] * len(row)
        return [""] * len(row)

    styled = all_metrics.style.apply(highlight_selected, axis=1).format(
        {c: "{:.4f}" for c in all_metrics.columns if c != "ML Model Name"}
    )
    st.dataframe(styled, use_container_width=True, hide_index=True)

    # Bar chart
    st.markdown("#### Accuracy Comparison")
    fig2, ax = plt.subplots(figsize=(9, 4))
    colors = ["#2196F3" if m != selected_model else "#4CAF50"
              for m in all_metrics["ML Model Name"]]
    bars = ax.bar(all_metrics["ML Model Name"], all_metrics["Accuracy"], color=colors, edgecolor="white")
    ax.bar_label(bars, fmt="%.4f", padding=3, fontsize=9)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Accuracy")
    ax.set_title("Model Accuracy on Test Set (green = currently selected)")
    plt.xticks(rotation=15, ha="right")
    plt.tight_layout()
    st.pyplot(fig2, use_container_width=True)
    plt.close(fig2)

st.divider()
st.caption(
    "BITS Pilani WILP — M.Tech AIML — ML Assignment 2  |  "
    "Dataset: Heart Disease (UCI/Cleveland)  |  "
    "Models: LR · DT · KNN · NB · SVM"
)
