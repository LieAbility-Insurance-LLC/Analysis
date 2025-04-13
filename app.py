# ──────────────────────────────────────────────────────────
# Insurance‑Fraud Streamlit Dashboard · app.py
# ──────────────────────────────────────────────────────────
import io
import logging
import warnings
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from imblearn.over_sampling import SMOTE
from sklearn.exceptions import ConvergenceWarning
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split

# ──────────────────────────────────────────────────────────
# Silence noisy warnings
# ──────────────────────────────────────────────────────────
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=ConvergenceWarning)
pd.options.mode.chained_assignment = None

# ──────────────────────────────────────────────────────────
# Project‑specific imports
# ──────────────────────────────────────────────────────────
from feature_engineering import feature_engineering
from model_training import train_and_evaluate_models
from preprocessing import preprocess_data
from evaluation import explain_model_shap

# ──────────────────────────────────────────────────────────
# Utility helpers
# ──────────────────────────────────────────────────────────
MODELS_DIR = Path("models")
BEST_MODEL_PATH = MODELS_DIR / "best_rf_model.pkl"


def numeric_impute(df: pd.DataFrame) -> pd.DataFrame:
    """Coerce all columns to numeric, drop all‑NaN cols, median‑impute the rest."""
    df_num = df.apply(pd.to_numeric, errors="coerce").dropna(axis=1, how="all")
    imputer = SimpleImputer(strategy="median")
    df_imp = pd.DataFrame(imputer.fit_transform(df_num), columns=df_num.columns)
    return df_imp


@st.cache_data(show_spinner=False)
def cached_preprocess(raw: pd.DataFrame) -> pd.DataFrame:
    """Preprocess + feature engineer (cached)."""
    df = preprocess_data(raw.copy(), target_column="fraud_reported")
    df = feature_engineering(df)
    return df


def pretty_confusion(cm: np.ndarray):
    import seaborn as sns

    fig, ax = plt.subplots(figsize=(4, 3))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        cbar=False,
        ax=ax,
        linewidths=0.5,
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    st.pyplot(fig, clear_figure=True)


# ──────────────────────────────────────────────────────────
# Streamlit page config
# ──────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Insurance Fraud Dashboard",
    page_icon="🚦",
    layout="wide",
)
st.title("🚦 Insurance Claim Fraud Detection Dashboard")

# ──────────────────────────────────────────────────────────
# Sidebar: navigation
# ──────────────────────────────────────────────────────────
section = st.sidebar.radio(
    "Navigate",
    ["Upload & Preview", "Preprocess", "Train", "Predict", "Explain (SHAP)"],
)

# Session state
if "raw" not in st.session_state:
    st.session_state.raw = None
if "prep" not in st.session_state:
    st.session_state.prep = None
if "pred_df" not in st.session_state:
    st.session_state.pred_df = None

# ──────────────────────────────────────────────────────────
# 1 · Upload & Preview
# ──────────────────────────────────────────────────────────
if section == "Upload & Preview":
    st.header("1 · Upload CSV")
    file = st.file_uploader("Drop or select a CSV file", type=["csv"])
    if file:
        st.session_state.raw = pd.read_csv(file)
        st.success(f"Loaded file – shape {st.session_state.raw.shape}")
        st.dataframe(st.session_state.raw.head())

# ──────────────────────────────────────────────────────────
# 2 · Preprocess
# ──────────────────────────────────────────────────────────
elif section == "Preprocess":
    st.header("2 · Preprocessing")
    if st.session_state.raw is None:
        st.info("Please upload a dataset first.")
    else:
        if st.button("Run preprocessing"):
            with st.spinner("Processing…"):
                st.session_state.prep = cached_preprocess(st.session_state.raw)
            st.success("Done!")
        if st.session_state.prep is not None:
            st.write(f"Processed shape: {st.session_state.prep.shape}")
            st.dataframe(st.session_state.prep.head())

# ──────────────────────────────────────────────────────────
# 3 · Model Training
# ──────────────────────────────────────────────────────────
elif section == "Train":
    st.header("3 · Model Training")
    if st.session_state.prep is None:
        st.info("Run preprocessing first.")
    else:
        if st.button("Train models"):
            with st.spinner("Training… this may take a minute"):
                df = st.session_state.prep.copy().dropna(subset=["fraud_reported"])
                X = df.drop("fraud_reported", axis=1)
                y = df["fraud_reported"]

                X_imp = numeric_impute(X)
                X_bal, y_bal = SMOTE(random_state=42).fit_resample(X_imp, y)
                X_tr, X_te, y_tr, y_te = train_test_split(
                    X_bal, y_bal, test_size=0.2, random_state=42
                )

                # Capture logs
                buf = io.StringIO()
                handler = logging.StreamHandler(buf)
                logging.getLogger().addHandler(handler)

                old_show = plt.show
                plt.show = lambda *a, **kw: st.pyplot(plt.gcf(), clear_figure=True)

                train_and_evaluate_models(X_tr, X_te, y_tr, y_te)

                plt.show = old_show
                logging.getLogger().removeHandler(handler)

                st.expander("Logs").text(buf.getvalue())

                # Quick metrics for default RF
                rf_pred = joblib.load(BEST_MODEL_PATH).predict(X_te)
                acc = accuracy_score(y_te, rf_pred)
                f1 = f1_score(y_te, rf_pred)
                auc = roc_auc_score(y_te, rf_pred)
                col1, col2, col3 = st.columns(3)
                col1.metric("Accuracy", f"{acc:.3f}")
                col2.metric("F1 Score", f"{f1:.3f}")
                col3.metric("ROC AUC", f"{auc:.3f}")
                pretty_confusion(confusion_matrix(y_te, rf_pred))

# ──────────────────────────────────────────────────────────
# 4 · Prediction
# ──────────────────────────────────────────────────────────
elif section == "Predict":
    st.header("4 · Prediction")
    if st.session_state.prep is None:
        st.info("Need preprocessed data.")
    else:
        if not BEST_MODEL_PATH.exists():
            st.error(f"Pre‑trained model not found at {BEST_MODEL_PATH}")
        else:
            if st.button("Run prediction"):
                with st.spinner("Predicting…"):
                    model = joblib.load(BEST_MODEL_PATH)
                    X_pred = st.session_state.prep.copy()
                    if "fraud_reported" in X_pred.columns:
                        X_pred = X_pred.drop("fraud_reported", axis=1)
                    X_pred = numeric_impute(X_pred)
                    preds = model.predict(X_pred)
                    out = st.session_state.prep.copy()
                    out["Prediction"] = preds
                    st.session_state.pred_df = out
                st.success("Prediction complete!")

        if st.session_state.pred_df is not None:
            st.dataframe(st.session_state.pred_df.head())
            csv = st.session_state.pred_df.to_csv(index=False).encode()
            st.download_button(
                "Download predictions CSV",
                csv,
                "predictions.csv",
                "text/csv",
            )

# ──────────────────────────────────────────────────────────
# 5 · SHAP Explanation
# ──────────────────────────────────────────────────────────
elif section == "Explain (SHAP)":
    st.header("5 · Model Explainability")
    if st.session_state.prep is None:
        st.info("Need preprocessed data.")
    elif not BEST_MODEL_PATH.exists():
        st.error(f"Pre‑trained model not found at {BEST_MODEL_PATH}")
    else:
        if st.button("Generate SHAP plots"):
            with st.spinner("Calculating SHAP values…"):
                model = joblib.load(BEST_MODEL_PATH)
                X_exp = st.session_state.prep.copy()
                if "fraud_reported" in X_exp.columns:
                    X_exp = X_exp.drop("fraud_reported", axis=1)
                X_exp = numeric_impute(X_exp)

                old_show = plt.show
                plt.show = lambda *a, **kw: st.pyplot(plt.gcf(), clear_figure=True)
                explain_model_shap(model, X_exp)
                plt.show = old_show
