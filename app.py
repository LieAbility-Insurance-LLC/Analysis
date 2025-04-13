# ──────────────────────────────────────────────────────────
# Insurance‑Fraud Streamlit Dashboard · app.py  (v2.0)
# ──────────────────────────────────────────────────────────
from __future__ import annotations

import io
import logging
import warnings
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from imblearn.over_sampling import SMOTE
from sklearn.exceptions import ConvergenceWarning
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split

# ──────────────────────────────────────────────────────────
# Silence noisy warnings
# ──────────────────────────────────────────────────────────
warnings.filterwarnings("ignore", category=(FutureWarning, RuntimeWarning, UserWarning))
warnings.filterwarnings("ignore", category=ConvergenceWarning)
pd.options.mode.chained_assignment = None

# ──────────────────────────────────────────────────────────
# Project‑specific imports
# ──────────────────────────────────────────────────────────
from evaluation import explain_model_shap
from feature_engineering import feature_engineering
from model_training import (
    hyperparameter_tuning_rf,
    train_and_evaluate_models,
)
from preprocessing import preprocess_data

# ──────────────────────────────────────────────────────────
# Paths / constants
# ──────────────────────────────────────────────────────────
MODELS_DIR = Path("models")
BEST_MODEL_PATH = MODELS_DIR / "best_rf_model.pkl"
PLOTLY_TMPL = "plotly_dark" if st.get_option("theme.base") == "dark" else "plotly"


# ──────────────────────────────────────────────────────────
# Helper utilities
# ──────────────────────────────────────────────────────────
def numeric_impute(df: pd.DataFrame) -> pd.DataFrame:
    """Convert to numeric, drop all‑NaN columns, median‑impute."""
    df_num = df.apply(pd.to_numeric, errors="coerce").dropna(axis=1, how="all")
    imp = SimpleImputer(strategy="median")
    return pd.DataFrame(imp.fit_transform(df_num), columns=df_num.columns)


def align_to_model(X: pd.DataFrame, model) -> pd.DataFrame:
    """Ensure column order matches model.feature_names_in_ (adds 0‑filled missing)."""
    if hasattr(model, "feature_names_in_"):
        cols = list(model.feature_names_in_)
        return X.reindex(cols, axis=1, fill_value=0)
    return X


@st.cache_resource(show_spinner=False)
def cached_preprocess(df_raw: pd.DataFrame) -> pd.DataFrame:
    df = preprocess_data(df_raw.copy(), target_column="fraud_reported")
    return feature_engineering(df)


# ──────────────────────────────────────────────────────────
# Beautiful CSS injection
# ──────────────────────────────────────────────────────────
st.markdown(
    """
<style>
/* center metrics */
[data-testid="metric-container"] {text-align:center !important;}
/* smoother plots */
.plot-container > div {border-radius:12px !important;}
/* subtle shadow for frames */
section.main > div {box-shadow:0 0 8px rgba(0,0,0,0.07);}
</style>
""",
    unsafe_allow_html=True,
)

# ──────────────────────────────────────────────────────────
# Page config
# ──────────────────────────────────────────────────────────
st.set_page_config("Fraud Dashboard", "🚦", layout="wide")
st.title("🚦 Insurance Claim Fraud Detection Dashboard")

# ──────────────────────────────────────────────────────────
# Session state
# ──────────────────────────────────────────────────────────
ss = st.session_state
ss.setdefault("raw", None)
ss.setdefault("prep", None)
ss.setdefault("model", None)  # tuned model
ss.setdefault("pred_df", None)

# ──────────────────────────────────────────────────────────
# Tabs
# ──────────────────────────────────────────────────────────
tabs = st.tabs(
    ["📁 Upload", "🔬 EDA", "⚙️ Preprocess", "🧠 Train", "📊 Predict", "🔎 Explain"]
)
tab_upload, tab_eda, tab_prep, tab_train, tab_pred, tab_explain = tabs

# ──────────────────────────────────────────────────────────
# 1 · Upload
# ──────────────────────────────────────────────────────────
with tab_upload:
    st.header("📁 Upload dataset")
    file = st.file_uploader("CSV only", type="csv")
    if file:
        ss.raw = pd.read_csv(file)
        st.toast("File loaded!", icon="✅")
        st.dataframe(ss.raw.head(), use_container_width=True)

# ──────────────────────────────────────────────────────────
# 2 · EDA (quick insights)
# ──────────────────────────────────────────────────────────
with tab_eda:
    st.header("🔬 Exploratory Data Analysis")
    if ss.raw is None:
        st.info("Upload a dataset first.")
    else:
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Class balance")
            fig = px.histogram(
                ss.raw,
                x="fraud_reported",
                template=PLOTLY_TMPL,
                color="fraud_reported",
            )
            st.plotly_chart(fig, use_container_width=True)
        with col2:
            st.subheader("Correlation heat‑map")
            corr = ss.raw.select_dtypes("number").corr()
            fig = px.imshow(
                corr,
                template=PLOTLY_TMPL,
                aspect="auto",
                color_continuous_scale="Blues",
            )
            st.plotly_chart(fig, use_container_width=True)

# ──────────────────────────────────────────────────────────
# 3 · Preprocess
# ──────────────────────────────────────────────────────────
with tab_prep:
    st.header("⚙️ Preprocess + Feature Engineering")
    if ss.raw is None:
        st.info("Upload a dataset first.")
    elif st.button("Run preprocessing", key="prep_btn"):
        with st.spinner("Preprocessing…"):
            ss.prep = cached_preprocess(ss.raw)
        st.success("Done ✔️")
    if ss.prep is not None:
        st.write(f"Shape: {ss.prep.shape}")
        st.dataframe(ss.prep.head(), use_container_width=True)

# ──────────────────────────────────────────────────────────
# 4 · Train
# ──────────────────────────────────────────────────────────
with tab_train:
    st.header("🧠 Model Training & Tuning")
    if ss.prep is None:
        st.info("Run preprocessing first.")
    else:
        tune = st.toggle("Hyperparameter tuning (GridSearch)", value=True)
        if st.button("Start training", key="train_btn"):
            with st.spinner("Training…"):
                df = ss.prep.copy().dropna(subset=["fraud_reported"])
                X, y = df.drop("fraud_reported", axis=1), df["fraud_reported"]
                X_imp = numeric_impute(X)
                X_bal, y_bal = SMOTE(random_state=42).fit_resample(X_imp, y)
                X_tr, X_te, y_tr, y_te = train_test_split(
                    X_bal, y_bal, test_size=0.2, random_state=42
                )

                # Capture training logs
                buf = io.StringIO()
                handler = logging.StreamHandler(buf)
                logging.getLogger().addHandler(handler)
                plt_show_orig = plt.show
                plt.show = lambda *a, **k: st.pyplot(plt.gcf(), clear_figure=True)

                train_and_evaluate_models(X_tr, X_te, y_tr, y_te)

                best_model = None
                if tune:
                    best_model = hyperparameter_tuning_rf(X_tr, X_te, y_tr, y_te)
                plt.show = plt_show_orig
                logging.getLogger().removeHandler(handler)
                st.expander("Logs").text(buf.getvalue())

                # Save tuned model if any
                if best_model:
                    MODELS_DIR.mkdir(exist_ok=True)
                    joblib.dump(best_model, BEST_MODEL_PATH)
                    ss.model = best_model
                    st.toast("Tuned model saved.", icon="💾")
                elif BEST_MODEL_PATH.exists():
                    ss.model = joblib.load(BEST_MODEL_PATH)

                # Quick metrics for chosen model
                if ss.model:
                    X_te_aligned = align_to_model(X_te, ss.model)
                    preds = ss.model.predict(X_te_aligned)
                    col1, col2, col3 = st.columns(3)
                    col1.metric("Accuracy", f"{accuracy_score(y_te, preds):.3f}")
                    col2.metric("F1", f"{f1_score(y_te, preds):.3f}")
                    col3.metric("ROC AUC", f"{roc_auc_score(y_te, preds):.3f}")

                    # ROC curve
                    fpr, tpr, _ = roc_curve(y_te, preds)
                    fig = go.Figure(
                        go.Scatter(x=fpr, y=tpr, mode="lines", name="ROC"),
                        layout=dict(
                            template=PLOTLY_TMPL,
                            xaxis_title="FPR",
                            yaxis_title="TPR",
                            title="ROC Curve",
                        ),
                    )
                    st.plotly_chart(fig, use_container_width=True)

                    # Confusion matrix
                    cm = confusion_matrix(y_te, preds)
                    fig_cm = px.imshow(
                        cm,
                        text_auto=True,
                        color_continuous_scale="Blues",
                        template=PLOTLY_TMPL,
                    )
                    fig_cm.update_layout(title="Confusion Matrix")
                    st.plotly_chart(fig_cm, use_container_width=True)

                    # Allow download
                    with open(BEST_MODEL_PATH, "rb") as f:
                        st.download_button(
                            "Download tuned model", f, "best_rf_model.pkl"
                        )

# ──────────────────────────────────────────────────────────
# 5 · Predict
# ──────────────────────────────────────────────────────────
with tab_pred:
    st.header("📊 Batch Prediction")
    if ss.prep is None:
        st.info("Need preprocessed data.")
    elif not BEST_MODEL_PATH.exists():
        st.error("No model found – train first.")
    else:
        if st.button("Predict", key="pred_btn"):
            with st.spinner("Predicting…"):
                model = joblib.load(BEST_MODEL_PATH)
                X_pred = ss.prep.drop(columns=["fraud_reported"], errors="ignore")
                X_pred = align_to_model(numeric_impute(X_pred), model)
                ss.pred_df = ss.prep.copy()
                ss.pred_df["Prediction"] = model.predict(X_pred)
            st.toast("Prediction complete!", icon="✅")

        if ss.pred_df is not None:
            st.dataframe(ss.pred_df.head(), use_container_width=True)
            st.download_button(
                "Download predictions",
                ss.pred_df.to_csv(index=False).encode(),
                "predictions.csv",
                "text/csv",
            )

# ──────────────────────────────────────────────────────────
# 6 · Explain
# ──────────────────────────────────────────────────────────
with tab_explain:
    st.header("🔎 SHAP Explainability")
    if ss.prep is None:
        st.info("Need preprocessed data.")
    elif not BEST_MODEL_PATH.exists():
        st.error("No model found – train first.")
    else:
        if st.button("Generate SHAP", key="shap_btn"):
            with st.spinner("Calculating SHAP…"):
                model = joblib.load(BEST_MODEL_PATH)
                X_exp = ss.prep.drop(columns=["fraud_reported"], errors="ignore")
                X_exp = align_to_model(numeric_impute(X_exp), model)
                plt_show_orig = plt.show
                plt.show = lambda *a, **k: st.pyplot(plt.gcf(), clear_figure=True)
                explain_model_shap(model, X_exp)
                plt.show = plt_show_orig
