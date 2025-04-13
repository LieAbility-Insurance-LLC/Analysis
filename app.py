# ──────────────────────────────────────────────────────────
# Insurance‑Fraud Streamlit Dashboard · app.py  (v2.4)
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
    confusion_matrix,
    f1_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split

# ──────────────────────────────────────────────────────────
# FIRST Streamlit command
# ──────────────────────────────────────────────────────────
st.set_page_config(page_title="Fraud Dashboard", page_icon="🚦", layout="wide")

# ──────────────────────────────────────────────────────────
# Silence noisy warnings
# ──────────────────────────────────────────────────────────
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=ConvergenceWarning)
pd.options.mode.chained_assignment = None

# ──────────────────────────────────────────────────────────
# Project imports
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
    df_num = df.apply(pd.to_numeric, errors="coerce").dropna(axis=1, how="all")
    imp = SimpleImputer(strategy="median")
    return pd.DataFrame(imp.fit_transform(df_num), columns=df_num.columns)


def align_to_model(X: pd.DataFrame, model) -> pd.DataFrame:
    if hasattr(model, "feature_names_in_"):
        cols = list(model.feature_names_in_)
        return X.reindex(cols, axis=1, fill_value=0)
    return X


def get_top_features_rf(model, top_n: int = 10) -> pd.Series:
    """Return top_n features by importance from a RandomForest‑like model."""
    importances = model.feature_importances_
    return (
        pd.Series(importances, index=model.feature_names_in_)
        .sort_values(ascending=False)
        .head(top_n)
    )


@st.cache_resource(show_spinner=False)
def cached_preprocess(df_raw: pd.DataFrame) -> pd.DataFrame:
    df = preprocess_data(df_raw.copy(), target_column="fraud_reported")
    return feature_engineering(df)

# ──────────────────────────────────────────────────────────
# CSS tweaks
# ──────────────────────────────────────────────────────────
st.markdown(
    """
<style>
[data-testid="metric-container"] {text-align:center !important;}
.plot-container > div {border-radius:12px !important;}
section.main > div {box-shadow:0 0 8px rgba(0,0,0,0.07);}
</style>
""",
    unsafe_allow_html=True,
)

# ──────────────────────────────────────────────────────────
# Title
# ──────────────────────────────────────────────────────────
st.title("🚦 Insurance Claim Fraud Detection Dashboard")

# ──────────────────────────────────────────────────────────
# Session state shortcuts
# ──────────────────────────────────────────────────────────
ss = st.session_state
ss.setdefault("raw", None)
ss.setdefault("prep", None)
ss.setdefault("model", None)
ss.setdefault("pred_df", None)
ss.setdefault("top_features", None)  # NEW

# ──────────────────────────────────────────────────────────
# Tabs
# ──────────────────────────────────────────────────────────
tabs = st.tabs(
    [
        "📁 Upload",
        "🔬 EDA",
        "⚙️ Preprocess",
        "🧠 Train",
        "📊 Predict",
        "📝 Single Prediction",
        "🔎 Explain",
    ]
)
(
    tab_upload,
    tab_eda,
    tab_prep,
    tab_train,
    tab_pred,
    tab_single,
    tab_explain,
) = tabs

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
# 2 · EDA
# ──────────────────────────────────────────────────────────
with tab_eda:
    st.header("🔬 Exploratory Data Analysis")
    if ss.raw is None:
        st.info("Upload a dataset first.")
    else:
        c1, c2 = st.columns(2)
        with c1:
            st.subheader("Class balance")
            fig = px.histogram(
                ss.raw, x="fraud_reported", color="fraud_reported", template=PLOTLY_TMPL
            )
            st.plotly_chart(fig, use_container_width=True)
        with c2:
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
# 4 · Train  (now saves top_features)
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

                buf = io.StringIO()
                handler = logging.StreamHandler(buf)
                logging.getLogger().addHandler(handler)
                plt_orig = plt.show
                plt.show = lambda *a, **k: st.pyplot(plt.gcf(), clear_figure=True)

                train_and_evaluate_models(X_tr, X_te, y_tr, y_te)
                best_model = (
                    hyperparameter_tuning_rf(X_tr, X_te, y_tr, y_te) if tune else None
                )

                plt.show = plt_orig
                logging.getLogger().removeHandler(handler)
                st.expander("Logs").text(buf.getvalue())

                if best_model:
                    MODELS_DIR.mkdir(exist_ok=True)
                    joblib.dump(best_model, BEST_MODEL_PATH)
                    ss.model = best_model
                    st.toast("Tuned model saved.", icon="💾")
                elif BEST_MODEL_PATH.exists():
                    ss.model = joblib.load(BEST_MODEL_PATH)

                # ---- NEW: compute & store top predictors --------------------
                if ss.model:
                    ss.top_features = get_top_features_rf(ss.model, top_n=10).index.tolist()
                    st.write("🔑 **Top predictors** (saved for single‑prediction):")
                    st.write(ss.top_features)

                    # Quick metrics
                    X_te_aligned = align_to_model(X_te, ss.model)
                    preds = ss.model.predict(X_te_aligned)
                    c1, c2, c3 = st.columns(3)
                    c1.metric("Accuracy", f"{accuracy_score(y_te, preds):.3f}")
                    c2.metric("F1", f"{f1_score(y_te, preds):.3f}")
                    c3.metric("ROC AUC", f"{roc_auc_score(y_te, preds):.3f}")

                    fpr, tpr, _ = roc_curve(y_te, preds)
                    roc_fig = go.Figure(
                        go.Scatter(x=fpr, y=tpr, mode="lines", name="ROC"),
                        layout=dict(
                            template=PLOTLY_TMPL,
                            xaxis_title="FPR",
                            yaxis_title="TPR",
                            title="ROC Curve",
                        ),
                    )
                    st.plotly_chart(roc_fig, use_container_width=True)

                    cm = confusion_matrix(y_te, preds)
                    cm_fig = px.imshow(
                        cm,
                        text_auto=True,
                        color_continuous_scale="Blues",
                        template=PLOTLY_TMPL,
                    )
                    cm_fig.update_layout(title="Confusion Matrix")
                    st.plotly_chart(cm_fig, use_container_width=True)

                    with open(BEST_MODEL_PATH, "rb") as f:
                        st.download_button("Download tuned model", f, "best_rf_model.pkl")

# ──────────────────────────────────────────────────────────
# 5 · Batch Predict
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
# 6 · 📝 Single Prediction (now uses top_features)
# ──────────────────────────────────────────────────────────
with tab_single:
    st.header("📝 Single Prediction (Probability)")
    if not BEST_MODEL_PATH.exists():
        st.error("No trained model found. Train or load a model first.")
    elif ss.raw is None:
        st.info("Upload a dataset first (needed for categorical choices).")
    else:
        raw_df = ss.raw
        model = joblib.load(BEST_MODEL_PATH)
        top_cols = ss.top_features or list(model.feature_names_in_)

        with st.form("single_pred_form"):
            st.write("Enter feature values (top predictors):")
            inputs: dict[str, object] = {}

            for col in top_cols:
                if col not in raw_df.columns:
                    inputs[col] = st.number_input(col, value=0.0, key=f"single_{col}")
                    continue

                if raw_df[col].dtype == object:
                    opts = sorted(raw_df[col].dropna().unique().tolist())
                    if 2 < len(opts) <= 25:
                        inputs[col] = st.selectbox(col, options=opts, key=f"single_{col}")
                    else:
                        inputs[col] = st.text_input(col, key=f"single_{col}")
                else:
                    default_val = float(raw_df[col].median())
                    inputs[col] = st.number_input(col, value=default_val, key=f"single_{col}")

            submitted = st.form_submit_button("Predict")

        if submitted:
            new_raw = pd.DataFrame([inputs])
            tmp_raw = pd.concat([raw_df.head(1).copy(), new_raw], ignore_index=True)
            tmp_proc = cached_preprocess(tmp_raw)
            single_proc = tmp_proc.tail(1).drop(columns=["fraud_reported"], errors="ignore")
            single_proc = align_to_model(numeric_impute(single_proc), model)
            proba = (
                model.predict_proba(single_proc)[0][1]
                if hasattr(model, "predict_proba")
                else model.predict(single_proc)[0]
            )
            percent = float(proba) * 100

            st.subheader("Fraud Probability")
            gauge = go.Figure(
                go.Indicator(
                    mode="gauge+number",
                    value=percent,
                    number={"suffix": "%"},
                    gauge={
                        "axis": {"range": [0, 100]},
                        "bar": {"color": "#ff4b4b"},
                        "steps": [
                            {"range": [0, 50], "color": "#2ecc71"},
                            {"range": [50, 100], "color": "#ff4b4b"},
                        ],
                    },
                )
            )
            gauge.update_layout(template=PLOTLY_TMPL, height=300)
            st.plotly_chart(gauge, use_container_width=False)
            st.success(
                f"Estimated fraud probability: **{percent:.2f}%** "
                f"({'Fraud' if percent>=50 else 'Not Fraud'})"
            )

# ──────────────────────────────────────────────────────────
# 7 · Explain
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
                plt_orig = plt.show
                plt.show = lambda *a, **k: st.pyplot(plt.gcf(), clear_figure=True)
                explain_model_shap(model, X_exp)
                plt.show = plt_orig
