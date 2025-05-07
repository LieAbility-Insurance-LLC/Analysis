# ──────────────────────────────────────────────────────────
# Insurance‑Fraud Streamlit Dashboard · app.py  (v2.5 · 2025‑05‑07)
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
from evaluation import (
    explain_model_shap,
    plot_tsne_embedding,          # NEW
)
from feature_engineering import feature_engineering
from model_training import (
    hyperparameter_tuning_rf,
    train_and_evaluate_models,
)
from preprocessing import (
    preprocess_data,
    show_top_feature_violin_plots,  # NEW
)

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
    """
    Ensure numeric input and median‑impute missing values.
    Keeps original index to preserve alignment downstream.
    """
    df_num = df.apply(pd.to_numeric, errors="coerce")
    imp = SimpleImputer(strategy="median")
    imputed = pd.DataFrame(imp.fit_transform(df_num), columns=df_num.columns, index=df_num.index)
    return imputed


def align_to_model(X: pd.DataFrame, model) -> pd.DataFrame:
    """
    Re‑orders / pads columns so they match the training‑time feature set.
    """
    if hasattr(model, "feature_names_in_"):
        cols = list(model.feature_names_in_)
        return X.reindex(cols, axis=1, fill_value=0)
    return X


def get_top_features_rf(model, top_n: int = 10) -> pd.Series:
    """
    Return top_n features by importance from a RandomForest‑like model.
    """
    if not hasattr(model, "feature_importances_"):
        return pd.Series(dtype=float)
    importances = model.feature_importances_
    return (
        pd.Series(importances, index=model.feature_names_in_)
        .sort_values(ascending=False)
        .head(top_n)
    )


@st.cache_resource(show_spinner=False)
def cached_preprocess(df_raw: pd.DataFrame) -> pd.DataFrame:
    """
    Cached preprocessing + feature engineering pipeline.
    """
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
st.title("🚦 Insurance Claim Fraud Detection Dashboard")

# ──────────────────────────────────────────────────────────
# Session state shortcuts
# ──────────────────────────────────────────────────────────
ss = st.session_state
ss.setdefault("raw", None)
ss.setdefault("prep", None)
ss.setdefault("model", None)
ss.setdefault("pred_df", None)
ss.setdefault("top_features", None)  # ← saved after training

# ──────────────────────────────────────────────────────────
# Tabs
# ──────────────────────────────────────────────────────────
tabs = st.tabs(
    [
        "📁 Upload",
        "🔬 EDA",
        "⚙️ Preprocess",
        "🧠 Train",
        "📊 Predict",
        "📝 Single Prediction",
        "🔎 Explain",
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
    st.header("📁 Upload dataset")
    file = st.file_uploader("CSV only", type="csv")

    # Load only the first time the file appears
    if file and "raw_loaded" not in ss:
        ss.raw = pd.read_csv(file)
        ss.raw_loaded = True
        st.toast("Dataset successfully uploaded!", icon="📥")

    if ss.get("raw") is not None:
        st.dataframe(ss.raw.head(), use_container_width=True)

# ──────────────────────────────────────────────────────────
# 2 · EDA  (NEW Advanced Visuals)
# ──────────────────────────────────────────────────────────
with tab_eda:
    st.header("🔬 Exploratory Data Analysis")

    if ss.raw is None:
        st.info("Upload a dataset first.")
    else:
        c1, c2 = st.columns(2)

        with c1:
            st.subheader("Class balance")
            fig = px.histogram(
                ss.raw,
                x="fraud_reported",
                color="fraud_reported",
                template=PLOTLY_TMPL,
            )
            st.plotly_chart(fig, use_container_width=True)

        with c2:
            st.subheader("Correlation heat‑map")
            num_cols = ss.raw.select_dtypes("number")
            if num_cols.shape[1] >= 2:
                corr = num_cols.corr()
                fig = px.imshow(
                    corr,
                    template=PLOTLY_TMPL,
                    aspect="auto",
                    color_continuous_scale="Blues",
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("Not enough numeric columns for correlation heat‑map.")

        # ─────────── Advanced visuals (NEW) ────────────
        st.divider()
        st.subheader("⌛ Advanced Visuals")
        if st.button("Generate t‑SNE + Violin plots", key="adv_vis_btn"):
            with st.spinner("Generating advanced visuals…"):
                try:
                    processed = cached_preprocess(ss.raw)
                    X = processed.drop(columns=["fraud_reported"], errors="ignore")
                    y = processed["fraud_reported"]
                    # t‑SNE
                    plot_tsne_embedding(X, y, model_name="Preprocessed data")
                    # Top‑feature violins (variance‑based)
                    show_top_feature_violin_plots(processed, top_n=10)
                except Exception as exc:
                    st.error(f"Failed to generate visuals: {exc}")

# ──────────────────────────────────────────────────────────
# 3 · Preprocess
# ──────────────────────────────────────────────────────────
with tab_prep:
    st.header("⚙️ Preprocess + Feature Engineering")

    if ss.raw is None:
        st.info("Upload a dataset first.")
    elif st.button("Run preprocessing", key="prep_btn"):
        with st.spinner("Preprocessing…"):
            ss.prep = cached_preprocess(ss.raw)
        st.success("Preprocessing complete ✔️")

    if ss.prep is not None:
        st.write(f"Shape: {ss.prep.shape}")
        st.dataframe(ss.prep.head(), use_container_width=True)

# ──────────────────────────────────────────────────────────
# 4 · Train  (stores top_features)
# ──────────────────────────────────────────────────────────
with tab_train:
    st.header("🧠 Model Training & Tuning")

    if ss.prep is None:
        st.info("Run preprocessing first.")
    else:
        tune = st.toggle("Hyperparameter tuning (GridSearch)", value=True)

        if st.button("Start training", key="train_btn"):
            with st.spinner("Training…"):
                df = ss.prep.dropna(subset=["fraud_reported"]).copy()
                X, y = df.drop("fraud_reported", axis=1), df["fraud_reported"]

                # ‑‑ impute & balance
                X_imp = numeric_impute(X)
                X_bal, y_bal = SMOTE(random_state=42).fit_resample(X_imp, y)

                # ‑‑ split
                X_tr, X_te, y_tr, y_te = train_test_split(
                    X_bal, y_bal, test_size=0.2, random_state=42, stratify=y_bal
                )

                # capture logs & inline matplotlib
                buf = io.StringIO()
                handler = logging.StreamHandler(buf)
                root_logger = logging.getLogger()
                root_logger.addHandler(handler)
                plt_orig_show = plt.show
                plt.show = lambda *a, **k: st.pyplot(plt.gcf(), clear_figure=True)

                # ‑‑ train baseline models
                train_and_evaluate_models(X_tr, X_te, y_tr, y_te)

                # ‑‑ hyperparameter tuning (optional)
                best_model = (
                    hyperparameter_tuning_rf(X_tr, X_te, y_tr, y_te) if tune else None
                )

                # reset matplotlib show & logger
                plt.show = plt_orig_show
                root_logger.removeHandler(handler)
                st.expander("Logs").text(buf.getvalue())

                # persist or load best model
                if best_model:
                    MODELS_DIR.mkdir(exist_ok=True)
                    joblib.dump(best_model, BEST_MODEL_PATH)
                    ss.model = best_model
                    st.toast("Tuned model saved.", icon="💾")
                elif BEST_MODEL_PATH.exists():
                    ss.model = joblib.load(BEST_MODEL_PATH)

                # ---- Save top predictors to session -----------------------
                if ss.model:
                    ss.top_features = get_top_features_rf(ss.model, top_n=10).index.tolist()
                    st.write("🔑 **Top predictors** stored for single‑prediction:")
                    st.write(ss.top_features)

                    # Quick metrics
                    X_te_aligned = align_to_model(X_te, ss.model)
                    preds = ss.model.predict(X_te_aligned)
                    c1, c2, c3 = st.columns(3)
                    c1.metric("Accuracy", f"{accuracy_score(y_te, preds):.3f}")
                    c2.metric("F1‑Score", f"{f1_score(y_te, preds):.3f}")
                    c3.metric("ROC AUC", f"{roc_auc_score(y_te, preds):.3f}")

                    # ROC curve
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

                    # Confusion matrix
                    cm = confusion_matrix(y_te, preds)
                    cm_fig = px.imshow(
                        cm,
                        text_auto=True,
                        color_continuous_scale="Blues",
                        template=PLOTLY_TMPL,
                    )
                    cm_fig.update_layout(title="Confusion Matrix")
                    st.plotly_chart(cm_fig, use_container_width=True)

                    # Download model
                    with open(BEST_MODEL_PATH, "rb") as f:
                        st.download_button("Download tuned model", f, "best_rf_model.pkl")

# ──────────────────────────────────────────────────────────
# 5 · Batch Predict
# ──────────────────────────────────────────────────────────
with tab_pred:
    st.header("📊 Batch Prediction")

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
# 6 · 📝 Single Prediction (unchanged except bug‑fixes)
# ──────────────────────────────────────────────────────────
with tab_single:
    st.header("📝 Single Prediction (Probability)")

    if not BEST_MODEL_PATH.exists():
        st.error("No trained model found. Train or load a model first.")
    elif ss.raw is None:
        st.info("Upload a dataset first (needed for categorical choices).")
    else:
        raw_df = ss.raw
        model = joblib.load(BEST_MODEL_PATH)
        top_cols = ss.top_features or list(model.feature_names_in_)

        with st.form("single_pred_form", clear_on_submit=False):
            st.write("Enter feature values:")
            inputs: dict[str, object] = {}

            for col in top_cols:
                # --- Handle date fields -----------------------------------
                if "date" in col.lower():
                    default_date = pd.to_datetime(
                        raw_df[col].dropna().iloc[0]
                        if col in raw_df.columns
                        else "2020-01-01"
                    )
                    chosen = st.date_input(col, value=default_date, key=f"single_{col}")
                    inputs[col] = chosen.strftime("%m/%d/%Y")
                    continue

                # --- Column absent from raw (allow free numeric input) ----
                if col not in raw_df.columns:
                    inputs[col] = st.number_input(col, value=0.0, key=f"single_{col}")
                    continue

                series = raw_df[col]
                if series.dtype == object:
                    opts = sorted(series.dropna().unique().tolist())
                    if 2 < len(opts) <= 25:
                        inputs[col] = st.selectbox(col, options=opts, key=f"single_{col}")
                    else:
                        inputs[col] = st.text_input(col, key=f"single_{col}")
                else:
                    # decide if integer‑like
                    is_int_like = pd.api.types.is_integer_dtype(series) or (
                        series.dropna() % 1 == 0
                    ).all()
                    default_val = float(series.median())
                    if is_int_like:
                        inputs[col] = st.number_input(
                            col,
                            value=int(default_val),
                            step=1,
                            format="%d",
                            key=f"single_{col}",
                        )
                    else:
                        inputs[col] = st.number_input(
                            col, value=default_val, key=f"single_{col}"
                        )

            submitted = st.form_submit_button("Predict")

        # ---------- prediction & display -------------------------------
        if submitted:
            try:
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
                    f"({'Fraud' if percent >= 50 else 'Not Fraud'})"
                )
                st.toast("Single prediction complete!", icon="🎉")
            except Exception as exc:
                st.error(f"Prediction failed: {exc}")

# ──────────────────────────────────────────────────────────
# 7 · Explain
# ──────────────────────────────────────────────────────────
with tab_explain:
    st.header("🔎SHAP Explainability")

    if ss.prep is None:
        st.info("Need preprocessed data.")
    elif not BEST_MODEL_PATH.exists():
        st.error("No model found – train first.")
    else:
        if st.button("Generate SHAP", key="shap_btn"):
            with st.spinner("Calculating SHAP…"):
                try:
                    model = joblib.load(BEST_MODEL_PATH)
                    X_exp = ss.prep.drop(columns=["fraud_reported"], errors="ignore")
                    X_exp = align_to_model(numeric_impute(X_exp), model)
                    plt_orig_show = plt.show
                    plt.show = lambda *a, **k: st.pyplot(plt.gcf(), clear_figure=True)
                    explain_model_shap(model, X_exp)
                    plt.show = plt_orig_show
                except Exception as exc:
                    st.error(f"SHAP computation failed: {exc}")
