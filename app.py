from __future__ import annotations

import io, logging, warnings
from pathlib import Path

import joblib, matplotlib.pyplot as plt, numpy as np, pandas as pd
import plotly.express as px, plotly.graph_objects as go, streamlit as st
from imblearn.over_sampling import SMOTE
from sklearn.exceptions import ConvergenceWarning
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score, confusion_matrix, f1_score,
    roc_auc_score, roc_curve,
)
from sklearn.model_selection import train_test_split

# ──────────────────────────────────────────────────────────
# FIRST Streamlit command
# ──────────────────────────────────────────────────────────
st.set_page_config(page_title="Fraud Dashboard", page_icon="🚦", layout="wide")

# ──────────────────────────────────────────────────────────
# Silence noisy warnings
# ──────────────────────────────────────────────────────────
for w in (FutureWarning, RuntimeWarning, UserWarning, ConvergenceWarning):
    warnings.filterwarnings("ignore", category=w)
pd.options.mode.chained_assignment = None

# ──────────────────────────────────────────────────────────
# Project imports
# ──────────────────────────────────────────────────────────
from evaluation import explain_model_shap
from feature_engineering import feature_engineering
from model_training import hyperparameter_tuning_rf, train_and_evaluate_models
from preprocessing import preprocess_data

# ──────────────────────────────────────────────────────────
# Paths / constants
# ──────────────────────────────────────────────────────────
MODELS_DIR = Path("models"); MODELS_DIR.mkdir(exist_ok=True)
DEFAULT_CSV  = Path("insurance_claims.csv")
PLOTLY_TMPL = "plotly_dark" if st.get_option("theme.base") == "dark" else "plotly"

# ──────────────────────────────────────────────────────────
# Helper utilities
# ──────────────────────────────────────────────────────────
def numeric_impute(df: pd.DataFrame) -> pd.DataFrame:
    """
    Cast everything to numeric (invalid → NaN), DROP columns that are now all‑NaN,
    then median‑impute.  Safeguard prevents pandas/ndarray column‑count mismatch.
    """
    df_num = df.apply(pd.to_numeric, errors="coerce")
    df_num = df_num.dropna(axis=1, how="all")          # <- key fix
    imputed = SimpleImputer(strategy="median").fit_transform(df_num)
    return pd.DataFrame(imputed, columns=df_num.columns, index=df_num.index)

def align_to_model(X: pd.DataFrame, model) -> pd.DataFrame:
    if hasattr(model, "feature_names_in_"):
        cols = list(model.feature_names_in_)
        return X.reindex(cols, axis=1, fill_value=0)
    return X

def get_top_features_rf(model, top_n: int = 10) -> pd.Series:
    if not hasattr(model, "feature_importances_"):
        return pd.Series(dtype=float)
    return (
        pd.Series(model.feature_importances_, index=model.feature_names_in_)
        .sort_values(ascending=False)
        .head(top_n)
    )

@st.cache_resource(show_spinner=False)
def cached_preprocess(df_raw: pd.DataFrame) -> pd.DataFrame:
    return feature_engineering(preprocess_data(df_raw.copy(), target_column="fraud_reported"))

def list_available_models() -> list[str]:
    return sorted([p.name for p in MODELS_DIR.glob("*.pkl")])

def load_selected_model(name: str):
    path = MODELS_DIR / name
    return joblib.load(path) if path.exists() else None

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
st.title("🚦Insurance Claim Fraud Detection Dashboard")

# ──────────────────────────────────────────────────────────
# Session state shortcuts
# ──────────────────────────────────────────────────────────
ss = st.session_state
ss.setdefault("raw", None)
ss.setdefault("prep", None)
ss.setdefault("model_name", None)
ss.setdefault("model", None)
ss.setdefault("pred_df", None)
ss.setdefault("top_features", None)

# ──────────────────────────────────────────────────────────
# Tabs
# ──────────────────────────────────────────────────────────
tabs = st.tabs(
    [
        "📁 Upload / Load",
        "🔬 EDA",
        "⚙️ Preprocess",
        "🧠 Train",
        "📊 Predict",
        "📝 Single Prediction",
        "🔎 Explain",
    ]
)
(tab_upload, tab_eda, tab_prep, tab_train,
 tab_pred, tab_single, tab_explain) = tabs

# ──────────────────────────────────────────────────────────
# 1 · Upload or auto‑load default CSV
# ──────────────────────────────────────────────────────────
with tab_upload:
    st.header("📁Dataset Loader")
    if ss.raw is None and DEFAULT_CSV.exists():
        ss.raw = pd.read_csv(DEFAULT_CSV)
        st.toast(f'Loaded "{DEFAULT_CSV}" automatically.', icon="⚡")
    file = st.file_uploader("Upload a CSV (optional)", type="csv")
    if file:
        ss.raw = pd.read_csv(file)
        st.toast("Dataset uploaded!", icon="📥")
    if ss.raw is not None:
        st.dataframe(ss.raw.head(), use_container_width=True)
        st.success(f"Active dataset shape: {ss.raw.shape}")

# ──────────────────────────────────────────────────────────
# 2 · EDA
# ──────────────────────────────────────────────────────────
with tab_eda:
    st.header("🔬Exploratory Data Analysis")
    if ss.raw is None:
        st.info("Load a dataset first.")
    else:
        c1, c2 = st.columns(2)
        with c1:
            fig = px.histogram(ss.raw, x="fraud_reported", color="fraud_reported",
                               template=PLOTLY_TMPL)
            st.subheader("Class balance")
            st.plotly_chart(fig, use_container_width=True)
        with c2:
            st.subheader("Correlation heat-map")
            num_cols = ss.raw.select_dtypes("number")
            if num_cols.shape[1] >= 2:
                corr = num_cols.corr()
                fig = px.imshow(corr, template=PLOTLY_TMPL, aspect="auto",
                                color_continuous_scale="Blues")
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("Not enough numeric columns.")

# ──────────────────────────────────────────────────────────
# 3 · Preprocess
# ──────────────────────────────────────────────────────────
with tab_prep:
    st.header("⚙️Preprocess+Feature Engineering")
    if ss.raw is None:
        st.info("Load a dataset first.")
    elif st.button("Run preprocessing"):
        with st.spinner("Preprocessing…"):
            ss.prep = cached_preprocess(ss.raw)
        st.success("Preprocessing complete ✔️")
    if ss.prep is not None:
        st.write(f"Shape: {ss.prep.shape}")
        st.dataframe(ss.prep.head(), use_container_width=True)

# ──────────────────────────────────────────────────────────
# 4 · Train
# ──────────────────────────────────────────────────────────
with tab_train:
    st.header("🧠Model Training&Tuning")
    if ss.prep is None:
        st.info("Run preprocessing first.")
    else:
        tune = st.toggle("Hyperparameter tuning (GridSearch)", value=True)
        save_as = st.text_input("Save model as (.pkl name)", value="best_model")
        if st.button("Start training"):
            with st.spinner("Training…"):
                df = ss.prep.dropna(subset=["fraud_reported"]).copy()
                X, y = df.drop("fraud_reported", axis=1), df["fraud_reported"]
                X_imp = numeric_impute(X)
                X_bal, y_bal = SMOTE(random_state=42).fit_resample(X_imp, y)
                X_tr, X_te, y_tr, y_te = train_test_split(
                    X_bal, y_bal, test_size=0.2, random_state=42, stratify=y_bal
                )

                buf = io.StringIO()
                handler = logging.StreamHandler(buf)
                root_logger = logging.getLogger()
                root_logger.addHandler(handler)
                plt_orig_show = plt.show
                plt.show = lambda *a, **k: st.pyplot(plt.gcf(), clear_figure=True)

                train_and_evaluate_models(X_tr, X_te, y_tr, y_te)
                best_model = hyperparameter_tuning_rf(X_tr, X_te, y_tr, y_te) if tune else None

                plt.show = plt_orig_show
                root_logger.removeHandler(handler)
                st.expander("Logs").text(buf.getvalue())

                if best_model:
                    fname = f"{save_as}.pkl"
                    joblib.dump(best_model, MODELS_DIR / fname)
                    ss.model_name, ss.model = fname, best_model
                    st.toast(f"Model saved as '{fname}'.", icon="💾")
                    ss.top_features = get_top_features_rf(best_model, 10).index.tolist()

# ──────────────────────────────────────────────────────────
# 5 · Batch Predict
# ──────────────────────────────────────────────────────────
with tab_pred:
    st.header("📊Batch Prediction")
    models_available = list_available_models()
    choice = st.selectbox("Choose a trained model", models_available)
    if choice and (ss.model_name != choice or ss.model is None):
        ss.model_name, ss.model = choice, load_selected_model(choice)

    if ss.prep is None:
        st.info("Need preprocessed data.")
    elif ss.model is None:
        st.error("Select or train a model.")
    else:
        if st.button("Predict"):
            with st.spinner("Predicting…"):
                X_pred = ss.prep.drop(columns=["fraud_reported"], errors="ignore")
                X_pred = align_to_model(numeric_impute(X_pred), ss.model)
                ss.pred_df = ss.prep.copy()
                ss.pred_df["Prediction"] = ss.model.predict(X_pred)
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
# 6 · 📝 Single Prediction
# ──────────────────────────────────────────────────────────
with tab_single:
    st.header("📝Single Prediction (Probability)")
    if ss.model is None:
        st.error("Select or train a model first.")
    elif ss.raw is None:
        st.info("Load a dataset first.")
    else:
        raw_df, model = ss.raw, ss.model
        top_cols = ss.top_features or list(model.feature_names_in_)

        with st.form("single_pred_form"):
            inputs: dict[str, object] = {}
            for col in top_cols:
                if "date" in col.lower():
                    default_date = pd.to_datetime(
                        raw_df[col].dropna().iloc[0] if col in raw_df.columns else "2020-01-01"
                    )
                    inputs[col] = st.date_input(col, value=default_date).strftime("%m/%d/%Y")
                    continue

                if col not in raw_df.columns:
                    inputs[col] = st.number_input(col, value=0.0)
                    continue

                series = raw_df[col]
                if series.dtype == object:
                    opts = sorted(series.dropna().unique().tolist())
                    inputs[col] = (
                        st.selectbox(col, options=opts) if 2 < len(opts) <= 25
                        else st.text_input(col)
                    )
                else:
                    is_int = pd.api.types.is_integer_dtype(series) or (
                        series.dropna() % 1 == 0
                    ).all()
                    default = float(series.median())
                    inputs[col] = st.number_input(
                        col, value=int(default) if is_int else default,
                        step=1 if is_int else 0.01,
                    )
            submitted = st.form_submit_button("Predict")

        if submitted:
            try:
                new_raw = pd.DataFrame([inputs])
                tmp_proc = cached_preprocess(
                    pd.concat([raw_df.head(1).copy(), new_raw], ignore_index=True)
                )
                single = tmp_proc.tail(1).drop(columns=["fraud_reported"], errors="ignore")
                single = align_to_model(numeric_impute(single), model)
                proba = (
                    model.predict_proba(single)[0][1]
                    if hasattr(model, "predict_proba") else model.predict(single)[0]
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
                st.plotly_chart(gauge)
                st.success(
                    f"Estimated fraud probability: **{percent:.2f}%** "
                    f"({'Fraud' if percent >= 50 else 'Not Fraud'})"
                )
            except Exception as exc:
                st.error(f"Prediction failed: {exc}")

# ──────────────────────────────────────────────────────────
# 7 · Explain
# ──────────────────────────────────────────────────────────
with tab_explain:
    st.header("🔎SHAP Explainability")
    if ss.prep is None:
        st.info("Need preprocessed data.")
    elif ss.model is None:
        st.error("Select or train a model first.")
    else:
        if st.button("Generate SHAP"):
            with st.spinner("Calculating SHAP…"):
                try:
                    X_exp = ss.prep.drop(columns=["fraud_reported"], errors="ignore")
                    X_exp = align_to_model(numeric_impute(X_exp), ss.model)
                    plt_orig_show = plt.show
                    plt.show = lambda *a, **k: st.pyplot(plt.gcf(), clear_figure=True)
                    explain_model_shap(ss.model, X_exp)
                    plt.show = plt_orig_show
                except Exception as exc:
                    st.error(f"SHAP computation failed: {exc}")
