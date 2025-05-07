# main.py · v2.2  (plots saved to figures/*.png, no GUI backend)
# ──────────────────────────────────────────────────────────
import logging
import warnings
import os
from pathlib import Path

warnings.filterwarnings("ignore")

# ──────────────────────────────────────────────────────────
# Matplotlib – use non‑GUI backend and auto‑save every figure
# ──────────────────────────────────────────────────────────
import joblib
import matplotlib

matplotlib.use("Agg")  # avoids Tkinter, no GUI windows
import matplotlib.pyplot as plt

FIG_DIR = Path("figures")
FIG_DIR.mkdir(exist_ok=True)
_plot_counter = {"i": 0}


def _save_and_close(*_args, **_kwargs):
    """
    Replacement for plt.show() – saves the current figure and closes it.
    """
    idx = _plot_counter["i"]
    fname = FIG_DIR / f"fig_{idx:03d}.png"
    plt.savefig(fname, dpi=300, bbox_inches="tight")
    plt.close()
    logging.info(f"Saved figure → {fname}")
    _plot_counter["i"] += 1


# Monkey‑patch matplotlib’s global show function
plt.show = _save_and_close

# ──────────────────────────────────────────────────────────
# Logging
# ──────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

# ──────────────────────────────────────────────────────────
# Third‑party libs
# ──────────────────────────────────────────────────────────
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import IsolationForest
from imblearn.over_sampling import SMOTE

# ──────────────────────────────────────────────────────────
# Local modules
# ──────────────────────────────────────────────────────────
from data_handling import load_dataset, validate_required_columns
from preprocessing import preprocess_data, eda_plots
from feature_engineering import (
    feature_engineering,
    select_top_features,
)
from model_training import (
    train_and_evaluate_models,
    hyperparameter_tuning_rf,
    evaluate_unsupervised_model,
)
from evaluation import explain_model_shap

# ──────────────────────────────────────────────────────────
# Main driver
# ──────────────────────────────────────────────────────────
def main() -> None:
    # === 1 · Load ===========================================================
    df = load_dataset("insurance_claims.csv")
    if df.empty:
        logging.error("Exiting – empty dataset.")
        return

    # === 2 · Sanity check ===================================================
    must_have = ["months_as_customer", "age", "policy_number", "fraud_reported"]
    if not validate_required_columns(df, must_have):
        logging.error("Exiting – required columns missing.")
        return

    # === 3 · Pre‑process ====================================================
    df = preprocess_data(df, target_column="fraud_reported")

    # === 4 · Feature engineering ===========================================
    df = feature_engineering(df)

    # === 5 · Top‑N feature selection =======================================
    top_features = select_top_features(df, target="fraud_reported", top_n=20)

    # === 6 · EDA (restricted heat‑map) =====================================
    corr_cols = top_features[:9] if top_features else None
    eda_plots(df, target_column="fraud_reported", corr_cols=corr_cols)

    # keep only top predictors for modelling
    if top_features:
        df = df[top_features + ["fraud_reported"]]

    # === 7 · Train / test split + SMOTE ====================================
    X, y = df.drop("fraud_reported", axis=1), df["fraud_reported"]

    try:
        X, y = SMOTE(random_state=42).fit_resample(X, y)
    except Exception as e:
        logging.error(f"SMOTE failed: {e} – continuing without resampling.")

    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # === 8 · Model training & baseline eval ================================
    train_and_evaluate_models(X_tr, X_te, y_tr, y_te)

    # === 9 · RF tuning ======================================================
    best_rf = hyperparameter_tuning_rf(X_tr, X_te, y_tr, y_te)

    # === 10 · Isolation Forest (unsupervised) ==============================
    try:
        iso = IsolationForest(contamination=0.01, random_state=42).fit(X_tr)
        evaluate_unsupervised_model(
            iso, X_te, y_te, model_name="Isolation Forest (stand‑alone)"
        )
    except Exception as e:
        logging.error(f"Isolation Forest error: {e}")

    # === 11 · SHAP explainability ==========================================
    if best_rf is not None:
        explain_model_shap(best_rf, X_te)

    # === 12 · Persist tuned model ==========================================
    if best_rf is not None:
        os.makedirs("models", exist_ok=True)
        joblib.dump(best_rf, "models/best_rf_model.pkl")
        logging.info("Saved tuned RandomForest → models/best_rf_model.pkl")


if __name__ == "__main__":
    main()
