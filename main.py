from __future__ import annotations
import logging, warnings
from pathlib import Path
import joblib, numpy as np, pandas as pd

# ── Matplotlib (headless, to save without popup) ────────────────────────────
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
FIG_DIR = Path("figures"); FIG_DIR.mkdir(exist_ok=True)
_plot_i = 0
def _save_and_close(*_a, **_k):
    global _plot_i
    fname = FIG_DIR / f"fig_{_plot_i:03d}.png"
    plt.savefig(fname, dpi=300, bbox_inches="tight")
    plt.close(); logging.info(f"Saved figure → {fname}")
    _plot_i += 1
plt.show = _save_and_close

# ── Sklearn / Imbalanced‑learn ──────────────────────────────────
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score, classification_report
from imblearn.over_sampling import SMOTE
from imblearn.ensemble import BalancedRandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier, StackingClassifier, IsolationForest
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV

# ── Local modules ───────────────────────────────────────────────
from data_handling import load_dataset, validate_required_columns
from preprocessing import eda_plots
from feature_engineering import feature_engineering, select_top_features
from model_training import train_and_evaluate_models, hyperparameter_tuning_rf, evaluate_unsupervised_model
from evaluation import explain_model_shap, plot_lift_curve

# ── Logging ─────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y‑%m‑%d %H:%M:%S",
)
warnings.filterwarnings("ignore")
MODELS_DIR = Path("models"); MODELS_DIR.mkdir(exist_ok=True)

# ═══════════════════════════════════════════════════════════════
# Utility helpers
# ═══════════════════════════════════════════════════════════════
def basic_clean(df: pd.DataFrame, target: str = "fraud_reported") -> pd.DataFrame:
    if df[target].dtype == object:
        df[target] = df[target].map({"Y": 1, "N": 0})
    num_cols = df.select_dtypes("number").columns.tolist()
    cat_cols = df.select_dtypes("object").columns.tolist()
    for col in num_cols: df[col].fillna(df[col].median(), inplace=True)
    for col in cat_cols: df[col].fillna(df[col].mode()[0], inplace=True)
    if "umbrella_limit" in df.columns:
        hi = df["umbrella_limit"].quantile(0.99)
        df.loc[df["umbrella_limit"] > hi, "umbrella_limit"] = hi
    return df

def make_preprocessor(df, target="fraud_reported"):
    num_cols = [c for c in df.select_dtypes("number").columns if c != target]
    cat_cols = [c for c in df.select_dtypes("object").columns if c != target]
    num_pipe = Pipeline([("imp", SimpleImputer(strategy="median")),
                         ("sc",  StandardScaler())])
    cat_pipe = Pipeline([("imp", SimpleImputer(strategy="most_frequent")),
                         ("oh",  OneHotEncoder(handle_unknown="ignore"))])
    return ColumnTransformer(
        [("num", num_pipe, num_cols),
         ("cat", cat_pipe, cat_cols)],
        remainder="drop",
        verbose_feature_names_out=False,
    )
# ═══════════════════════════════════════════════════════════════
# Main workflow
# ═══════════════════════════════════════════════════════════════
def main() -> None:
    # 1 · Load & sanity check
    df = load_dataset("insurance_claims.csv")
    if df.empty or not validate_required_columns(df, ["fraud_reported"]):
        return

    # 2 · Clean + feature engineering
    df = feature_engineering(basic_clean(df))
    eda_plots(df, target_column="fraud_reported")

    # 3 · Optional top‑N feature pruning
    feats = select_top_features(df, "fraud_reported", top_n=20)
    if feats: df = df[feats + ["fraud_reported"]]

    # 4 · Train/test split (no leakage)
    X_raw, y = df.drop("fraud_reported", axis=1), df["fraud_reported"]
    X_tr_raw, X_te_raw, y_tr, y_te = train_test_split(
        X_raw, y, test_size=0.2, random_state=42, stratify=y
    )

    # 5 · Encode
    preproc = make_preprocessor(X_tr_raw)
    X_tr_enc = preproc.fit_transform(X_tr_raw)
    X_te_enc = preproc.transform(X_te_raw)
    if hasattr(X_tr_enc, "toarray"):
        X_tr_enc, X_te_enc = X_tr_enc.toarray(), X_te_enc.toarray()
    feat_names = preproc.get_feature_names_out()
    X_tr = pd.DataFrame(X_tr_enc, columns=feat_names, index=X_tr_raw.index)
    X_te = pd.DataFrame(X_te_enc, columns=feat_names, index=X_te_raw.index)

    # 6 · SMOTE (train only)
    X_tr_bal, y_tr_bal = SMOTE(random_state=42).fit_resample(X_tr, y_tr)

    # 7 · Baseline models (existing helpers)
    train_and_evaluate_models(X_tr_bal, X_te, y_tr_bal, y_te)
    best_rf = hyperparameter_tuning_rf(X_tr_bal, X_te, y_tr_bal, y_te)

    # 8 · New high‑accuracy models
    brf = BalancedRandomForestClassifier(
        n_estimators=400, max_depth=None, random_state=42, n_jobs=-1
    ).fit(X_tr_bal, y_tr_bal)

    gb  = GradientBoostingClassifier(random_state=42).fit(X_tr_bal, y_tr_bal)

    stack = StackingClassifier(
        estimators=[("brf", brf), ("gb", gb)],
        final_estimator=LogisticRegression(max_iter=1000),
        stack_method="predict_proba",
        n_jobs=-1,
        cv=StratifiedKFold(5, shuffle=True, random_state=42),
    ).fit(X_tr_bal, y_tr_bal)

    cal_stack = CalibratedClassifierCV(
        stack, method="isotonic", cv=StratifiedKFold(5, shuffle=True, random_state=42)
    ).fit(X_tr_bal, y_tr_bal)

    # 9 · Evaluate & pick the best
    candidates = {
        "BalancedRF": brf,
        "GradientBoost": gb,
        "CalibratedStack": cal_stack,
    }
    if best_rf: candidates["TunedRF"] = best_rf

    best_name, best_auc, best_model = None, -np.inf, None
    for name, mdl in candidates.items():
        proba = mdl.predict_proba(X_te)[:, 1]
        auc   = roc_auc_score(y_te, proba)
        logging.info(f"{name:<15} ROC‑AUC = {auc:0.4f}")
        if auc > best_auc: best_name, best_auc, best_model = name, auc, mdl

    # 10 · Explainability + lift
    if best_name in {"TunedRF", "BalancedRF"}:
        explain_model_shap(best_model, X_te)
    plot_lift_curve(y_te, best_model.predict_proba(X_te)[:, 1], model_name=best_name)

    # 11 · Anomaly detector
    iso = IsolationForest(contamination=0.01, random_state=42).fit(X_tr_bal)
    evaluate_unsupervised_model(iso, X_te, y_te, "Isolation Forest")

    # 12 · Persist artifacts
    joblib.dump(best_model, MODELS_DIR / "best_model.pkl")
    joblib.dump(preproc,   MODELS_DIR / "preprocessor.pkl")
    logging.info(f"Saved best model ({best_name}) → models/best_model.pkl")
    logging.info("Saved ColumnTransformer → models/preprocessor.pkl")

# ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    main()
