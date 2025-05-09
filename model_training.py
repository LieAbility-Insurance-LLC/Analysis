import logging
import matplotlib.pyplot as plt
import numpy as np

from sklearn.metrics import (
    roc_auc_score,
    roc_curve,
    precision_recall_curve,
    average_precision_score,
)

# -------- supervised learners ----------------------------------
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier,
)
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier

# -------- unsupervised / anomaly / clustering ------------------
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.svm import OneClassSVM
from sklearn.cluster import KMeans, DBSCAN

# -------- project helpers --------------------------------------
from evaluation import (
    evaluate_model,
    plot_pr_curve,
    plot_calibration_curve,
    plot_cumulative_gain,
)

# ────────────────────────────────────────────────────────────────
# Generic helpers
# ────────────────────────────────────────────────────────────────
def _plot_roc_pr_curves(y_true, scores, model_name: str):
    """Utility for ROC & PR curves when we only have continuous scores."""
    roc_auc = roc_auc_score(y_true, scores)
    fpr, tpr, _ = roc_curve(y_true, scores)
    precision, recall, _ = precision_recall_curve(y_true, scores)
    ap = average_precision_score(y_true, scores)

    plt.figure()
    plt.plot(fpr, tpr, label=f"ROC AUC = {roc_auc:.2f}")
    plt.plot([0, 1], [0, 1], "k--")
    plt.xlabel("FPR")
    plt.ylabel("TPR")
    plt.title(f"ROC Curve · {model_name}")
    plt.legend()
    plt.tight_layout()
    plt.show()

    plt.figure()
    plt.plot(recall, precision, label=f"AP = {ap:.2f}")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title(f"Precision-Recall · {model_name}")
    plt.legend()
    plt.tight_layout()
    plt.show()


def evaluate_unsupervised_model(model, X_test, y_test, model_name="Unsupervised Model"):
    """
    Evaluate an outlier -detection model that provides *continuous* anomaly scores
    via ``decision_function`` or ``score_samples``. Higher score → more anomalous.
    """
    # 1 · scores --------------------------------------------------
    if hasattr(model, "decision_function"):
        scores = model.decision_function(X_test)
    elif hasattr(model, "score_samples"):
        scores = -model.score_samples(X_test)  # invert: high = outlier
    else:
        raise ValueError(
            f"{model_name} has no decision_function / score_samples interface."
        )

    # 2 · threshold → hard labels (median is a quick heuristic)
    thresh = np.median(scores)
    y_pred = (scores >= thresh).astype(int)  # 1 = anomaly / fraud

    # 3 · metrics -------------------------------------------------
    evaluate_model(y_test, y_pred, model_name=model_name)
    _plot_roc_pr_curves(y_test, scores, model_name)


def _evaluate_dbscan(dbscan_labels, y_test, model_name="DBSCAN"):
    """
    DBSCAN produces cluster labels; -1 denotes noise. Treat noise → fraud.
    """
    y_pred = (dbscan_labels == -1).astype(int)
    evaluate_model(y_test, y_pred, model_name=model_name)


# ────────────────────────────────────────────────────────────────
# Primary API called from main.py / Streamlit
# ────────────────────────────────────────────────────────────────
def train_and_evaluate_models(X_train, X_test, y_train, y_test):
    """
    Runs a battery of supervised + unsupervised models and logs / plots
    their performance. Keeps the original Random Forest + XGBoost stack
    but widens coverage with simple, explainable learners.
    """
    # ============================================================
    # 1 · Supervised models
    # ============================================================
    models_sup = {
        # ------ baseline stack (kept) ---------------------------
        "Logistic Regression": LogisticRegression(max_iter=200, random_state=42),
        "Decision Tree": DecisionTreeClassifier(max_depth=None, random_state=42),
        "Random Forest": RandomForestClassifier(random_state=42),
        "Gradient Boosting": GradientBoostingClassifier(random_state=42),
        "XGBoost": XGBClassifier(
            random_state=42, use_label_encoder=False, eval_metric="logloss"
        ),
        "SVM (RBF)": SVC(kernel="rbf", probability=True, random_state=42),
    }

    for name, model in models_sup.items():
        try:
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            evaluate_model(y_test, y_pred, model_name=name)

            # --- curves ----------------------------------------
            if hasattr(model, "predict_proba"):
                y_score = model.predict_proba(X_test)[:, 1]
            elif hasattr(model, "decision_function"):
                y_score = model.decision_function(X_test)
            else:
                y_score = y_pred.astype(float)

            plot_pr_curve(y_test, y_score, model_name=name)
            plot_calibration_curve(model, X_test, y_test, model_name=name)
            plot_cumulative_gain(y_test, y_score, model_name=name)
        except Exception as exc:
            logging.error(f"{name} failed: {exc}")

    # ============================================================
    # 2 · Unsupervised / Anomaly -detection models
    # ============================================================
    logging.info("🛰 Evaluating unsupervised models (outliers = fraud)…")

    # ---------- Isolation Forest (kept) --------------------------
    try:
        iso = IsolationForest(contamination=0.01, random_state=42)
        iso.fit(X_train)
        evaluate_unsupervised_model(
            iso, X_test, y_test, model_name="Isolation Forest"
        )
    except Exception as exc:
        logging.error(f"Isolation Forest error: {exc}")

    # ---------- Local Outlier Factor ----------------------------
    try:
        lof = LocalOutlierFactor(n_neighbors=20, novelty=True)
        lof.fit(X_train)
        evaluate_unsupervised_model(
            lof, X_test, y_test, model_name="Local Outlier Factor"
        )
    except Exception as exc:
        logging.error(f"Local Outlier Factor error: {exc}")

    # ---------- One -Class SVM -----------------------------------
    try:
        ocsvm = OneClassSVM(kernel="rbf", nu=0.05, gamma="auto")
        ocsvm.fit(X_train)
        evaluate_unsupervised_model(ocsvm, X_test, y_test, model_name="One-Class SVM")
    except Exception as exc:
        logging.error(f"One-Class SVM error: {exc}")

    # ---------- K -Means  (binary cluster) -----------------------
    try:
        km = KMeans(n_clusters=2, random_state=42)
        km.fit(X_train)
        y_pred_km = km.predict(X_test)
        evaluate_model(y_test, y_pred_km, model_name="K -Means (2-cluster)")
    except Exception as exc:
        logging.error(f"K -Means error: {exc}")

    # ---------- DBSCAN  (density clustering) --------------------
    try:
        db = DBSCAN(eps=0.7, min_samples=10)
        db.fit(X_train)
        _evaluate_dbscan(db.labels_, y_train, model_name="DBSCAN (train)")

        db_test_lbl = db.fit_predict(X_test)  # fit on test to get labels
        _evaluate_dbscan(db_test_lbl, y_test, model_name="DBSCAN (test)")
    except Exception as exc:
        logging.error(f"DBSCAN error: {exc}")


# ────────────────────────────────────────────────────────────────
# 3 · Random Forest hyper -tuning helper (unchanged)
# ────────────────────────────────────────────────────────────────
from sklearn.model_selection import GridSearchCV


def hyperparameter_tuning_rf(X_train, X_test, y_train, y_test):
    """
    Simple GridSearch over a handful of RF hyper -parameters.
    Returns the best estimator or None on failure.
    """
    grid = {
        "n_estimators": [100, 200],
        "max_depth": [None, 10, 20],
        "min_samples_split": [2, 5],
    }
    try:
        gs = GridSearchCV(
            RandomForestClassifier(random_state=42),
            param_grid=grid,
            scoring="f1",
            cv=3,
            n_jobs=-1,
        )
        gs.fit(X_train, y_train)
        best = gs.best_estimator_
        logging.info(f"Best RF params → {gs.best_params_}")
        y_pred = best.predict(X_test)
        evaluate_model(y_test, y_pred, model_name="Tuned Random Forest")
        return best
    except Exception as exc:
        logging.error(f"RF tuning failed: {exc}")
        return None
