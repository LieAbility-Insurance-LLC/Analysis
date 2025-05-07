import logging
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, roc_curve, precision_recall_curve, average_precision_score

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from xgboost import XGBClassifier
from sklearn.cluster import KMeans
from sklearn.model_selection import GridSearchCV
from evaluation import evaluate_model

def evaluate_unsupervised_model(model, X_test, y_test, model_name="Unsupervised Model"):
    """
    Evaluate an unsupervised anomaly detection model using its continuous anomaly scores.
    This function plots the ROC and Precision-Recall curves and prints the associated metrics.
    
    Parameters:
        model: The unsupervised model (e.g., IsolationForest) with decision_function or score_samples.
        X_test: Test features (pandas DataFrame or numpy array).
        y_test: True binary labels (0: normal, 1: anomaly).
        model_name: A string name for the model (used in plot titles and printouts).
    """
    # Get continuous anomaly scores.
    # Prefer decision_function if available, otherwise use score_samples (inverted so that higher means more anomalous)
    if hasattr(model, "decision_function"):
        anomaly_scores = model.decision_function(X_test)
    elif hasattr(model, "score_samples"):
        anomaly_scores = -model.score_samples(X_test)
    else:
        raise ValueError("Model must have either a decision_function or score_samples method.")
    
    # Compute ROC-AUC
    roc_auc = roc_auc_score(y_test, anomaly_scores)
    print(f"{model_name} ROC-AUC: {roc_auc:.4f}")
    
    # Plot ROC Curve
    fpr, tpr, _ = roc_curve(y_test, anomaly_scores)
    plt.figure()
    plt.plot(fpr, tpr, label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curve for {model_name}')
    plt.legend(loc="lower right")
    plt.show()
    
    # Compute Precision-Recall metrics
    precision, recall, _ = precision_recall_curve(y_test, anomaly_scores)
    avg_precision = average_precision_score(y_test, anomaly_scores)
    print(f"{model_name} Average Precision: {avg_precision:.4f}")
    
    # Plot Precision-Recall Curve
    plt.figure()
    plt.plot(recall, precision, label=f'Precision-Recall (AP = {avg_precision:.2f})')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title(f'Precision-Recall Curve for {model_name}')
    plt.legend(loc="upper right")
    plt.show()


def train_and_evaluate_models(X_train, X_test, y_train, y_test):
    # ──────────────────────────────────────────────────────────
    # Local imports – avoids circular dependencies
    # ──────────────────────────────────────────────────────────
    from evaluation import (
        evaluate_model,
        plot_pr_curve,
        plot_calibration_curve,
        plot_cumulative_gain,
    )
    import logging
    import numpy as np

    # -------------------
    # SUPERVISED MODELS
    # -------------------
    models = {
        "Logistic Regression": LogisticRegression(random_state=42),
        "Random Forest": RandomForestClassifier(random_state=42),
        "XGBoost": XGBClassifier(
            random_state=42,
            use_label_encoder=False,
            eval_metric="logloss",
        ),
    }

    for name, model in models.items():
        try:
            # ── fit & hard predictions ───────────────────────────
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            evaluate_model(y_test, y_pred, model_name=name)

            # ── probability / score vector (for curves) ─────────
            if hasattr(model, "predict_proba"):
                y_score = model.predict_proba(X_test)[:, 1]
            elif hasattr(model, "decision_function"):
                y_score = model.decision_function(X_test)
            else:
                # fall back to hard predictions (step‑function curves)
                y_score = y_pred.astype(float)

            # ── advanced diagnostic plots ───────────────────────
            plot_pr_curve(y_test, y_score, model_name=name)
            plot_calibration_curve(model, X_test, y_test, model_name=name)
            plot_cumulative_gain(y_test, y_score, model_name=name)

        except Exception as e:
            logging.error(f"Error training {name}: {e}")

    # -------------------
    # UNSUPERVISED MODELS
    # -------------------
    logging.info("Evaluating Unsupervised Models (outliers/clusters as 'fraud')")

    # Isolation Forest – uses continuous anomaly scores
    try:
        iso_forest = IsolationForest(contamination=0.01, random_state=42)
        iso_forest.fit(X_train)
        evaluate_unsupervised_model(
            iso_forest,
            X_test,
            y_test,
            model_name="Isolation Forest",
        )
    except Exception as e:
        logging.error(f"Error training Isolation Forest: {e}")

    # K‑Means – still evaluated via hard predictions
    try:
        kmeans = KMeans(n_clusters=2, random_state=42)
        kmeans.fit(X_train)
        y_pred_km = kmeans.predict(X_test)
        # Assume cluster “1” → fraud, “0” → non‑fraud
        evaluate_model(y_test, y_pred_km, model_name="K‑Means")
    except Exception as e:
        logging.error(f"Error training K‑Means: {e}")


def hyperparameter_tuning_rf(X_train, X_test, y_train, y_test):
    """
    Hyperparameter tuning for RandomForestClassifier using GridSearchCV.
    """
    param_grid = {
        'n_estimators': [100, 200],
        'max_depth': [None, 10, 20],
        'min_samples_split': [2, 5]
    }
    try:
        grid_search = GridSearchCV(
            estimator=RandomForestClassifier(random_state=42),
            param_grid=param_grid,
            scoring='f1',
            cv=3,
            n_jobs=-1
        )
        grid_search.fit(X_train, y_train)
        best_model = grid_search.best_estimator_
        logging.info(f"Best RF Params: {grid_search.best_params_}")

        y_pred_best = best_model.predict(X_test)
        evaluate_model(y_test, y_pred_best, "Tuned RandomForest")
        return best_model
    except Exception as e:
        logging.error(f"Error in hyperparameter tuning: {e}")
        return None
