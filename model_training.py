# model_training.py

import logging
import numpy as np

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from xgboost import XGBClassifier
from sklearn.cluster import KMeans
from sklearn.model_selection import GridSearchCV
from evaluation import evaluate_model

def train_and_evaluate_models(X_train, X_test, y_train, y_test):
    """
    Train and evaluate a few supervised and unsupervised models.
    """
    # -------------------
    # SUPERVISED MODELS
    # -------------------
    models = {
        "Logistic Regression": LogisticRegression(random_state=42),
        "Random Forest": RandomForestClassifier(random_state=42),
        "XGBoost": XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='logloss')
    }

    for name, model in models.items():
        try:
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            evaluate_model(y_test, y_pred, model_name=name)
        except Exception as e:
            logging.error(f"Error training {name}: {e}")

    # -------------------
    # UNSUPERVISED MODELS
    # -------------------
    logging.info("Evaluating Unsupervised Models (outliers/clusters as 'fraud')")

    # Isolation Forest
    try:
        iso_forest = IsolationForest(contamination=0.01, random_state=42)
        iso_forest.fit(X_train)
        y_pred_iso = iso_forest.predict(X_test)
        # Map -1 (anomaly) to 1 (fraud), otherwise 0
        y_pred_iso = np.where(y_pred_iso == -1, 1, 0)
        evaluate_model(y_test, y_pred_iso, "Isolation Forest")
    except Exception as e:
        logging.error(f"Error training Isolation Forest: {e}")

    # K-Means
    try:
        kmeans = KMeans(n_clusters=2, random_state=42)
        kmeans.fit(X_train)
        y_pred_km = kmeans.predict(X_test)
        # Assume cluster "1" = fraud, "0" = not fraud
        evaluate_model(y_test, y_pred_km, "K-Means")
    except Exception as e:
        logging.error(f"Error training K-Means: {e}")


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
