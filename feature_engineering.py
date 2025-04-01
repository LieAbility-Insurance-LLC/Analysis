# feature_engineering.py

import logging
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier

def feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds or transforms features in the DataFrame based on domain knowledge.
    """
    # Example: is_rush_hour
    if 'incident_hour_of_the_day' in df.columns:
        df['is_rush_hour'] = df['incident_hour_of_the_day'].apply(
            lambda x: 1 if 7 <= x <= 9 or 16 <= x <= 18 else 0
        )
    else:
        logging.info("Column 'incident_hour_of_the_day' not found. "
                     "Skipping is_rush_hour feature.")

    logging.info("Feature engineering complete.")
    return df


def select_top_features(df: pd.DataFrame, target: str, top_n: int = 20, log_importances: bool = True):
    """
    Select top N features based on a RandomForest importance ranking.
    Returns a list of the top features.
    """
    if df.empty:
        logging.warning("Feature selection skipped. DataFrame is empty.")
        return []

    if target not in df.columns:
        logging.error(f"Target '{target}' not found in DataFrame.")
        return []

    X_temp = df.drop(target, axis=1)
    y_temp = df[target]

    if X_temp.shape[1] < 1:
        logging.error("No features available for feature selection.")
        return []

    rf_temp = RandomForestClassifier(random_state=42)
    try:
        rf_temp.fit(X_temp, y_temp)
        importances = rf_temp.feature_importances_
        features = X_temp.columns
        feat_imp = sorted(zip(features, importances), key=lambda x: x[1], reverse=True)

        if log_importances:
            logging.info("All Feature Importances (sorted):")
            for f, imp in feat_imp:
                logging.info(f"  {f}: {imp:.5f}")

        top_features = [f[0] for f in feat_imp[:top_n]]
        logging.info(f"Selected top {top_n} features via RandomForest.")
        return top_features
    except Exception as e:
        logging.error(f"Error in feature selection: {e}")
        return []
