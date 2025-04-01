import logging
import warnings
warnings.filterwarnings("ignore")

# Configure logging once here
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)

import numpy as np
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE

# Local module imports
from data_handling import load_dataset, validate_required_columns
from preprocessing import preprocess_data, eda_plots
from feature_engineering import feature_engineering, select_top_features
from model_training import train_and_evaluate_models, hyperparameter_tuning_rf
from evaluation import explain_model_shap

def main():
    # === Step 1: Load Data ===
    file_path = "insurance_claims.csv"
    df = load_dataset(file_path)
    if df.empty:
        logging.error("Exiting due to empty dataset.")
        return

    # === Step 2: Validate Required Columns ===
    required_columns = ["months_as_customer", "age", "policy_number", "fraud_reported"]
    if not validate_required_columns(df, required_columns):
        logging.error("Exiting due to missing required columns.")
        return

    # === Step 3: Preprocess Data ===
    df = preprocess_data(df, target_column="fraud_reported")

    # === Step 4: EDA ===
    eda_plots(df, target_column="fraud_reported")

    # === Step 5: Feature Engineering ===
    df = feature_engineering(df)

    # === Step 6: Feature Selection (Optional) ===
    top_features = select_top_features(df, target="fraud_reported", top_n=20)
    if top_features:
        df = df[top_features + ["fraud_reported"]]

    # === Step 7: Train-Test Split & Handle Imbalance ===
    X = df.drop("fraud_reported", axis=1)
    y = df["fraud_reported"]

    sm = SMOTE(random_state=42)
    try:
        X_res, y_res = sm.fit_resample(X, y)
    except Exception as e:
        logging.error(f"SMOTE error: {e}. Proceeding without SMOTE.")
        X_res, y_res = X, y

    X_train, X_test, y_train, y_test = train_test_split(
        X_res, y_res, test_size=0.2, random_state=42
    )

    # === Step 8: Train & Evaluate Models ===
    train_and_evaluate_models(X_train, X_test, y_train, y_test)

    # === Step 9: Hyperparameter Tuning (Optional) ===
    best_rf = hyperparameter_tuning_rf(X_train, X_test, y_train, y_test)

    # === Step 10: Model Interpretation (SHAP) ===
    if best_rf is not None:
        explain_model_shap(best_rf, X_test)


if __name__ == "__main__":
    main()
