import os
import logging
import warnings

warnings.filterwarnings("ignore")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)

##############################################################################
# 1. LIBRARY IMPORTS WITH FALLBACKS
##############################################################################
# If certain packages are missing, the script will attempt to install or skip.

def import_or_install(package):
    """
    Attempts to import a package; if not found, install it via pip.
    """
    import importlib
    try:
        importlib.import_module(package)
        logging.info(f"Successfully imported {package}.")
    except ImportError:
        logging.warning(f"{package} not installed. Installing now...")
        import subprocess
        subprocess.check_call(["python", "-m", "pip", "install", package])
        logging.info(f"Restart the script if {package} was just installed.")

# List of packages to ensure availability
packages_to_check = ["pandas", "numpy", "matplotlib", "seaborn", 
                     "sklearn", "imblearn", "xgboost", "shap"]
for pkg in packages_to_check:
    import_or_install(pkg)

# Now safely import
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier

# Imbalanced learning
from imblearn.over_sampling import SMOTE

# Unsupervised
from sklearn.ensemble import IsolationForest
from sklearn.cluster import KMeans

# Model evaluation
from sklearn.metrics import (classification_report, confusion_matrix,
                             roc_auc_score)

# Try importing shap
try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    logging.warning("SHAP is not installed or failed to import. "
                    "Skipping model explanation steps.")
    SHAP_AVAILABLE = False

##############################################################################
# 2. HELPER FUNCTIONS
##############################################################################

def load_dataset(file_path: str) -> pd.DataFrame:
    """
    Loads the dataset into a pandas DataFrame.
    Includes error handling if file does not exist or is unreadable.

    :param file_path: Path to the CSV file.
    :return: DataFrame with loaded data.
    """
    if not os.path.exists(file_path):
        logging.error(f"File '{file_path}' not found. Please verify the path.")
        return pd.DataFrame()  # Return empty DataFrame as fallback

    try:
        df = pd.read_csv(file_path)
        logging.info(f"Loaded dataset with shape {df.shape} from {file_path}")
        return df
    except Exception as e:
        logging.error(f"Failed to read CSV file: {e}")
        return pd.DataFrame()  # Return empty DataFrame as fallback


def validate_required_columns(df: pd.DataFrame, required_cols: list) -> bool:
    """
    Checks if required columns are present in the DataFrame.

    :param df: Input DataFrame.
    :param required_cols: List of required column names.
    :return: True if all columns exist, False otherwise.
    """
    df_cols = set(df.columns)
    missing = [col for col in required_cols if col not in df_cols]
    if missing:
        logging.error(f"Missing required columns: {missing}")
        return False
    return True


def preprocess_data(df: pd.DataFrame,
                    target_column: str = "fraud_reported",
                    cap_outliers: bool = True) -> pd.DataFrame:
    """
    Cleans and preprocesses the dataset:
      - Converts 'Y'/'N' target to 1/0 if needed
      - Logs columns with missing values
      - Imputes numeric columns (median) and categorical columns (mode)
      - Optionally caps outliers in 'umbrella_limit'
      - Encodes categorical columns
      - Scales numeric columns

    :param df: Input DataFrame
    :param target_column: Name of the target column
    :param cap_outliers: Whether to cap outliers for 'umbrella_limit'
    :return: Preprocessed DataFrame
    """
    if df.empty:
        logging.warning("Received empty DataFrame for preprocessing. Skipping.")
        return df

    # Convert target from 'Y'/'N' to 1/0
    if df[target_column].dtype == object:
        df[target_column] = df[target_column].map({'Y': 1, 'N': 0})

    # Identify missing values
    missing_counts = df.isnull().sum()
    for col, count in missing_counts.items():
        if count > 0:
            logging.info(f"Column '{col}' has {count} missing values. Imputing...")

    # Separate numeric vs categorical
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = df.select_dtypes(include=['object']).columns.tolist()

    # Impute numeric columns
    for col in numeric_cols:
        df[col].fillna(df[col].median(), inplace=True)

    # Impute categorical columns
    for col in categorical_cols:
        df[col].fillna(df[col].mode()[0], inplace=True)

    # Outlier capping
    if cap_outliers and 'umbrella_limit' in df.columns:
        q_high = df['umbrella_limit'].quantile(0.99)
        df.loc[df['umbrella_limit'] > q_high, 'umbrella_limit'] = q_high
        logging.info("Applied outlier capping on 'umbrella_limit' at 99th percentile.")

    # Encode categorical features
    label_encoders = {}
    for col in categorical_cols:
        le = LabelEncoder()
        try:
            df[col] = le.fit_transform(df[col].astype(str))
            label_encoders[col] = le
        except Exception as e:
            logging.error(f"Error encoding column '{col}': {e}")

    # Scale numeric columns, excluding target if present
    if target_column in numeric_cols:
        numeric_cols.remove(target_column)

    scaler = StandardScaler()
    df[numeric_cols] = scaler.fit_transform(df[numeric_cols])

    logging.info("Data preprocessing complete.")
    return df



def eda_plots(df: pd.DataFrame, target_column: str = "fraud_reported"):
    """
    Basic Exploratory Data Analysis:
    - Class distribution
    - Correlation heatmap

    :param df: Preprocessed DataFrame
    :param target_column: Name of the target column
    """
    if df.empty:
        logging.warning("EDA skipped. DataFrame is empty.")
        return

    plt.figure(figsize=(6, 4))
    sns.countplot(x=target_column, data=df)
    plt.title('Class Distribution')
    plt.show()

    # Correlation matrix
    plt.figure(figsize=(12, 8))
    sns.heatmap(df.corr(), cmap='coolwarm', annot=False)
    plt.title('Correlation Heatmap')
    plt.show()


def feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    """
    Example feature engineering. Modify based on domain knowledge.
    Adds or transforms features in the DataFrame.

    :param df: Preprocessed DataFrame
    :return: DataFrame with engineered features
    """
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
    Select top N features based on RandomForest importance.

    :param df: DataFrame including target
    :param target: Target column name
    :param top_n: Number of top features to keep
    :param log_importances: Whether to log the sorted feature importances
    :return: List of top feature names
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
        feat_imp = sorted(zip(features, importances),
                          key=lambda x: x[1], reverse=True)
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



def evaluate_model(y_true, y_pred, model_name="Model"):
    """
    Prints evaluation metrics for a classification model.

    :param y_true: Ground truth labels
    :param y_pred: Model predictions
    :param model_name: Name of the model (for logging)
    """
    logging.info(f"--- {model_name} Evaluation ---")
    logging.info(f"Confusion Matrix:\n{confusion_matrix(y_true, y_pred)}")
    logging.info(f"Classification Report:\n{classification_report(y_true, y_pred, digits=4)}")
    roc_auc = 0
    try:
        roc_auc = roc_auc_score(y_true, y_pred)
    except ValueError as e:
        logging.warning(f"Cannot compute ROC-AUC for {model_name}: {e}")
    logging.info(f"{model_name} ROC-AUC: {roc_auc:.4f}")


def train_and_evaluate_models(X_train, X_test, y_train, y_test):
    """
    Train and evaluate a few supervised and unsupervised models.

    :param X_train, X_test: Feature sets
    :param y_train, y_test: Target sets
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
    logging.info("Evaluating Unsupervised Models (treating outliers or clusters as 'fraud')")

    # 1) Isolation Forest
    try:
        iso_forest = IsolationForest(contamination=0.01, random_state=42)
        iso_forest.fit(X_train)
        y_pred_iso = iso_forest.predict(X_test)
        # Map -1 (anomaly) to 1, otherwise 0
        y_pred_iso = np.where(y_pred_iso == -1, 1, 0)
        evaluate_model(y_test, y_pred_iso, "Isolation Forest")
    except Exception as e:
        logging.error(f"Error training Isolation Forest: {e}")

    # 2) K-Means
    try:
        kmeans = KMeans(n_clusters=2, random_state=42)
        kmeans.fit(X_train)
        y_pred_km = kmeans.predict(X_test)
        # Assume cluster "1" is fraud, cluster "0" is not
        evaluate_model(y_test, y_pred_km, "K-Means")
    except Exception as e:
        logging.error(f"Error training K-Means: {e}")


def hyperparameter_tuning_rf(X_train, X_test, y_train, y_test):
    """
    Hyperparameter tuning for RandomForestClassifier using GridSearchCV.

    :param X_train, X_test, y_train, y_test: Training and Test sets
    :return: The best RandomForest model after tuning
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


def explain_model_shap(model, X_test):
    """
    Generates SHAP explanation plots for a given model and test set.

    :param model: Trained model (tree-based recommended)
    :param X_test: Test features
    """
    if not SHAP_AVAILABLE:
        logging.warning("SHAP not available. Skipping model interpretation.")
        return

    try:
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_test)

        # Summary plot
        shap.summary_plot(shap_values[1], X_test, plot_type="bar", show=False)
        plt.title("SHAP Feature Importance (Bar)")
        plt.show()

        # Optionally, a single prediction explanation
        if len(X_test) > 0:
            sample_index = 0
            shap.force_plot(explainer.expected_value[1],
                            shap_values[1][sample_index,:],
                            X_test.iloc[sample_index,:])
    except Exception as e:
        logging.error(f"Error generating SHAP plots: {e}")


##############################################################################
# 3. MAIN EXECUTION BLOCK (PIPELINE)
##############################################################################
def main():
    # === Step 1: Load Data ===
    file_path = "insurance_claims.csv"  # Or pass as argument
    df = load_dataset(file_path)
    if df.empty:
        logging.error("Exiting due to empty dataset.")
        return  # Exit if we can't proceed

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
