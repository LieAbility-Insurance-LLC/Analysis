import streamlit as st
import pandas as pd
import numpy as np
import joblib
import io
import matplotlib.pyplot as plt
import logging

# Import your custom modules
from data_handling import load_dataset, validate_required_columns
from preprocessing import preprocess_data, eda_plots
from feature_engineering import feature_engineering
from model_training import train_and_evaluate_models
from evaluation import explain_model_shap

from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split

# ----------------------------
# Streamlit Dashboard Settings
# ----------------------------
st.set_page_config(page_title="Insurance Claim Fraud Dashboard", layout="wide")
st.title("Insurance Claim Fraud Detection Dashboard")

# Initialize session state to store data and preprocessed data
if "data" not in st.session_state:
    st.session_state.data = None
if "preprocessed_data" not in st.session_state:
    st.session_state.preprocessed_data = None

# ----------------------------
# Sidebar: File Upload
# ----------------------------
st.sidebar.header("1. Upload CSV File")
uploaded_file = st.sidebar.file_uploader("Upload a CSV file", type=["csv"])
if uploaded_file is not None:
    try:
        data = pd.read_csv(uploaded_file)
        st.session_state.data = data
        st.sidebar.success("File uploaded successfully!")
        st.write("### Uploaded Data Preview")
        st.write(f"Data Shape: {data.shape}")
        st.dataframe(data.head())
    except Exception as e:
        st.sidebar.error(f"Error loading CSV file: {e}")
else:
    st.sidebar.info("Please upload your dataset here.")

# ----------------------------
# Section: Preprocessing
# ----------------------------
st.header("2. Data Preprocessing")
with st.expander("View / Run Preprocessing"):
    if st.session_state.data is not None:
        if st.button("Preprocess Data"):
            try:
                data = st.session_state.data.copy()
                # Preprocess data using your module (handles cleaning, encoding, scaling, etc.)
                processed = preprocess_data(data, target_column="fraud_reported")
                # Apply feature engineering (e.g., is_rush_hour)
                processed = feature_engineering(processed)
                st.session_state.preprocessed_data = processed
                st.success("Data preprocessing completed!")
                st.write("#### Processed Data Preview")
                st.dataframe(processed.head())
            except Exception as e:
                st.error(f"Preprocessing error: {e}")
    else:
        st.info("Upload a CSV file to start preprocessing.")

# ----------------------------
# Section: Model Training
# ----------------------------
st.header("3. Model Training")
with st.expander("Run Model Training"):
    if st.session_state.preprocessed_data is not None:
        if "fraud_reported" not in st.session_state.preprocessed_data.columns:
            st.error("The target column 'fraud_reported' is missing in the processed data.")
        else:
            if st.button("Run Model Training"):
                try:
                    # Split features and target
                    data = st.session_state.preprocessed_data.copy()
                    X = data.drop("fraud_reported", axis=1)
                    y = data["fraud_reported"]

                    # Handle imbalance with SMOTE
                    sm = SMOTE(random_state=42)
                    X_res, y_res = sm.fit_resample(X, y)

                    # Train-Test split
                    X_train, X_test, y_train, y_test = train_test_split(
                        X_res, y_res, test_size=0.2, random_state=42
                    )

                    # Set up log capture to display training logs in the dashboard
                    log_stream = io.StringIO()
                    log_handler = logging.StreamHandler(log_stream)
                    log_handler.setLevel(logging.INFO)
                    logger = logging.getLogger()
                    logger.addHandler(log_handler)

                    # Train and evaluate models (RandomForest by default is among the options)
                    train_and_evaluate_models(X_train, X_test, y_train, y_test)

                    # Remove custom log handler and display captured logs
                    logger.removeHandler(log_handler)
                    st.text_area("Training Logs", log_stream.getvalue(), height=300)
                except Exception as e:
                    st.error(f"Error during training: {e}")
    else:
        st.info("Preprocess the data first to enable training.")

# ----------------------------
# Section: Prediction
# ----------------------------
st.header("4. Prediction")
with st.expander("Run Prediction on Uploaded Data"):
    if st.session_state.preprocessed_data is not None:
        if st.button("Run Prediction"):
            try:
                # Load the pre-trained model (ensure that the model file exists at models/best_rf_model.pkl)
                model = joblib.load("models/best_rf_model.pkl")
                data = st.session_state.preprocessed_data.copy()

                # Drop target column if present
                if "fraud_reported" in data.columns:
                    X_pred = data.drop("fraud_reported", axis=1)
                else:
                    X_pred = data

                predictions = model.predict(X_pred)
                # Add predictions to a copy of the data for display
                pred_df = data.copy()
                pred_df["Prediction"] = predictions
                st.success("Prediction complete!")
                st.write("#### Predictions Preview")
                st.dataframe(pred_df.head())
            except Exception as e:
                st.error(f"Prediction error: {e}")
    else:
        st.info("Preprocess the data first to enable prediction.")

# ----------------------------
# Section: SHAP Explanation
# ----------------------------
st.header("5. SHAP Explanation")
with st.expander("Generate SHAP Plots"):
    if st.session_state.preprocessed_data is not None:
        if st.button("Generate SHAP Explanation"):
            try:
                # Load the pre-trained model
                model = joblib.load("models/best_rf_model.pkl")
                data = st.session_state.preprocessed_data.copy()
                # Use all available features (drop target if present)
                if "fraud_reported" in data.columns:
                    X_explain = data.drop("fraud_reported", axis=1)
                else:
                    X_explain = data

                # Override plt.show to display plots in Streamlit
                old_show = plt.show
                plt.show = lambda: st.pyplot(plt.gcf())

                # Generate SHAP explanation plots (summary bar plot and force plot)
                explain_model_shap(model, X_explain)

                # Restore original plt.show
                plt.show = old_show
            except Exception as e:
                st.error(f"SHAP explanation error: {e}")
    else:
        st.info("Preprocess the data first to generate SHAP explanations.")
