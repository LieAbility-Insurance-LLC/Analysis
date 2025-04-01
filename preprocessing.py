import logging
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler, LabelEncoder

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
    """
    if df.empty:
        logging.warning("Received empty DataFrame for preprocessing. Skipping.")
        return df

    # Convert target from 'Y'/'N' to 1/0 if it’s object
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

    # Outlier capping for 'umbrella_limit'
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
    """
    if df.empty:
        logging.warning("EDA skipped. DataFrame is empty.")
        return

    # Class distribution
    plt.figure(figsize=(6, 4))
    sns.countplot(x=target_column, data=df)
    plt.title('Class Distribution')
    plt.show()

    # Correlation matrix
    plt.figure(figsize=(12, 8))
    sns.heatmap(df.corr(), cmap='coolwarm', annot=False)
    plt.title('Correlation Heatmap')
    plt.show()
