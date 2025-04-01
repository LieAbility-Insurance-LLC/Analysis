import os
import logging
import pandas as pd

def load_dataset(file_path: str) -> pd.DataFrame:
    """
    Loads the dataset into a pandas DataFrame.
    Includes error handling if the file does not exist or is unreadable.
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
    :return: True if all columns exist, False otherwise.
    """
    df_cols = set(df.columns)
    missing = [col for col in required_cols if col not in df_cols]
    if missing:
        logging.error(f"Missing required columns: {missing}")
        return False
    return True
