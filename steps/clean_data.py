import logging
import pandas as pd
from zenml import step
from src.data_cleaning import DataProcessor, SplitHandler, PreprocessHandler

from typing import Tuple


@step
def clean_df(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    The function returns a tuple consisting of:
    X_train: Training features
    X_test: Testing features
    y_train: Training target
    y_test: Testing target
    """
    try:
        # Preprocess the data
        processed_data = DataProcessor(df, PreprocessHandler()).execute()
        logging.info("Data cleaning completed")
        
        # Split the data
        return DataProcessor(processed_data, SplitHandler()).execute()
       

    except Exception as e:
        logging.error(f"Error in data cleaning: {e}")
        raise e
