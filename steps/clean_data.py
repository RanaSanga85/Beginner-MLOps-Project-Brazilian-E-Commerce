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
        processed_data = PreprocessHandler()
        data_cleaning = DataProcessor(df, processed_data)
        processed_data = data_cleaning.execute()

        # Split the data
        split_data = SplitHandler()
        data_cleaning = DataProcessor(processed_data, split_data)
        X_train, X_test, y_train, y_test = data_cleaning.execute()
        
        logging.info("Data cleaning completed")
        
        return X_train, X_test, y_train, y_test
       

    except Exception as e:
        logging.error(f"Error in data cleaning: {e}")
        raise e
