import logging
from abc import ABC, abstractmethod
from typing import Union

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


class DataHandler(ABC):
    @abstractmethod
    def process(self, data: pd.DataFrame) -> Union[pd.DataFrame, pd.Series]:
        pass


class PreprocessHandler(DataHandler):
    """
    This class implements the handle_data method to perform preprocessing tasks on a DataFrame.
    """
    def process(self, data: pd.DataFrame) -> pd.DataFrame:
        try:
            columns_to_drop = [
                "order_approved_at", "order_delivered_carrier_date",
                "order_delivered_customer_date", "order_estimate_delivery_date",
                "order_purchase_timestamp"
            ]
            data = data.drop(columns=columns_to_drop, axis=1)

            # Filling missing values with the median for numerical columns
            num_cols = ["product_weight_g", "product_length_cm", "product_height_cm", "product_width_cm"]
            for col in num_cols:
                data[col].fillna(data[col].median(), inplace=True)
            
            # Keep only numerical columns and drop unnecessary ones
            data = data.select_dtypes(include=[np.number]).drop(["customer_zip_code_prefix", "order_item_id"], axis=1) 
            return data
        except Exception as e:
            logging.error(f"Error during data split: {e}")
            raise e
        

class SplitHandler(DataHandler):
    """
    Splits the dataset into training and testing sets.
    """
    def process(self, data: pd.DataFrame) -> Union[pd.DataFrame, pd.Series]:
        
        try:
            X = data.drop(["review_score"], axis=1)
            y = data["review_score"]
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            return X_train, X_test, y_train, y_test
        except Exception as e:
            logging.error(f"Error during data split: {e}")
            raise e
        
# Data Processor to apply strategies
class DataProcessor:
    def __init__(self, data:pd.DataFrame, strategy:DataHandler):
        self.data = data
        self.strategy = strategy #An instance of a class that implements the DataStrategy interface.


    def execute(self) -> Union[pd.DataFrame, pd.Series]:
        """
          Executes the given strategy (preprocessing or splitting).
        """
        try:
            return self.strategy.process(self.data)
        except Exception as e:
            logging.error(f"Error during data split: {e}")
            raise e