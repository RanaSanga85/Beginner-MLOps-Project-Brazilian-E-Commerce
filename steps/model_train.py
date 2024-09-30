"""import logging
import pandas as pd
from zenml import step
from src.model_dev import LinearRegressionModel

#mixin from sklearn indicating the method returns a regression model.
from sklearn.base import RegressorMixin 

#stores configuration details for selecting the model.
from .config import ModelNameConfig



@step
def train_model(
    X_train: pd.DataFrame,
    X_tests: pd.DataFrame,
    y_train: pd.DataFrame,
    y_test: pd.DataFrame,
    config: ModelNameConfig,
) -> RegressorMixin:
    
    Trains a machine learning regression model based on the provided configuration.

    Args:
        X_train (pd.DataFrame): Training feature data.
        y_train (pd.Series): Training target data.
        config (ModelNameConfig): Configuration specifying the model to train.

    Returns:
        RegressorMixin: The trained regression model.
   

    try:
        model=None
        if config.model_name == "LinearRegression":
            model = LinearRegressionModel()
            trained_model = model.train(X_train, y_train)
            logging.info("Linear Regression model trained successfully.")
            return trained_model
        else:
            raise ValueError("Model {} not supporting".format(config.model_name))
    except Exception as e:
        logging.error(f"Error in training model: {e}")
        raise e

"""
import logging
import pandas as pd
from zenml import step
from src.model_dev import LinearRegressionModel
from sklearn.base import RegressorMixin
from .config import ModelNameConfig

@step
def train_model(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    config: ModelNameConfig,
) -> RegressorMixin:
    """
    Trains a machine learning regression model based on the provided configuration.

    Args:
        X_train (pd.DataFrame): Training feature data.
        y_train (pd.Series): Training target data.
        config (ModelNameConfig): Configuration specifying the model to train.

    Returns:
        RegressorMixin: The trained regression model.
    """
    try:
        model_dict = {
            "LinearRegression": LinearRegressionModel,
            # Add more models here as needed
        }

        if config.model_name in model_dict:
            model = model_dict[config.model_name]()
            trained_model = model.train(X_train, y_train)
            logging.info(f"{config.model_name} model trained successfully.")
            return trained_model
        else:
            raise ValueError(f"Model {config.model_name} not supported")
    
    except Exception as e:
        logging.error(f"Error in training model: {e}")
        raise e



   