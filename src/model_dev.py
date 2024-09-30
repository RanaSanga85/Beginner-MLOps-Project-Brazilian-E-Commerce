import logging
from abc import ABC, abstractmethod
from sklearn.linear_model import LinearRegression

class Model(ABC):
    """
    Model is an abstract base class, which means it defines a template for any model that inherits from it
    """
    @abstractmethod
    def train(self, X_train, y_train):
        pass

class LinearRegressionModel(Model):
    def train(self, X_train, y_train, **kwargs):
        try:
            reg = LinearRegression(**kwargs) 
            reg.fit(X_train, y_train)
            logging.info("Model training completed")
            return reg
        except Exception as e:
            logging.error(f"Error in training model: {e}")
            raise e

