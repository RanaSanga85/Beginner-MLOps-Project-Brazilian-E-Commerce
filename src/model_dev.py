import logging
from abc import ABC, abstractmethod
from sklearn.linear_model import LinearRegression

class Model(ABC):
    @abstractmethod
    def train(self, X_train, y_train):
        pass


class LineraRegressionModel(Model):
    def train(self, X_train, y_train, **kwargs):
        try:
            reg = LineraRegressionModel(**kwargs)
            reg.fit(X_train, y_train)
            logging.info("Model training completed")
