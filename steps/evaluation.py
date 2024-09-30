import logging
import pandas as pd
from zenml import step
from src. evaluation import MSE, R2, RMSE
from sklearn.base import RegressorMixin
from typing_extensions import Annotated

@step
def evaluate_model(model: RegressorMixin,
                   X_test: pd.DataFrame,
                   y_test: pd.DataFrame) -> tuple[
                       Annotated[float, "r2_score"],
                       Annotated[float, "rmse"],
                   ]:
    try:
        prediction = model.predict(X_test)

        # Calculate metrics
        r2 = R2().calculate_scores(y_test, prediction)
        rmse = RMSE().calculate_scores(y_test, prediction)

        return r2, rmse  # Corrected return values
    except Exception as e:
        logging.error(f"Error in evaluating model: {e}")
        raise e
