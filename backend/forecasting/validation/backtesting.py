import pandas as pd
import numpy as np
from typing import Tuple, Dict


class BaseForecaster:
    """Abstract base class for forecasting models."""

    def __init__(self):
        self.model = None
        self.data = None  # Store data here after train method

    def train(self, data: pd.DataFrame) -> None:
        """Trains the forecasting model on the provided data."""
        self.data = data  # Common place to store data for all models
        raise NotImplementedError("Train method must be implemented by subclasses.")

    def predict(self, periods: int) -> Tuple[pd.DataFrame, Dict]:
        """Generates a forecast for the specified number of periods."""
        raise NotImplementedError("Predict method must be implemented by subclasses.")

    def _calculate_metrics(self, actual: pd.Series, forecast: pd.Series) -> Dict:
        """Helper to calculate common forecasting metrics (MAE, RMSE, MAPE)."""
        from sklearn.metrics import mean_absolute_error, mean_squared_error

        # Ensure inputs are Series and aligned by index
        actual = pd.Series(actual).dropna()
        forecast = pd.Series(forecast).dropna()

        common_index = actual.index.intersection(forecast.index)
        if common_index.empty:
            return {'mape': np.nan, 'rmse': np.nan, 'mae': np.nan}

        actual = actual.loc[common_index]
        forecast = forecast.loc[common_index]

        mape = np.mean(np.abs((actual - forecast) / actual)) * 100
        # Handle division by zero for MAPE if actuals contain zeros
        if (actual == 0).any():
            non_zero_actuals = actual[actual != 0]
            non_zero_forecasts = forecast[actual != 0]
            if not non_zero_actuals.empty:
                mape = np.mean(np.abs((non_zero_actuals - non_zero_forecasts) / non_zero_actuals)) * 100
            else:
                mape = np.nan  # If all actuals are zero, MAPE is undefined

        rmse = np.sqrt(mean_squared_error(actual, forecast))
        mae = mean_absolute_error(actual, forecast)

        return {'mape': mape, 'rmse': rmse, 'mae': mae}