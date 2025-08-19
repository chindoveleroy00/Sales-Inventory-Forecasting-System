from abc import ABC, abstractmethod
import pandas as pd
from typing import Tuple, Dict
import logging

class BaseForecaster(ABC):
    """Abstract base class for all forecasters in the SIFS system"""
    
    def __init__(self):
        self.data = None
        self.logger = logging.getLogger(__name__)
    
    def train(self, data: pd.DataFrame) -> None:
        """Store training data and perform any preprocessing"""
        if data is None or data.empty:
            raise ValueError("Training data cannot be None or empty")
        self.data = data.copy()
        self._validate_data()
    
    @abstractmethod
    def predict(self, periods: int = 30) -> Tuple[pd.DataFrame, Dict]:
        """Generate forecast with confidence intervals"""
        pass
    
    def _validate_data(self) -> None:
        """Validate required columns in training data"""
        required_cols = {'date', 'quantity_sold', 'sku_id'}
        if not required_cols.issubset(self.data.columns):
            missing = required_cols - set(self.data.columns)
            raise ValueError(f"Missing required columns in training data: {missing}")
    
    def _calculate_metrics(self, actual: pd.Series, predicted: pd.Series) -> Dict:
        """Calculate forecast accuracy metrics"""
        if len(actual) != len(predicted):
            raise ValueError("Actual and predicted series must be same length")
        
        metrics = {
            'mape': self._mean_absolute_percentage_error(actual, predicted),
            'rmse': self._root_mean_squared_error(actual, predicted),
            'bias': (predicted - actual).mean(),
            'accuracy': 100 - self._mean_absolute_percentage_error(actual, predicted)
        }
        return metrics
    
    @staticmethod
    def _mean_absolute_percentage_error(actual: pd.Series, predicted: pd.Series) -> float:
        """Calculate MAPE, handling zero values"""
        mask = actual != 0
        if not mask.any():
            return float('inf')
        return (abs((actual[mask] - predicted[mask]) / actual[mask]).mean() * 100)
    
    @staticmethod
    def _root_mean_squared_error(actual: pd.Series, predicted: pd.Series) -> float:
        """Calculate RMSE"""
        return ((actual - predicted) ** 2).mean() ** 0.5