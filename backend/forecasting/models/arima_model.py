import pandas as pd
import numpy as np
from statsmodels.tsa.statespace.sarimax import SARIMAX
from typing import Tuple, Dict
import logging
from .base import BaseForecaster

logger = logging.getLogger(__name__)

class ARIMAForecaster(BaseForecaster):
    """Enhanced SARIMAX model for Zimbabwean retail patterns"""
    
    def __init__(self, order=(1,1,1), seasonal_order=(1,1,1,7)):
        super().__init__()
        self.order = order
        self.seasonal_order = seasonal_order
        self.model_fit = None
    
    def train(self, data: pd.DataFrame) -> None:
        """Train SARIMAX model with automatic differencing checks"""
        super().train(data)
        
        # Check if data is sufficient
        if len(data) < 10:
            raise ValueError("Insufficient data for ARIMA training (need at least 10 points)")
        
        # Prepare time series data
        ts_data = self._prepare_time_series()
        
        # Check if time series has enough data
        if len(ts_data) < 5:
            raise ValueError("Insufficient time series data for ARIMA training")
        
        # Auto-detect differencing order if needed
        if self._needs_differencing(ts_data):
            self.order = (self.order[0], min(2, self.order[1]+1), self.order[2])
            logger.info(f"Auto-adjusted differencing order to {self.order}")
        
        try:
            model = SARIMAX(
                ts_data,
                order=self.order,
                seasonal_order=self.seasonal_order,
                enforce_stationarity=False,
                enforce_invertibility=False,
                initialization='approximate_diffuse'
            )
            self.model_fit = model.fit(disp=False)
        except Exception as e:
            logger.error(f"ARIMA training failed: {e}")
            raise RuntimeError(f"ARIMA model training failed: {str(e)}")
    
    def predict(self, periods: int = 30) -> Tuple[pd.DataFrame, Dict]:
        """Generate forecast with dynamic confidence intervals"""
        if self.model_fit is None:
            raise RuntimeError("Model not trained. Call train() first.")
        
        if periods <= 0:
            raise ValueError("Periods must be greater than 0")
        
        try:
            # Get forecast with dynamic confidence intervals
            forecast_obj = self.model_fit.get_forecast(steps=periods)
            forecast_mean = forecast_obj.predicted_mean
            ci = forecast_obj.conf_int(alpha=0.2)  # 80% CI
            
            # Create forecast DataFrame
            forecast_df = self._create_forecast_df(forecast_mean, ci, periods)
            
            # Calculate metrics on historical data
            metrics = self._calculate_historical_metrics()
            
            return forecast_df, metrics
        except Exception as e:
            logger.error(f"ARIMA prediction failed: {e}")
            raise RuntimeError(f"Prediction failed: {str(e)}")
    
    def _prepare_time_series(self) -> pd.Series:
        """Prepare time series data for ARIMA"""
        ts_data = self.data.groupby('date')['quantity_sold'].sum()
        ts_data = ts_data.asfreq('D').fillna(0)
        return ts_data
    
    def _needs_differencing(self, ts_data: pd.Series) -> bool:
        """Check if time series needs differencing"""
        from statsmodels.tsa.stattools import adfuller
        
        # Need at least 10 observations for ADF test
        if len(ts_data.dropna()) < 10:
            return False
            
        try:
            result = adfuller(ts_data.dropna())
            return result[1] > 0.05  # p-value > 0.05 suggests non-stationary
        except Exception:
            return False
    
    def _create_forecast_df(self, forecast_mean, ci, periods) -> pd.DataFrame:
        """Create formatted forecast DataFrame"""
        last_date = self.data['date'].max()
        future_dates = pd.date_range(
            start=last_date + pd.Timedelta(days=1), 
            periods=periods, 
            freq='D'
        )
        
        return pd.DataFrame({
            'date': future_dates,
            'yhat': np.maximum(0, forecast_mean.values),
            'yhat_lower': np.maximum(0, ci.iloc[:, 0].values),
            'yhat_upper': np.maximum(0, ci.iloc[:, 1].values)
        })
    
    def _calculate_historical_metrics(self) -> Dict:
        """Calculate in-sample metrics"""
        if self.data is None or self.model_fit is None:
            return {}
            
        try:
            ts_data = self._prepare_time_series()
            
            # Ensure we have valid date range
            if len(ts_data) < 2:
                return {}
            
            start_date = ts_data.index[0]
            end_date = ts_data.index[-1]
            
            # Ensure start is before end
            if start_date >= end_date:
                return {}
            
            # Get in-sample predictions
            in_sample_pred = self.model_fit.predict(start=start_date, end=end_date)
            
            # Find common dates
            common_dates = ts_data.index.intersection(in_sample_pred.index)
            if len(common_dates) > 0:
                return self._calculate_metrics(
                    ts_data.loc[common_dates],
                    in_sample_pred.loc[common_dates]
                )
            return {}
        except Exception as e:
            logger.warning(f"Failed to calculate historical metrics: {e}")
            return {}