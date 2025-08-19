from prophet import Prophet
import pandas as pd
import numpy as np
from typing import Tuple, Dict
from sklearn import logger
from .base import BaseForecaster
from .utils import get_zim_holidays

class ProphetForecaster(BaseForecaster):
    """Enhanced Prophet model with Zimbabwe-specific adaptations"""
    
    def __init__(self, growth='linear', seasonality_mode='multiplicative'):
        super().__init__()
        self.growth = growth
        self.seasonality_mode = seasonality_mode
        self.holidays = get_zim_holidays()
        self.model = None
    
    def train(self, data: pd.DataFrame) -> None:
        """Train Prophet model with Zimbabwe-specific features"""
        super().train(data)
        
        # Check if data is sufficient
        if len(data) < 10:
            raise ValueError("Insufficient data for Prophet training (need at least 10 points)")
        
        # Prepare Prophet-compatible DataFrame
        df_prophet = self._prepare_prophet_data()
        
        # Check if prepared data is empty
        if df_prophet.empty:
            raise ValueError("Dataframe has no rows")
        
        # Check if we have valid y values
        if df_prophet['y'].isna().all() or (df_prophet['y'] == 0).all():
            raise ValueError("No valid sales data found")
        
        # Initialize and configure model
        self.model = Prophet(
            growth=self.growth,
            yearly_seasonality=False,
            weekly_seasonality=True,
            daily_seasonality=False,
            holidays=self.holidays,
            seasonality_mode=self.seasonality_mode,
            changepoint_prior_scale=0.05,
            holidays_prior_scale=0.15
        )
        
        # Add Zimbabwe-specific components
        self._add_zim_features(df_prophet)
        
        # Fit model
        self.model.fit(df_prophet)
    
    def predict(self, periods: int = 30) -> Tuple[pd.DataFrame, Dict]:
        """Generate forecast with Zimbabwe context"""
        if self.model is None:
            raise RuntimeError("Model not trained. Call train() first.")
        
        if periods <= 0:
            raise ValueError("Periods must be greater than 0")
        
        # Create future DataFrame with regressors
        future = self._create_future_df(periods)
        
        # Generate forecast
        forecast = self.model.predict(future)
        
        # Process forecast results
        forecast_df = self._process_forecast(forecast, periods)
        
        # Calculate metrics
        metrics = self._calculate_historical_metrics(forecast)
        
        return forecast_df, metrics
    
    def _prepare_prophet_data(self) -> pd.DataFrame:
        """Prepare data for Prophet training"""
        # Check if required columns exist
        required_cols = ['date', 'quantity_sold']
        missing_cols = [col for col in required_cols if col not in self.data.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
        
        df = self.data.rename(columns={'date': 'ds', 'quantity_sold': 'y'}).copy()
        
        # Ensure proper data types
        df['ds'] = pd.to_datetime(df['ds'])
        df['y'] = pd.to_numeric(df['y'], errors='coerce')
        
        # Remove rows with invalid dates or y values
        df = df.dropna(subset=['ds', 'y'])
        
        # Add regressors if present
        for col in ['is_payday', 'price_usd', 'school_term']:
            if col in self.data.columns:
                df[col] = pd.to_numeric(self.data[col], errors='coerce').fillna(0)
        
        return df
    
    def _add_zim_features(self, df_prophet: pd.DataFrame) -> None:
        """Add Zimbabwe-specific features to Prophet model"""
        # Monthly seasonality
        self.model.add_seasonality(
            name='monthly',
            period=30.5,
            fourier_order=5
        )
        
        # Payday effect as regressor
        if 'is_payday' in df_prophet.columns:
            self.model.add_regressor('is_payday')
        
        # Price effect
        if 'price_usd' in df_prophet.columns:
            self.model.add_regressor('price_usd')
    
    def _create_future_df(self, periods: int) -> pd.DataFrame:
        """Create future DataFrame with Zimbabwe context"""
        future = self.model.make_future_dataframe(
            periods=periods,
            include_history=False
        )
        
        # Add Zimbabwe-specific regressors to future dates
        future['is_payday'] = (future['ds'].dt.day > 25) | (future['ds'].dt.is_month_end)
        future['is_payday'] = future['is_payday'].astype(int)
        
        if 'price_usd' in self.data.columns:
            last_price = self.data['price_usd'].iloc[-1] if len(self.data) > 0 else 1.0
            future['price_usd'] = last_price
        
        return future
    
    def _process_forecast(self, forecast: pd.DataFrame, periods: int) -> pd.DataFrame:
        """Process Prophet forecast into standard format"""
        # Take only the future periods (not historical)
        forecast = forecast.tail(periods).copy()
        
        return pd.DataFrame({
            'date': forecast['ds'],
            'yhat': np.maximum(0, forecast['yhat']),
            'yhat_lower': np.maximum(0, forecast['yhat_lower']),
            'yhat_upper': np.maximum(0, forecast['yhat_upper'])
        })
    
    def _calculate_historical_metrics(self, forecast: pd.DataFrame) -> Dict:
        """Calculate in-sample metrics"""
        if self.data is None:
            return {}
            
        try:
            # Get historical predictions (full forecast includes history)
            historical = forecast[forecast['ds'].isin(self.data['date'])]
            
            if historical.empty:
                return {}
                
            # Align with actuals
            actuals = self.data.set_index('date')['quantity_sold']
            preds = historical.set_index('ds')['yhat']
            
            common_dates = actuals.index.intersection(preds.index)
            if len(common_dates) == 0:
                return {}
                
            return self._calculate_metrics(
                actuals.loc[common_dates],
                preds.loc[common_dates]
            )
        except Exception as e:
            logger.warning(f"Failed to calculate historical metrics: {e}")
            return {}