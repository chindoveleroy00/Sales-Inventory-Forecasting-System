import pandas as pd
import numpy as np
from typing import Tuple, Dict
from statsforecast import StatsForecast
from statsforecast.models import TSB
from .base import BaseForecaster
import logging

logger = logging.getLogger(__name__)

class CrostonTSB(BaseForecaster):
    """Enhanced Croston's TSB model for intermittent demand"""
    
    def __init__(self, alpha_d=0.1, alpha_p=0.1):
        super().__init__()
        self.alpha_d = alpha_d  # Demand smoothing parameter
        self.alpha_p = alpha_p  # Probability smoothing parameter
        self.model = None
        self.sf_data = None
    
    def train(self, data: pd.DataFrame) -> None:
        """Train TSB model on intermittent demand data"""
        super().train(data)
        
        # Check if data is sufficient
        if len(data) < 5:
            raise ValueError("Insufficient data for Croston training (need at least 5 points)")
        
        # Prepare data for StatsForecast
        self.sf_data = self._prepare_sf_data()
        
        # Check if prepared data is empty
        if self.sf_data.empty:
            raise ValueError("No valid data for Croston training")
        
        # Check if we have any non-zero values
        if (self.sf_data['y'] == 0).all():
            raise ValueError("All sales values are zero - cannot train Croston model")
        
        # Initialize and fit model
        try:
            self.model = StatsForecast(
                models=[TSB(alpha_d=self.alpha_d, alpha_p=self.alpha_p)],
                freq='D',
                n_jobs=1
            )
            self.model.fit(self.sf_data)
        except Exception as e:
            logger.error(f"Croston training failed: {e}")
            raise RuntimeError(f"Croston model training failed: {str(e)}")
    
    def predict(self, periods: int = 30) -> Tuple[pd.DataFrame, Dict]:
        """Generate forecast for intermittent demand patterns"""
        if self.model is None:
            raise RuntimeError("Model not trained. Call train() first.")
        
        if periods <= 0:
            raise ValueError("Periods must be greater than 0")
        
        try:
            # Generate forecast
            forecast_df = self.model.predict(h=periods)
            
            # Process results
            processed_df = self._process_forecast(forecast_df)
            
            # Calculate metrics if possible
            metrics = self._calculate_metrics_if_possible(processed_df)
            
            return processed_df, metrics
        except Exception as e:
            logger.error(f"Croston prediction failed: {e}")
            raise RuntimeError(f"Croston prediction failed: {str(e)}")
    
    def _prepare_sf_data(self) -> pd.DataFrame:
        """Prepare data for StatsForecast"""
        # Check if required columns exist
        required_cols = ['date', 'quantity_sold', 'sku_id']
        missing_cols = [col for col in required_cols if col not in self.data.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
        
        sf_data = self.data.rename(columns={
            'date': 'ds',
            'quantity_sold': 'y',
            'sku_id': 'unique_id'
        }).copy()
        
        # Ensure proper types
        sf_data['ds'] = pd.to_datetime(sf_data['ds'])
        sf_data['y'] = pd.to_numeric(sf_data['y'], errors='coerce')
        
        # Remove rows with invalid data
        sf_data = sf_data.dropna(subset=['ds', 'y', 'unique_id'])
        
        # Ensure we have at least one unique_id
        if sf_data['unique_id'].nunique() == 0:
            raise ValueError("No valid SKU IDs found")
        
        return sf_data
    
    def _process_forecast(self, forecast_df: pd.DataFrame) -> pd.DataFrame:
        """Process StatsForecast output into standard format"""
        if forecast_df.empty:
            raise ValueError("Empty forecast returned from StatsForecast")
        
        # Reset index if needed
        if forecast_df.index.name == 'ds':
            forecast_df = forecast_df.reset_index()
        
        # Rename columns
        forecast_df = forecast_df.rename(columns={
            'ds': 'date',
            'unique_id': 'sku_id',
            'TSB': 'yhat'
        })
        
        # Add confidence intervals (TSB doesn't provide them natively)
        forecast_df['yhat_lower'] = forecast_df['yhat'] * 0.8
        forecast_df['yhat_upper'] = forecast_df['yhat'] * 1.2
        
        # Ensure non-negative predictions
        forecast_df['yhat'] = np.maximum(0, forecast_df['yhat'])
        forecast_df['yhat_lower'] = np.maximum(0, forecast_df['yhat_lower'])
        forecast_df['yhat_upper'] = np.maximum(0, forecast_df['yhat_upper'])
        
        # Select and order columns
        result_cols = ['date', 'yhat', 'yhat_lower', 'yhat_upper']
        if 'sku_id' in forecast_df.columns:
            result_cols.insert(1, 'sku_id')
        
        return forecast_df[result_cols]
    
    def _calculate_metrics_if_possible(self, forecast_df: pd.DataFrame) -> Dict:
        """Calculate metrics if historical data is available"""
        if self.data is None:
            return {}
            
        try:
            # For Croston, we can use cross-validation approach
            # Split data into train/test
            split_point = int(len(self.data) * 0.8)
            train_data = self.data.iloc[:split_point]
            test_data = self.data.iloc[split_point:]
            
            if len(test_data) < 2:
                return {}
            
            # Train on subset and predict on test period
            temp_model = StatsForecast(
                models=[TSB(alpha_d=self.alpha_d, alpha_p=self.alpha_p)],
                freq='D',
                n_jobs=1
            )
            
            temp_sf_data = train_data.rename(columns={
                'date': 'ds',
                'quantity_sold': 'y',
                'sku_id': 'unique_id'
            })
            temp_sf_data['ds'] = pd.to_datetime(temp_sf_data['ds'])
            temp_sf_data['y'] = pd.to_numeric(temp_sf_data['y'], errors='coerce')
            temp_sf_data = temp_sf_data.dropna(subset=['ds', 'y', 'unique_id'])
            
            if temp_sf_data.empty:
                return {}
            
            temp_model.fit(temp_sf_data)
            test_forecast = temp_model.predict(h=len(test_data))
            
            if test_forecast.empty:
                return {}
            
            # Process and align predictions with actuals
            test_forecast = test_forecast.reset_index() if test_forecast.index.name == 'ds' else test_forecast
            test_forecast = test_forecast.rename(columns={'ds': 'date', 'TSB': 'yhat'})
            
            # Merge with test actuals
            merged = pd.merge(
                test_forecast,
                test_data[['date', 'sku_id', 'quantity_sold']],
                on=['date', 'sku_id'] if 'sku_id' in test_forecast.columns else ['date'],
                how='inner'
            )
            
            if merged.empty:
                return {}
                
            return self._calculate_metrics(
                merged['quantity_sold'],
                merged['yhat']
            )
        except Exception as e:
            logger.warning(f"Failed to calculate Croston metrics: {e}")
            return {}