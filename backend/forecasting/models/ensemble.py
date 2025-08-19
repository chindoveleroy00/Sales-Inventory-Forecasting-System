from typing import Tuple, Dict, List, Optional
import pandas as pd
import numpy as np
from .base import BaseForecaster
import logging

logger = logging.getLogger(__name__)

class EnsembleForecaster(BaseForecaster):
    """Smart ensemble forecaster with dynamic weighting"""
    
    def __init__(self, models: List[BaseForecaster], weights: Optional[List[float]] = None):
        super().__init__()
        self.models = models
        self.weights = self._validate_weights(weights)
        self.model_performance = {}
    
    def train(self, data: pd.DataFrame) -> None:
        """Train all ensemble components and evaluate their performance"""
        super().train(data)
        
        # Check if data is sufficient
        if len(data) < 10:
            raise ValueError("Insufficient data for ensemble training (need at least 10 points)")
        
        # Train each model and evaluate its performance
        successful_models = []
        for model in self.models:
            try:
                model.train(data.copy())
                successful_models.append(model)
                
                # Try to get in-sample metrics for weight adjustment
                try:
                    _, metrics = model.predict(periods=7)  # Short prediction for validation
                    if metrics:
                        self.model_performance[model.__class__.__name__] = metrics.get('mape', 100)
                    else:
                        self.model_performance[model.__class__.__name__] = 100
                except Exception:
                    self.model_performance[model.__class__.__name__] = 100
                    
            except Exception as e:
                logger.warning(f"Model {model.__class__.__name__} training failed: {e}")
                self.model_performance[model.__class__.__name__] = float('inf')
        
        # Update models list to only include successful ones
        self.models = successful_models
        
        if not self.models:
            raise RuntimeError("All ensemble models failed to train")
        
        # Adjust weights based on performance
        self._adjust_weights()
    
    def predict(self, periods: int = 30) -> Tuple[pd.DataFrame, Dict]:
        """Generate ensemble forecast with dynamic weighting"""
        if periods <= 0:
            raise ValueError("Periods must be greater than 0")
        
        forecasts = []
        all_metrics = {}
        
        # Collect predictions from all models
        for model in self.models:
            try:
                forecast_df, metrics = model.predict(periods)
                
                # Get model weight (default to 0 if not found)
                model_name = model.__class__.__name__
                weight = self.weights.get(model_name, 0)
                
                # Apply weight to forecast
                forecast_df = forecast_df.copy()
                forecast_df['model_weight'] = weight
                forecasts.append(forecast_df)
                all_metrics[model_name] = metrics
            except Exception as e:
                logger.warning(f"Model {model.__class__.__name__} prediction failed: {e}")
        
        if not forecasts:
            raise RuntimeError("All ensemble models failed to predict")
        
        # Combine forecasts using weights
        combined_forecast = self._combine_forecasts(forecasts)
        
        # Calculate ensemble metrics if possible
        ensemble_metrics = self._calculate_ensemble_metrics(combined_forecast)
        all_metrics['Ensemble'] = ensemble_metrics
        
        return combined_forecast, all_metrics
    
    def _validate_weights(self, weights: Optional[List[float]]) -> Dict[str, float]:
        """Validate and normalize weights"""
        if weights is None:
            # Default equal weights
            return {m.__class__.__name__: 1/len(self.models) for m in self.models}
        
        if len(weights) != len(self.models):
            raise ValueError("Number of weights must match number of models")
        
        # Normalize weights to sum to 1
        total = sum(weights)
        if total <= 0:
            raise ValueError("Weights must sum to a positive value")
            
        return {
            m.__class__.__name__: w/total 
            for m, w in zip(self.models, weights)
        }
    
    def _adjust_weights(self) -> None:
        """Adjust weights based on model performance"""
        if not self.model_performance:
            return
            
        # Filter out models that failed completely
        valid_performance = {k: v for k, v in self.model_performance.items() 
                           if v != float('inf')}
        
        if not valid_performance:
            # If all models failed, use equal weights
            self.weights = {m.__class__.__name__: 1/len(self.models) for m in self.models}
            return
            
        # Convert MAPE to weights (lower MAPE = higher weight)
        # Add small epsilon to avoid division by zero
        inverse_mape = {k: 1/(v + 1e-6) for k, v in valid_performance.items()}
        total = sum(inverse_mape.values())
        
        if total > 0:
            # Update weights only for models that have performance metrics
            for model_name in inverse_mape:
                self.weights[model_name] = inverse_mape[model_name] / total
    
    def _combine_forecasts(self, forecasts: List[pd.DataFrame]) -> pd.DataFrame:
        """Combine forecasts using weighted average"""
        if not forecasts:
            return pd.DataFrame()
        
        # Combine all forecasts
        combined = pd.concat(forecasts, ignore_index=True)
        
        # Check if we have the required columns
        required_cols = ['date', 'yhat', 'yhat_lower', 'yhat_upper', 'model_weight']
        missing_cols = [col for col in required_cols if col not in combined.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns in forecasts: {missing_cols}")
        
        # Group by date and calculate weighted averages
        # Fix for the pandas FutureWarning
        def weighted_average(group):
            return pd.Series({
                'yhat': np.average(group['yhat'], weights=group['model_weight']),
                'yhat_lower': np.average(group['yhat_lower'], weights=group['model_weight']),
                'yhat_upper': np.average(group['yhat_upper'], weights=group['model_weight'])
            })
        
        # REMOVED 'include_groups=False' for broader pandas compatibility
        weighted_avg = combined.groupby('date').apply(weighted_average).reset_index()
        
        # Ensure non-negative predictions
        weighted_avg['yhat'] = np.maximum(0, weighted_avg['yhat'])
        weighted_avg['yhat_lower'] = np.maximum(0, weighted_avg['yhat_lower'])
        weighted_avg['yhat_upper'] = np.maximum(0, weighted_avg['yhat_upper'])
        
        return weighted_avg
    
    def _calculate_ensemble_metrics(self, forecast: pd.DataFrame) -> Dict:
        """Calculate metrics for ensemble forecast"""
        if self.data is None or forecast.empty:
            return {}
            
        try:
            # Get historical period from forecast (if any)
            historical = forecast[forecast['date'].isin(self.data['date'])]
            
            if historical.empty:
                return {}
                
            # Merge with actuals
            merged = pd.merge(
                historical,
                self.data[['date', 'quantity_sold']],
                on='date',
                how='inner'
            )
            
            if merged.empty:
                return {}
                
            return self._calculate_metrics(
                merged['quantity_sold'],
                merged['yhat']
            )
        except Exception as e:
            logger.warning(f"Failed to calculate ensemble metrics: {e}")
            return {}