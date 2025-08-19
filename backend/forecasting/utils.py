"""
Utility functions for the SIFS forecasting system
"""
import pandas as pd
from typing import Dict, List
import logging

logger = logging.getLogger(__name__)

def get_zim_product_weights() -> Dict[str, Dict[str, float]]:
    """
    Get product-specific model weights for Zimbabwe market
    
    Returns:
        Dict mapping product categories to model weights
    """
    # Default weights for Zimbabwe market based on product characteristics
    product_weights = {
        'default': {
            'prophet': 0.4,
            'arima': 0.3,
            'croston': 0.3
        },
        'staples': {  # Mealie meal, rice, sugar
            'prophet': 0.5,
            'arima': 0.3,
            'croston': 0.2
        },
        'seasonal': {  # Seasonal vegetables, fruits
            'prophet': 0.6,
            'arima': 0.2,
            'croston': 0.2
        },
        'fast_moving': {  # High turnover items
            'prophet': 0.4,
            'arima': 0.4,
            'croston': 0.2
        },
        'slow_moving': {  # Low turnover items
            'prophet': 0.2,
            'arima': 0.2,
            'croston': 0.6
        }
    }
    
    return product_weights

def categorize_product(sku_id: str) -> str:
    """
    Categorize product based on SKU ID
    
    Args:
        sku_id: Product SKU identifier
        
    Returns:
        Product category string
    """
    sku_lower = sku_id.lower()
    
    # Staple foods
    if any(keyword in sku_lower for keyword in ['mealie', 'rice', 'sugar', 'flour', 'bread']):
        return 'staples'
    
    # Seasonal products
    if any(keyword in sku_lower for keyword in ['tomato', 'onion', 'potato', 'cabbage', 'fruit']):
        return 'seasonal'
    
    # Default category
    return 'default'

def prepare_time_series(data: pd.DataFrame, sku_id: str = None) -> pd.DataFrame:
    """
    Prepare time series data for forecasting
    
    Args:
        data: Raw sales data
        sku_id: Optional SKU filter
        
    Returns:
        Prepared time series data
    """
    df = data.copy()
    
    # Filter by SKU if specified
    if sku_id:
        df = df[df['sku_id'] == sku_id]
    
    # Ensure date column is datetime
    df['date'] = pd.to_datetime(df['date'])
    
    # Sort by date
    df = df.sort_values('date')
    
    # Fill missing dates with zero sales
    if not df.empty:
        date_range = pd.date_range(start=df['date'].min(), end=df['date'].max(), freq='D')
        df = df.set_index('date').reindex(date_range, fill_value=0).reset_index()
        df.rename(columns={'index': 'date'}, inplace=True)
    
    return df

def detect_anomalies(data: pd.Series, threshold: float = 3.0) -> pd.Series:
    """
    Detect anomalies in time series data using z-score
    
    Args:
        data: Time series data
        threshold: Z-score threshold for anomaly detection
        
    Returns:
        Boolean series indicating anomalies
    """
    z_scores = abs((data - data.mean()) / data.std())
    return z_scores > threshold

def calculate_seasonal_indices(data: pd.Series, period: int = 7) -> Dict[int, float]:
    """
    Calculate seasonal indices for time series
    
    Args:
        data: Time series data
        period: Seasonal period (default 7 for weekly)
        
    Returns:
        Dictionary mapping period indices to seasonal factors
    """
    if len(data) < period * 2:
        logger.warning("Insufficient data for seasonal index calculation")
        return {i: 1.0 for i in range(period)}
    
    # Calculate moving average
    moving_avg = data.rolling(window=period, center=True).mean()
    
    # Calculate seasonal ratios
    seasonal_ratios = data / moving_avg
    
    # Calculate seasonal indices
    seasonal_indices = {}
    for i in range(period):
        period_ratios = seasonal_ratios.iloc[i::period].dropna()
        if not period_ratios.empty:
            seasonal_indices[i] = period_ratios.mean()
        else:
            seasonal_indices[i] = 1.0
    
    return seasonal_indices

def validate_forecast_input(data: pd.DataFrame) -> bool:
    """
    Validate input data for forecasting
    
    Args:
        data: Input DataFrame
        
    Returns:
        True if valid, raises ValueError if invalid
    """
    required_columns = {'date', 'quantity_sold', 'sku_id'}
    
    if data.empty:
        raise ValueError("Input data cannot be empty")
    
    if not required_columns.issubset(data.columns):
        missing = required_columns - set(data.columns)
        raise ValueError(f"Missing required columns: {missing}")
    
    if data['quantity_sold'].isnull().any():
        raise ValueError("quantity_sold cannot contain null values")
    
    if data['quantity_sold'].min() < 0:
        raise ValueError("quantity_sold cannot contain negative values")
    
    return True

def get_forecast_horizon(sku_id: str, default_horizon: int = 30) -> int:
    """
    Get appropriate forecast horizon based on product characteristics
    
    Args:
        sku_id: Product SKU identifier
        default_horizon: Default forecast horizon
        
    Returns:
        Forecast horizon in days
    """
    category = categorize_product(sku_id)
    
    horizon_map = {
        'staples': 45,      # Longer horizon for staples
        'seasonal': 21,     # Shorter for seasonal items
        'fast_moving': 30,  # Standard for fast-moving
        'slow_moving': 60,  # Longer for slow-moving
        'default': default_horizon
    }
    
    return horizon_map.get(category, default_horizon)