import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
from pathlib import Path
import yaml
import holidays

def load_config(config_path: str) -> Dict:
    """Load configuration with Zimbabwe-specific defaults"""
    with open(config_path) as f:
        config = yaml.safe_load(f)
    
    # Add Zimbabwe holidays
    config['zim_holidays'] = get_zim_holidays()
    return config

def get_zim_holidays(year: int = 2025) -> List[str]:
    """Get Zimbabwe public holidays as DataFrame for Prophet"""
    zim_holidays = holidays.Zimbabwe(years=[year])
    holiday_df = pd.DataFrame([
        (date, name) for date, name in zim_holidays.items()
    ], columns=['ds', 'holiday'])
    
    # Add custom impact estimates
    holiday_df['lower_window'] = -1  # Day before holiday
    holiday_df['upper_window'] = 1   # Day after holiday
    
    # Set specific impacts for major holidays
    major_holidays = {
        'New Year': 1.5,
        'Independence Day': 2.0,
        'Christmas': 2.5
    }
    
    holiday_df['impact'] = holiday_df['holiday'].map(
        lambda x: major_holidays.get(x, 1.2)
    )
    
    return holiday_df

def get_zim_product_weights() -> Dict:
    """Optimal model weights by Zimbabwean product category"""
    return {
        'Staple': {'prophet': 0.6, 'arima': 0.3, 'croston': 0.1},
        'Beverage': {'prophet': 0.7, 'arima': 0.2, 'croston': 0.1},
        'Household': {'prophet': 0.5, 'arima': 0.4, 'croston': 0.1},
        'Personal Care': {'prophet': 0.4, 'arima': 0.3, 'croston': 0.3},
        'Snack': {'prophet': 0.7, 'arima': 0.2, 'croston': 0.1},
        'Bakery': {'prophet': 0.8, 'arima': 0.1, 'croston': 0.1},
        'default': {'prophet': 0.6, 'arima': 0.3, 'croston': 0.1}
    }

def add_zim_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add Zimbabwe-specific features to DataFrame"""
    df = df.copy()
    
    # Ensure date is datetime and set as index
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date')
    
    # Add economic indicators
    df = _add_economic_indicators(df)
    
    # Add school terms
    df = _add_school_terms(df)
    
    # Add payday flag
    df['is_payday'] = (df['date'].dt.day > 25) | (df['date'].dt.is_month_end)
    
    # Add holiday flag
    zim_holidays = get_zim_holidays()
    df['is_holiday'] = df['date'].isin(zim_holidays['ds'])
    
    # Add lag features
    df = _add_lag_features(df)
    
    return df

def _add_economic_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Add Zimbabwe economic indicators"""
    # Simulated exchange rates (in practice, use real data)
    exchange_rates = {
        '2025-01': 1.0, '2025-02': 1.05, '2025-03': 1.1,
        '2025-04': 1.12, '2025-05': 1.15, '2025-06': 1.2,
        '2025-07': 1.25, '2025-08': 1.3, '2025-09': 1.35,
        '2025-10': 1.4, '2025-11': 1.45, '2025-12': 1.5
    }
    
    df['exchange_rate'] = df['date'].dt.strftime('%Y-%m').map(exchange_rates).fillna(1.0)
    df['price_usd'] = df['price'] / df['exchange_rate']
    
    return df

def _add_school_terms(df: pd.DataFrame) -> pd.DataFrame:
    """Add Zimbabwe school term indicators"""
    school_terms = [
        ('2025-01-15', '2025-04-05'),  # Term 1
        ('2025-05-06', '2025-08-09'),  # Term 2
        ('2025-09-10', '2025-12-15')   # Term 3
    ]
    
    df['school_term'] = 'Holiday'
    for i, (start, end) in enumerate(school_terms, 1):
        mask = (df['date'] >= start) & (df['date'] <= end)
        df.loc[mask, 'school_term'] = f'Term{i}'
    
    return df

def _add_lag_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add lagged demand features"""
    df = df.sort_values(['sku_id', 'date'])
    
    # Add lag features
    for lag in [1, 7, 14, 30]:
        df[f'lag_{lag}'] = df.groupby('sku_id')['quantity_sold'].shift(lag)
    
    # Add rolling features
    df['rolling_avg_7d'] = df.groupby('sku_id')['quantity_sold'].transform(
        lambda x: x.rolling(7, min_periods=1).mean()
    )
    df['rolling_std_7d'] = df.groupby('sku_id')['quantity_sold'].transform(
        lambda x: x.rolling(7, min_periods=1).std()
    )
    
    return df